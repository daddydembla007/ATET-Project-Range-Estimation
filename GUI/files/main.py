"""
Regen Potential + EV Range API
==============================
Features per window (100 segments, matching training):
  1. speed_mps              — OSRM per-node annotations
  2. Gradient               — Open-Elevation API (rise/run)
  3. Speed Limit[km/h]      — Overpass API maxspeed per segment
  4. Elevation Smoothed[m]  — Open-Elevation API (smoothed)

Model: energy_lstm_checkpoint.pt  (PyTorch LSTM)
Scaler: feature_scaler.pkl        (sklearn StandardScaler, fit on train set)

Range logic:
  total_energy_kwh = SOC * battery_capacity_kwh
  For each window:
    if energy_consumption > 0 → subtract from total_energy_kwh
    if energy_consumption < 0 → regen, add back (capped at full battery)
    if total_energy_kwh <= 0 → range exhausted, mark last coord of prev window
                               + find exact exhaustion point within window
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import math
import asyncio
import os
import numpy as np
from typing import List, Dict, Optional, Tuple

app = FastAPI(title="Regen + Range API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

HEADERS            = {"User-Agent": "RegenPotentialApp/1.0 (research)"}
DEFAULT_SPEED_LIMIT = 50.0
DEFAULT_ELEVATION   = 0.0
WINDOW_SIZE         = 100          # must match training


# ─────────────────────────────────────────────────────────
# MODEL + SCALER LOADING
# Place energy_lstm_checkpoint.pt and feature_scaler.pkl
# in the same directory as main.py.
# ─────────────────────────────────────────────────────────
_model  = None
_scaler = None

def load_model_and_scaler():
    global _model, _scaler

    # ── Scaler ──────────────────────────────────────────
    scaler_path = "feature_scaler.pkl"
    if os.path.exists(scaler_path):
        import joblib
        _scaler = joblib.load(scaler_path)
        print(f"[INFO] Scaler loaded from {scaler_path}")
    else:
        print("[WARN] feature_scaler.pkl not found — using identity (no scaling)")

    # ── PyTorch LSTM ─────────────────────────────────────
    model_path = "energy_lstm_checkpoint.pt"
    if os.path.exists(model_path):
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")

        # Support both raw state_dict saves and full checkpoint dicts
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict  = checkpoint["model_state_dict"]
            model_cfg   = checkpoint.get("model_config", {})
        elif isinstance(checkpoint, dict) and any(k.startswith("lstm") for k in checkpoint):
            state_dict = checkpoint
            model_cfg  = {}
        else:
            state_dict = checkpoint
            model_cfg  = {}

        # Infer architecture from state_dict keys
        input_size  = model_cfg.get("input_size",  4)
        hidden_size = model_cfg.get("hidden_size", 64)
        num_layers  = model_cfg.get("num_layers",  2)
        output_size = model_cfg.get("output_size", 1)

        # Try to read hidden_size from weight shape
        for k, v in state_dict.items():
            if "lstm.weight_ih_l0" in k:
                hidden_size = v.shape[0] // 4
                break

        class EnergyLSTM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size,
                                          num_layers, batch_first=True)
                self.fc   = torch.nn.Linear(hidden_size, output_size)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        net = EnergyLSTM()
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        _model = net
        print(f"[INFO] LSTM loaded — hidden={hidden_size}, layers={num_layers}")
    else:
        print("[WARN] energy_lstm_checkpoint.pt not found — using mock model")


@app.on_event("startup")
async def startup():
    load_model_and_scaler()


# ─────────────────────────────────────────────────────────
# INFERENCE
# Input  : raw_windows  — list of (WINDOW_SIZE, 4) arrays
# Output : energy_kwh per window (denormalised, signed)
#          positive  = consuming energy
#          negative  = regenerating energy
# ─────────────────────────────────────────────────────────
def run_model(raw_windows: List[np.ndarray]) -> List[float]:
    """
    raw_windows : list of np.ndarray, each shape (WINDOW_SIZE, 4)
    Feature order: [speed_mps, Gradient, Speed_Limit_kmh, Elevation_m]
    """
    if not raw_windows:
        return []

    X = np.array(raw_windows, dtype=np.float32)   # (N, 100, 4)
    N, W, F = X.shape

    # ── Normalise ────────────────────────────────────────
    if _scaler is not None:
        X_2d     = X.reshape(-1, F)
        X_scaled = _scaler.transform(X_2d).reshape(N, W, F).astype(np.float32)
    else:
        X_scaled = X

    # ── Inference ────────────────────────────────────────
    if _model is not None:
        import torch
        with torch.no_grad():
            t    = torch.tensor(X_scaled)
            pred = _model(t).squeeze(1).numpy()   # shape (N,)
        # pred is in normalised target space; denormalise if scaler carries
        # y stats (it doesn't here — target scaler must be separate if used)
        return [float(v) for v in pred]

    # ── Mock fallback ────────────────────────────────────
    results = []
    for win in raw_windows:
        speed   = float(np.mean(win[:, 0]))
        grad    = float(np.mean(win[:, 1]))
        # positive gradient / higher speed → more consumption
        energy  = 0.05 + speed * 0.003 + grad * 10.0
        energy  = max(-0.05, min(0.5, energy))   # kWh per window
        results.append(round(energy, 5))
    return results


# ─────────────────────────────────────────────────────────
# REGEN CLASSIFICATION  (for map colouring)
# energy_kwh < -0.005  → regenerating (high regen)
# energy_kwh ∈ [-0.005, 0.02] → low consumption
# energy_kwh > 0.02   → consuming
# ─────────────────────────────────────────────────────────
def regen_label(energy_kwh: float) -> str:
    if energy_kwh < -0.005:
        return "high_regen"
    if energy_kwh <= 0.02:
        return "low"
    return "consuming"


# ─────────────────────────────────────────────────────────
# RANGE EXHAUSTION
# Returns (windows_with_energy, exhaustion_info_or_None)
# ─────────────────────────────────────────────────────────
def compute_range(
    windows: List[Dict],
    energy_kwh_per_window: List[float],
    total_energy_kwh: float,
    battery_capacity_kwh: float,
) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Walk through windows, subtracting/adding energy.
    Returns annotated windows and exhaustion point if it occurs.
    """
    remaining = total_energy_kwh
    annotated = []
    exhaustion = None

    for i, (win, e_kwh) in enumerate(zip(windows, energy_kwh_per_window)):
        if exhaustion:
            # Already exhausted — mark remaining windows as unreachable
            annotated.append({**win, "energy_kwh": e_kwh,
                               "remaining_kwh": 0.0, "reachable": False})
            continue

        new_remaining = remaining - e_kwh
        # Cap at full battery (regen can't exceed capacity)
        new_remaining = min(new_remaining, battery_capacity_kwh)

        if new_remaining <= 0:
            # Exhaustion happens inside this window
            # Find fraction of window covered before running out
            if e_kwh > 0:
                fraction = remaining / e_kwh   # 0-1: how far into window
            else:
                fraction = 1.0
            fraction = max(0.0, min(1.0, fraction))

            coords = win["coords"]
            n_pts  = len(coords)
            # Interpolate exhaustion coord
            idx_f  = fraction * (n_pts - 1)
            idx_lo = int(idx_f)
            idx_hi = min(idx_lo + 1, n_pts - 1)
            t      = idx_f - idx_lo
            ex_lat = coords[idx_lo][0] * (1-t) + coords[idx_hi][0] * t
            ex_lon = coords[idx_lo][1] * (1-t) + coords[idx_hi][1] * t

            # Distance from route start to exhaustion point
            # = sum of all previous window distances + fraction of this window
            prev_dist = sum(w["dist_m"] for w in windows[:i])
            this_dist = win["dist_m"] * fraction
            total_range_m = prev_dist + this_dist

            exhaustion = {
                "window_id":   i,
                "coord":       [round(ex_lat, 6), round(ex_lon, 6)],
                "range_km":    round(total_range_m / 1000, 2),
                "fraction_into_window": round(fraction, 3),
            }
            remaining = 0.0
        else:
            remaining = new_remaining

        annotated.append({
            **win,
            "energy_kwh":    round(e_kwh, 5),
            "remaining_kwh": round(remaining, 3),
            "reachable":     True,
        })

    final_soc = remaining / battery_capacity_kwh if battery_capacity_kwh > 0 else 0.0

    return annotated, exhaustion, round(final_soc * 100, 1)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def haversine_m(c1, c2) -> float:
    R = 6_371_000
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def smooth_elevations(elevs: List[float], k: int = 5) -> List[float]:
    """Simple moving-average smoothing (matches 'Elevation Smoothed[m]')."""
    out = []
    for i in range(len(elevs)):
        lo  = max(0, i - k)
        hi  = min(len(elevs), i + k + 1)
        out.append(round(sum(elevs[lo:hi]) / (hi - lo), 2))
    return out


def parse_speed_limit(raw: Optional[str]) -> float:
    if not raw:
        return DEFAULT_SPEED_LIMIT
    raw = raw.strip().lower()
    table = {"walk": 7.0, "living_street": 10.0, "urban": 50.0,
             "rural": 90.0, "motorway": 120.0, "none": 130.0, "signals": 50.0}
    if raw in table:
        return table[raw]
    if "mph" in raw:
        try:
            return round(float(raw.replace("mph","").strip()) * 1.60934, 1)
        except ValueError:
            return DEFAULT_SPEED_LIMIT
    try:
        return float(raw.split(";")[0].strip())
    except ValueError:
        return DEFAULT_SPEED_LIMIT


# ─────────────────────────────────────────────────────────
# API 1 — NOMINATIM
# ─────────────────────────────────────────────────────────
async def geocode(query: str) -> Dict:
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as c:
        r    = await c.get("https://nominatim.openstreetmap.org/search",
                           params={"q": query, "format": "json", "limit": 1})
        data = r.json()
    if not data:
        raise HTTPException(400, f"Could not geocode: {query}")
    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]),
            "display_name": data[0]["display_name"]}


# ─────────────────────────────────────────────────────────
# API 2 — OSRM  (speed_mps per node from annotations)
# ─────────────────────────────────────────────────────────
async def get_route(origin: Dict, dest: Dict) -> Dict:
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{origin['lon']},{origin['lat']};{dest['lon']},{dest['lat']}"
        f"?overview=full&geometries=geojson&steps=true&annotations=true"
    )
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
        r    = await c.get(url)
        data = r.json()
    if data.get("code") != "Ok":
        raise HTTPException(500, f"OSRM: {data.get('message','unknown')}")

    route  = data["routes"][0]
    coords = [[pt[1], pt[0]] for pt in route["geometry"]["coordinates"]]

    node_speeds: List[float] = []
    for leg in route["legs"]:
        ann  = leg.get("annotation", {})
        for dur, dist in zip(ann.get("duration",[]), ann.get("distance",[])):
            node_speeds.append(round(dist/dur if dur>0 else 0.0, 4))

    if not node_speeds:
        avg = route["distance"] / max(route["duration"], 1)
        node_speeds = [round(avg, 4)] * len(coords)
    else:
        node_speeds.append(node_speeds[-1])

    return {"coords": coords, "node_speeds": node_speeds,
            "distance_m": route["distance"], "duration_s": route["duration"]}


# ─────────────────────────────────────────────────────────
# API 3 — OPEN-ELEVATION  (Gradient + Elevation Smoothed)
# ─────────────────────────────────────────────────────────
async def get_elevations(coords: List[List[float]]) -> List[float]:
    step      = max(1, len(coords) // 100)
    sampled   = coords[::step]
    locations = [{"latitude": c[0], "longitude": c[1]} for c in sampled]
    try:
        async with httpx.AsyncClient(timeout=40, headers=HEADERS) as c:
            r    = await c.post("https://api.open-elevation.com/api/v1/lookup",
                                json={"locations": locations})
            data = r.json()
        elev_s = [res["elevation"] for res in data["results"]]
    except Exception:
        return [DEFAULT_ELEVATION] * len(coords)

    full: List[float] = []
    ratio = len(elev_s) / len(coords)
    for i in range(len(coords)):
        lo = int(i * ratio)
        hi = min(lo + 1, len(elev_s) - 1)
        t  = (i * ratio) - lo
        full.append(round(elev_s[lo]*(1-t) + elev_s[hi]*t, 2))
    return full


# ─────────────────────────────────────────────────────────
# API 4 — OVERPASS  (Speed Limit per segment)
# ─────────────────────────────────────────────────────────
async def get_speed_limits(coords: List[List[float]]) -> List[float]:
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    bb   = (min(lats)-0.01, min(lons)-0.01, max(lats)+0.01, max(lons)+0.01)
    q    = (f"[out:json][timeout:25];"
            f"way[highway][maxspeed]({bb[0]},{bb[1]},{bb[2]},{bb[3]});"
            f"out body geom;")
    try:
        async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
            r    = await c.post("https://overpass-api.de/api/interpreter",
                                data={"data": q})
            data = r.json()
    except Exception:
        return [DEFAULT_SPEED_LIMIT] * len(coords)

    ways: List[Tuple] = []
    for elem in data.get("elements", []):
        if elem.get("type") != "way":
            continue
        geom = elem.get("geometry", [])
        if not geom:
            continue
        spd  = parse_speed_limit(elem.get("tags",{}).get("maxspeed"))
        clat = sum(n["lat"] for n in geom) / len(geom)
        clon = sum(n["lon"] for n in geom) / len(geom)
        ways.append((clat, clon, spd))

    if not ways:
        return [DEFAULT_SPEED_LIMIT] * len(coords)

    result: List[float] = []
    for coord in coords:
        best_spd, best_d = DEFAULT_SPEED_LIMIT, float("inf")
        for wlat, wlon, wspd in ways:
            d = haversine_m(coord, [wlat, wlon])
            if d < best_d:
                best_d, best_spd = d, wspd
        result.append(best_spd)
    return result


# ─────────────────────────────────────────────────────────
# SEGMENT BUILDING
# Each segment = one coord pair → 4 raw feature values
# ─────────────────────────────────────────────────────────
def build_segments(coords, elevations, elevations_smoothed,
                   node_speeds, speed_limits) -> List[Dict]:
    segs = []
    for i in range(len(coords) - 1):
        dist = haversine_m(coords[i], coords[i+1])
        if dist < 0.5:
            continue
        gradient = (elevations[i+1] - elevations[i]) / dist
        segs.append({
            # raw feature values (in training column order)
            "speed_mps":        node_speeds[i],
            "gradient":         round(gradient, 6),
            "speed_limit_kmh":  speed_limits[i],
            "elevation_m":      elevations_smoothed[i],
            # metadata
            "dist_m":           round(dist, 2),
            "coord_start":      coords[i],
            "coord_end":        coords[i+1],
        })
    return segs


# ─────────────────────────────────────────────────────────
# WINDOWING  (window_size = 100 matching training)
# ─────────────────────────────────────────────────────────
def window_segments(segs: List[Dict], window_size: int = WINDOW_SIZE) -> List[Dict]:
    windows = []
    for i in range(0, len(segs), window_size):
        chunk = segs[i : i + window_size]
        if len(chunk) < window_size // 2:   # skip tiny tail windows
            continue
        n = len(chunk)

        # Feature matrix for this window: shape (n, 4) — training column order
        feature_matrix = np.array([
            [s["speed_mps"], s["gradient"], s["speed_limit_kmh"], s["elevation_m"]]
            for s in chunk
        ], dtype=np.float32)

        # Pad to exactly WINDOW_SIZE if shorter (tail window)
        if n < window_size:
            pad = np.tile(feature_matrix[-1], (window_size - n, 1))
            feature_matrix = np.vstack([feature_matrix, pad])

        avg_speed = float(np.mean(feature_matrix[:n, 0]))
        avg_grad  = float(np.mean(feature_matrix[:n, 1]))
        avg_limit = float(np.mean(feature_matrix[:n, 2]))
        avg_elev  = float(np.mean(feature_matrix[:n, 3]))

        win_coords = [chunk[0]["coord_start"]] + [s["coord_end"] for s in chunk]
        windows.append({
            "window_idx":     len(windows),
            "feature_matrix": feature_matrix,   # (WINDOW_SIZE, 4) for model
            "feature_labels": {
                "speed_mps":       round(avg_speed, 3),
                "gradient":        round(avg_grad, 5),
                "speed_limit_kmh": round(avg_limit, 1),
                "elevation_m":     round(avg_elev, 1),
            },
            "dist_m":    round(sum(s["dist_m"] for s in chunk), 1),
            "coords":    win_coords,
        })
    return windows


# ─────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────
class RouteRequest(BaseModel):
    origin:               str
    destination:          str
    soc_percent:          float = 80.0    # State of Charge 0-100
    battery_capacity_kwh: float = 60.0   # Usable battery capacity in kWh


# ─────────────────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────────────────
@app.post("/api/route")
async def analyze_route(req: RouteRequest):

    if not (0 < req.soc_percent <= 100):
        raise HTTPException(400, "soc_percent must be 1-100")
    if req.battery_capacity_kwh <= 0:
        raise HTTPException(400, "battery_capacity_kwh must be > 0")

    # ── 1. Geocode ────────────────────────────────────────
    origin_geo, dest_geo = await asyncio.gather(
        geocode(req.origin), geocode(req.destination)
    )

    # ── 2. Route (OSRM) ───────────────────────────────────
    route       = await get_route(origin_geo, dest_geo)
    coords      = route["coords"]
    node_speeds = route["node_speeds"]

    # ── 3. Elevation + Speed limits (parallel) ────────────
    elevations_raw, speed_limits = await asyncio.gather(
        get_elevations(coords),
        get_speed_limits(coords)
    )
    elevations_smoothed = smooth_elevations(elevations_raw)

    # ── 4. Segments ───────────────────────────────────────
    segs = build_segments(coords, elevations_raw, elevations_smoothed,
                          node_speeds, speed_limits)
    if not segs:
        raise HTTPException(500, "No valid route segments found")

    # ── 5. Windows (size=100) ─────────────────────────────
    windows = window_segments(segs, WINDOW_SIZE)
    if not windows:
        raise HTTPException(500, "Route too short to form windows")

    # ── 6. Model inference ────────────────────────────────
    raw_wins      = [w["feature_matrix"] for w in windows]
    energy_per_win = run_model(raw_wins)   # kWh per window, signed

    # ── 7. Range computation ──────────────────────────────
    total_energy_kwh = (req.soc_percent / 100.0) * req.battery_capacity_kwh
    annotated_windows, exhaustion, final_soc_pct = compute_range(
        windows, energy_per_win, total_energy_kwh, req.battery_capacity_kwh
    )

    # ── 8. Regen classification ───────────────────────────
    # For map colouring: based on energy sign & magnitude
    for i, (win, e) in enumerate(zip(annotated_windows, energy_per_win)):
        win["regen_label"] = regen_label(e)

    # ── 9. Summary ────────────────────────────────────────
    total_km       = route["distance_m"] / 1000
    total_consumed = sum(e for e in energy_per_win if e > 0)
    total_regen    = abs(sum(e for e in energy_per_win if e < 0))
    reachable_km   = exhaustion["range_km"] if exhaustion else total_km

    # ── 10. Elevation + Speed profiles for charts ─────────
    step = max(1, len(elevations_raw) // 80)
    cum, elev_profile = 0.0, []
    for i in range(0, len(coords)-1, step):
        if i > 0:
            cum += haversine_m(coords[i-1], coords[i]) / 1000
        elev_profile.append({"dist_km": round(cum, 2),
                              "elevation": round(elevations_smoothed[i], 1)})

    step2 = max(1, len(speed_limits) // 80)
    cum2, spd_profile = 0.0, []
    for i in range(0, len(coords)-1, step2):
        if i > 0:
            cum2 += haversine_m(coords[i-1], coords[i]) / 1000
        spd_profile.append({"dist_km": round(cum2, 2),
                             "speed_limit_kmh": speed_limits[i]})

    return {
        "origin":      origin_geo,
        "destination": dest_geo,
        "route": {
            "distance_km":  round(total_km, 2),
            "duration_min": round(route["duration_s"] / 60, 1),
            "full_coords":  coords,
        },
        "windows": [
            {
                "window_id":    w["window_idx"],
                "regen_label":  w["regen_label"],
                "energy_kwh":   w["energy_kwh"],
                "remaining_kwh": w["remaining_kwh"],
                "reachable":    w["reachable"],
                "coords":       w["coords"],
                "dist_m":       w["dist_m"],
                "features":     w["feature_labels"],
            }
            for w in annotated_windows
        ],
        "exhaustion":  exhaustion,   # null if route completes within battery
        "summary": {
            "total_windows":      len(windows),
            "reachable_km":       round(reachable_km, 2),
            "route_completed":    exhaustion is None,
            "final_soc_pct":      final_soc_pct if exhaustion is None else 0.0,
            "total_consumed_kwh": round(total_consumed, 3),
            "total_regen_kwh":    round(total_regen, 3),
            "net_energy_kwh":     round(total_consumed - total_regen, 3),
            "high_regen_windows": sum(1 for w in annotated_windows
                                      if w["regen_label"] == "high_regen"),
            "avg_speed_limit_kmh": round(sum(speed_limits)/len(speed_limits), 1),
            "soc_input":          req.soc_percent,
            "battery_kwh_input":  req.battery_capacity_kwh,
        },
        "elevation_profile": elev_profile,
        "speed_profile":     spd_profile,
    }


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
