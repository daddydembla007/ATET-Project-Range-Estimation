"""
Regen Potential + EV Range API  (dual-route: FAST & ECO)
=========================================================
Features per window (100 samples @ 10 m = 1 km/window):
  1. speed_mps              — OSRM per-node annotations
  2. Gradient               — Open-Elevation (rise/run)
  3. Speed Limit[km/h]      — Overpass API maxspeed
  4. Elevation Smoothed[m]  — Open-Elevation + moving avg

Dual-route logic:
  OSRM ?alternatives=true returns up to 3 routes.
  All alternatives are processed through the full pipeline.
  FAST  = shortest duration among all alternatives
  ECO   = lowest cumulative net energy consumption
  (They may be the same route if OSRM returns only one.)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, math, asyncio, os
import numpy as np
from typing import List, Dict, Optional, Tuple

app = FastAPI(title="Regen + Range API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

HEADERS             = {"User-Agent": "RegenPotentialApp/1.0 (research)"}
DEFAULT_SPEED_LIMIT = 50.0
DEFAULT_ELEVATION   = 0.0
WINDOW_SIZE         = 100
RESAMPLE_SPACING_M  = 10.0


# ─────────────────────────────────────────────────────────
# MODEL + SCALER
# ─────────────────────────────────────────────────────────
_model  = None
_scaler = None

def load_model_and_scaler():
    global _model, _scaler
    if os.path.exists("feature_scaler.pkl"):
        import joblib
        _scaler = joblib.load("feature_scaler.pkl")
        print(f"[INFO] Scaler loaded  mean={_scaler.mean_}  scale={_scaler.scale_}")
    else:
        print("[WARN] feature_scaler.pkl not found — no normalisation")

    if os.path.exists("energy_lstm_checkpoint.pt"):
        import torch
        ckpt = torch.load("energy_lstm_checkpoint.pt", map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model_cfg  = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
        input_size  = model_cfg.get("input_size",  4)
        hidden_size = model_cfg.get("hidden_size", 64)
        num_layers  = model_cfg.get("num_layers",  2)
        output_size = model_cfg.get("output_size", 1)
        for k, v in state_dict.items():
            if "lstm.weight_ih_l0" in k:
                hidden_size = v.shape[0] // 4; break

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
        print(f"[INFO] LSTM loaded  hidden={hidden_size}  layers={num_layers}")
    else:
        print("[WARN] energy_lstm_checkpoint.pt not found — mock model active")

@app.on_event("startup")
async def startup():
    load_model_and_scaler()


# ─────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────
def run_model(raw_windows: List[np.ndarray]) -> List[float]:
    if not raw_windows:
        return []
    X = np.array(raw_windows, dtype=np.float32)
    N, W, F = X.shape
    if _scaler is not None:
        X = _scaler.transform(X.reshape(-1, F)).reshape(N, W, F).astype(np.float32)
    if _model is not None:
        import torch
        with torch.no_grad():
            return [float(v) for v in _model(torch.tensor(X)).squeeze(1).numpy()]
    # mock fallback
    return [round(max(-0.05, min(0.5,
        0.05 + float(np.mean(w[:,0]))*0.003 + float(np.mean(w[:,1]))*10.0)), 5)
        for w in raw_windows]


# ─────────────────────────────────────────────────────────
# REGEN LABEL
# ─────────────────────────────────────────────────────────
def regen_label(e: float) -> str:
    if e < -0.005: return "high_regen"
    if e <= 0.02:  return "low"
    return "consuming"


# ─────────────────────────────────────────────────────────
# RANGE EXHAUSTION
# ─────────────────────────────────────────────────────────
def compute_range(windows, energy_per_win, total_energy_kwh,
                  battery_capacity_kwh) -> Tuple[List[Dict], Optional[Dict], float]:
    remaining = total_energy_kwh
    annotated, exhaustion = [], None
    for i, (win, e) in enumerate(zip(windows, energy_per_win)):
        if exhaustion:
            annotated.append({**win,"energy_kwh":e,"remaining_kwh":0.0,"reachable":False})
            continue
        new_rem = min(remaining - e, battery_capacity_kwh)
        if new_rem <= 0:
            frac    = max(0.0, min(1.0, remaining/e if e>0 else 1.0))
            coords  = win["coords"]; n = len(coords)
            idxf    = frac*(n-1); lo=int(idxf); hi=min(lo+1,n-1); t=idxf-lo
            ex_lat  = coords[lo][0]*(1-t)+coords[hi][0]*t
            ex_lon  = coords[lo][1]*(1-t)+coords[hi][1]*t
            prev_d  = sum(w["dist_m"] for w in windows[:i])
            exhaustion = {
                "window_id": i,
                "coord":     [round(ex_lat,6), round(ex_lon,6)],
                "range_km":  round((prev_d + win["dist_m"]*frac)/1000, 2),
                "fraction_into_window": round(frac,3),
            }
            remaining = 0.0
        else:
            remaining = new_rem
        annotated.append({**win,"energy_kwh":round(e,5),
                          "remaining_kwh":round(remaining,3),"reachable":True})
    return annotated, exhaustion, round(remaining/battery_capacity_kwh*100, 1)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def haversine_m(c1, c2) -> float:
    R=6_371_000
    la1,lo1=math.radians(c1[0]),math.radians(c1[1])
    la2,lo2=math.radians(c2[0]),math.radians(c2[1])
    dla,dlo=la2-la1,lo2-lo1
    a=math.sin(dla/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
    return R*2*math.asin(math.sqrt(a))

def smooth_elevations(elevs, k=5):
    out=[]
    for i in range(len(elevs)):
        lo=max(0,i-k); hi=min(len(elevs),i+k+1)
        out.append(round(sum(elevs[lo:hi])/(hi-lo),2))
    return out

def parse_speed_limit(raw):
    if not raw: return DEFAULT_SPEED_LIMIT
    raw=raw.strip().lower()
    table={"walk":7.0,"living_street":10.0,"urban":50.0,
           "rural":90.0,"motorway":120.0,"none":130.0,"signals":50.0}
    if raw in table: return table[raw]
    if "mph" in raw:
        try: return round(float(raw.replace("mph","").strip())*1.60934,1)
        except: return DEFAULT_SPEED_LIMIT
    try: return float(raw.split(";")[0].strip())
    except: return DEFAULT_SPEED_LIMIT

def lerp(a,b,t): return a+(b-a)*t


# ─────────────────────────────────────────────────────────
# API 1 — NOMINATIM
# ─────────────────────────────────────────────────────────
async def geocode(query: str) -> Dict:
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as c:
        r = await c.get("https://nominatim.openstreetmap.org/search",
                        params={"q":query,"format":"json","limit":1})
        d = r.json()
    if not d: raise HTTPException(400, f"Could not geocode: {query}")
    return {"lat":float(d[0]["lat"]),"lon":float(d[0]["lon"]),
            "display_name":d[0]["display_name"]}


# ─────────────────────────────────────────────────────────
# API 2 — OSRM  (with alternatives)
# Returns list of raw route dicts, each with coords+speeds
# ─────────────────────────────────────────────────────────
async def get_routes(origin: Dict, dest: Dict) -> List[Dict]:
    """
    Requests up to 3 alternative routes from OSRM.
    Returns a list of dicts, each with:
      coords, node_speeds, distance_m, duration_s
    """
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{origin['lon']},{origin['lat']};{dest['lon']},{dest['lat']}"
        f"?overview=full&geometries=geojson"
        f"&steps=true&annotations=true&alternatives=true"
    )
    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
        r = await c.get(url)
        data = r.json()
    if data.get("code") != "Ok":
        raise HTTPException(500, f"OSRM: {data.get('message','unknown')}")

    results = []
    for route in data.get("routes", []):
        coords = [[pt[1],pt[0]] for pt in route["geometry"]["coordinates"]]
        node_speeds: List[float] = []
        for leg in route["legs"]:
            ann = leg.get("annotation",{})
            for dur,dist in zip(ann.get("duration",[]),ann.get("distance",[])):
                node_speeds.append(round(dist/dur if dur>0 else 0.0, 4))
        if not node_speeds:
            avg = route["distance"]/max(route["duration"],1)
            node_speeds = [round(avg,4)]*len(coords)
        else:
            node_speeds.append(node_speeds[-1])
        results.append({
            "coords":     coords,
            "node_speeds":node_speeds,
            "distance_m": route["distance"],
            "duration_s": route["duration"],
        })
    return results


# ─────────────────────────────────────────────────────────
# API 3 — OPEN-ELEVATION
# ─────────────────────────────────────────────────────────
async def get_elevations(coords: List[List[float]]) -> List[float]:
    step    = max(1, len(coords)//100)
    sampled = coords[::step]
    locs    = [{"latitude":c[0],"longitude":c[1]} for c in sampled]
    try:
        async with httpx.AsyncClient(timeout=40, headers=HEADERS) as c:
            r = await c.post("https://api.open-elevation.com/api/v1/lookup",
                             json={"locations":locs})
            d = r.json()
        es = [x["elevation"] for x in d["results"]]
    except:
        return [DEFAULT_ELEVATION]*len(coords)
    full=[]; ratio=len(es)/len(coords)
    for i in range(len(coords)):
        lo=int(i*ratio); hi=min(lo+1,len(es)-1); t=(i*ratio)-lo
        full.append(round(es[lo]*(1-t)+es[hi]*t,2))
    return full


# ─────────────────────────────────────────────────────────
# API 4 — OVERPASS (speed limits)
# ─────────────────────────────────────────────────────────
async def get_speed_limits(coords: List[List[float]]) -> List[float]:
    lats=[c[0] for c in coords]; lons=[c[1] for c in coords]
    bb=(min(lats)-.01,min(lons)-.01,max(lats)+.01,max(lons)+.01)
    q=(f"[out:json][timeout:25];"
       f"way[highway][maxspeed]({bb[0]},{bb[1]},{bb[2]},{bb[3]});"
       f"out body geom;")
    try:
        async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
            r = await c.post("https://overpass-api.de/api/interpreter",
                             data={"data":q})
            d = r.json()
    except:
        return [DEFAULT_SPEED_LIMIT]*len(coords)
    ways=[]
    for e in d.get("elements",[]):
        if e.get("type")!="way": continue
        g=e.get("geometry",[]); 
        if not g: continue
        spd=parse_speed_limit(e.get("tags",{}).get("maxspeed"))
        ways.append((sum(n["lat"] for n in g)/len(g),
                     sum(n["lon"] for n in g)/len(g), spd))
    if not ways: return [DEFAULT_SPEED_LIMIT]*len(coords)
    result=[]
    for coord in coords:
        bd,bs=float("inf"),DEFAULT_SPEED_LIMIT
        for wla,wlo,ws in ways:
            d2=haversine_m(coord,[wla,wlo])
            if d2<bd: bd,bs=d2,ws
        result.append(bs)
    return result


# ─────────────────────────────────────────────────────────
# API 5 — OVERPASS (charging stations)
# ─────────────────────────────────────────────────────────
async def get_charging_stations(center, radius_m=10000, max_results=5):
    lat,lon = center
    q=(f"[out:json][timeout:20];"
       f"node[amenity=charging_station](around:{radius_m},{lat},{lon});"
       f"out body;")
    try:
        async with httpx.AsyncClient(timeout=25, headers=HEADERS) as c:
            r = await c.post("https://overpass-api.de/api/interpreter",
                             data={"data":q})
            d = r.json()
    except: return []
    stations=[]
    for e in d.get("elements",[]):
        if e.get("type")!="node": continue
        slat,slon=e.get("lat"),e.get("lon")
        if slat is None: continue
        t=e.get("tags",{})
        stations.append({
            "name":    t.get("name") or t.get("operator") or t.get("network","EV Charging Station"),
            "network": t.get("network",t.get("operator","")),
            "sockets": t.get("capacity",t.get("charging_station:output","?")),
            "coord":   [round(slat,6),round(slon,6)],
            "dist_km": round(haversine_m(center,[slat,slon])/1000,2),
        })
    stations.sort(key=lambda s:s["dist_km"])
    return stations[:max_results]


# ─────────────────────────────────────────────────────────
# RESAMPLING  (fixed 10 m spacing)
# ─────────────────────────────────────────────────────────
def resample_to_fixed_spacing(coords, elev_raw, elev_smooth,
                               node_speeds, speed_limits,
                               spacing_m=RESAMPLE_SPACING_M):
    samples=[]
    for i in range(len(coords)-1):
        seg_dist=haversine_m(coords[i],coords[i+1])
        if seg_dist<0.1: continue
        n_sub=max(1,int(round(seg_dist/spacing_m)))
        act=seg_dist/n_sub
        grad=round((elev_raw[i+1]-elev_raw[i])/seg_dist,6)
        for k in range(n_sub):
            t=k/n_sub
            samples.append({
                "speed_mps":       node_speeds[i],
                "gradient":        grad,
                "speed_limit_kmh": speed_limits[i],
                "elevation_m":     round(lerp(elev_smooth[i],elev_smooth[i+1],t),2),
                "dist_m":          round(act,3),
                "coord":           [round(lerp(coords[i][0],coords[i+1][0],t),7),
                                    round(lerp(coords[i][1],coords[i+1][1],t),7)],
            })
    return samples


# ─────────────────────────────────────────────────────────
# WINDOWING
# ─────────────────────────────────────────────────────────
def window_samples(samples, window_size=WINDOW_SIZE):
    windows=[]
    for i in range(0,len(samples),window_size):
        chunk=samples[i:i+window_size]
        if len(chunk)<window_size//2: continue
        n=len(chunk)
        fm=np.array([[s["speed_mps"],s["gradient"],
                      s["speed_limit_kmh"],s["elevation_m"]] for s in chunk],
                    dtype=np.float32)
        if n<window_size:
            fm=np.vstack([fm,np.tile(fm[-1],(window_size-n,1))])
        windows.append({
            "window_idx":     len(windows),
            "feature_matrix": fm,
            "feature_labels": {
                "speed_mps":       round(float(np.mean(fm[:n,0])),3),
                "gradient":        round(float(np.mean(fm[:n,1])),5),
                "speed_limit_kmh": round(float(np.mean(fm[:n,2])),1),
                "elevation_m":     round(float(np.mean(fm[:n,3])),1),
            },
            "dist_m":          round(sum(s["dist_m"] for s in chunk),1),
            "coords":          [s["coord"] for s in chunk],
            "n_samples_actual":n,
        })
    return windows


# ─────────────────────────────────────────────────────────
# PROCESS ONE ROUTE  (full pipeline for a single OSRM route)
# Returns a complete route result dict ready for the response
# ─────────────────────────────────────────────────────────
async def process_route(
    raw_route: Dict,
    total_energy_kwh: float,
    battery_capacity_kwh: float,
) -> Dict:
    coords      = raw_route["coords"]
    node_speeds = raw_route["node_speeds"]

    # Elevation + speed limits in parallel
    elev_raw, speed_limits = await asyncio.gather(
        get_elevations(coords),
        get_speed_limits(coords),
    )
    elev_smooth = smooth_elevations(elev_raw)

    # Resample → window → model → range
    samples = resample_to_fixed_spacing(coords, elev_raw, elev_smooth,
                                        node_speeds, speed_limits)
    if not samples:
        return None

    n_samples = len(samples)
    windows   = window_samples(samples, WINDOW_SIZE)
    if not windows:
        return None

    energy_per_win = run_model([w["feature_matrix"] for w in windows])

    annotated, exhaustion, final_soc = compute_range(
        windows, energy_per_win, total_energy_kwh, battery_capacity_kwh
    )
    for win, e in zip(annotated, energy_per_win):
        win["regen_label"] = regen_label(e)

    total_km       = raw_route["distance_m"] / 1000
    total_consumed = sum(e for e in energy_per_win if e > 0)
    total_regen    = abs(sum(e for e in energy_per_win if e < 0))
    net_energy     = total_consumed - total_regen
    reachable_km   = exhaustion["range_km"] if exhaustion else total_km

    # Charging stations
    charge_center  = exhaustion["coord"] if exhaustion else None
    charging       = await get_charging_stations(
        center  = charge_center or [coords[-1][0], coords[-1][1]],
        radius_m= 10_000, max_results=5
    )

    # Elevation + speed profiles for charts
    step=max(1,n_samples//80)
    cum,elev_profile=[],[]
    dist_acc=0.0
    for i in range(0,n_samples,step):
        if i>0: dist_acc+=samples[i]["dist_m"]/1000
        elev_profile.append({"dist_km":round(dist_acc,2),
                              "elevation":samples[i]["elevation_m"]})
    dist_acc2=0.0
    spd_profile=[]
    for i in range(0,n_samples,step):
        if i>0: dist_acc2+=samples[i]["dist_m"]/1000
        spd_profile.append({"dist_km":round(dist_acc2,2),
                             "speed_limit_kmh":samples[i]["speed_limit_kmh"]})

    return {
        "route": {
            "distance_km":  round(total_km, 2),
            "duration_min": round(raw_route["duration_s"]/60, 1),
            "full_coords":  coords,
        },
        "resampling": {
            "spacing_m":       RESAMPLE_SPACING_M,
            "total_samples":   n_samples,
            "window_size":     WINDOW_SIZE,
            "km_per_window":   round(RESAMPLE_SPACING_M*WINDOW_SIZE/1000, 2),
        },
        "windows": [
            {
                "window_id":     w["window_idx"],
                "regen_label":   w["regen_label"],
                "energy_kwh":    w["energy_kwh"],
                "remaining_kwh": w["remaining_kwh"],
                "reachable":     w["reachable"],
                "coords":        w["coords"],
                "dist_m":        w["dist_m"],
                "features":      w["feature_labels"],
                "n_samples":     w["n_samples_actual"],
            }
            for w in annotated
        ],
        "exhaustion": exhaustion,
        "charging_stations": {
            "context":   "exhaustion_point" if exhaustion else "destination",
            "center":    charge_center or [coords[-1][0],coords[-1][1]],
            "radius_km": 10,
            "stations":  charging,
        },
        "summary": {
            "total_windows":       len(windows),
            "reachable_km":        round(reachable_km, 2),
            "route_completed":     exhaustion is None,
            "final_soc_pct":       final_soc if exhaustion is None else 0.0,
            "total_consumed_kwh":  round(total_consumed, 3),
            "total_regen_kwh":     round(total_regen, 3),
            "net_energy_kwh":      round(net_energy, 3),
            "high_regen_windows":  sum(1 for w in annotated
                                       if w["regen_label"]=="high_regen"),
            "avg_speed_limit_kmh": round(sum(speed_limits)/len(speed_limits),1),
        },
        "elevation_profile": elev_profile,
        "speed_profile":     spd_profile,
        # internal — used for labelling, not returned to client
        "_net_energy":  net_energy,
        "_duration_s":  raw_route["duration_s"],
    }


# ─────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────
class RouteRequest(BaseModel):
    origin:               str
    destination:          str
    soc_percent:          float = 80.0
    battery_capacity_kwh: float = 60.0


# ─────────────────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────────────────
@app.post("/api/route")
async def analyze_route(req: RouteRequest):
    if not (0 < req.soc_percent <= 100):
        raise HTTPException(400, "soc_percent must be 1–100")
    if req.battery_capacity_kwh <= 0:
        raise HTTPException(400, "battery_capacity_kwh must be > 0")

    # 1. Geocode
    origin_geo, dest_geo = await asyncio.gather(
        geocode(req.origin), geocode(req.destination)
    )

    # 2. Get all OSRM alternatives (up to 3)
    raw_routes = await get_routes(origin_geo, dest_geo)
    if not raw_routes:
        raise HTTPException(500, "OSRM returned no routes")

    # 3. Process each alternative through the full pipeline (in parallel)
    total_energy_kwh = (req.soc_percent / 100.0) * req.battery_capacity_kwh
    processed = await asyncio.gather(*[
        process_route(r, total_energy_kwh, req.battery_capacity_kwh)
        for r in raw_routes
    ])
    processed = [p for p in processed if p is not None]
    if not processed:
        raise HTTPException(500, "All routes failed processing")

    # 4. Label FAST and ECO
    #   FAST = minimum duration
    #   ECO  = minimum net energy consumption
    #   If only one route exists, both labels point to it.
    fast_idx = min(range(len(processed)), key=lambda i: processed[i]["_duration_s"])
    eco_idx  = min(range(len(processed)), key=lambda i: processed[i]["_net_energy"])

    # Clean internal keys before returning
    for p in processed:
        p.pop("_net_energy", None)
        p.pop("_duration_s", None)

    # Build response — always return fast & eco (may be same route index)
    fast_result = processed[fast_idx]
    eco_result  = processed[eco_idx]

    # Comparison stats
    same_route = (fast_idx == eco_idx)
    energy_saving = round(
        fast_result["summary"]["net_energy_kwh"] -
        eco_result["summary"]["net_energy_kwh"], 3
    ) if not same_route else 0.0
    time_cost = round(
        eco_result["route"]["duration_min"] -
        fast_result["route"]["duration_min"], 1
    ) if not same_route else 0.0

    return {
        "origin":      origin_geo,
        "destination": dest_geo,
        "n_alternatives": len(processed),
        "same_route":     same_route,   # true when OSRM found only 1 route
        "comparison": {
            "energy_saving_kwh": energy_saving,   # ECO saves this vs FAST
            "time_cost_min":     time_cost,        # ECO takes this longer than FAST
        },
        "fast": {**fast_result,
                 "label":"FAST",
                 "soc_input": req.soc_percent,
                 "battery_kwh_input": req.battery_capacity_kwh},
        "eco":  {**eco_result,
                 "label":"ECO",
                 "soc_input": req.soc_percent,
                 "battery_kwh_input": req.battery_capacity_kwh},
    }


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
