# Road-Condition-Aware EV Range Prediction

This project implements a topography-aware electric vehicle (EV) range prediction and smart route optimization system using deep learning and live mapping APIs. It predicts real-world EV energy consumption by incorporating road elevation, speed profiles, and regenerative braking potential, enabling users to make smarter navigation decisions through Fast and Eco routing modes.

---

## Project Overview

Traditional EV range estimators often ignore terrain variation and regenerative braking opportunities, leading to inaccurate range predictions. This project addresses that challenge by integrating:

- **LSTM-based energy prediction**
- **Road gradient and elevation awareness**
- **Regenerative braking efficiency modeling**
- **Real-time route analysis using live APIs**
- **Interactive GUI for user navigation**

The system helps users determine:

- Whether their EV can reach the selected destination
- Which route is faster
- Which route is more energy efficient
- Estimated battery depletion or regeneration throughout the journey

---

## Core Features

### Topography-Aware Range Prediction
- Dynamically evaluates:
  - Elevation gain/loss
  - Road slopes
  - Gradient profiles
  - Regenerative braking zones
- Provides more realistic EV battery consumption estimates

---

### Dual-Mode Routing System

#### Fast Mode
- Prioritizes minimum travel time
- Uses standard OSRM shortest-time routing
- Best for urgent travel scenarios

#### Eco Mode
- Prioritizes maximum battery efficiency
- Uses LSTM predictions combined with:
  - Terrain analysis
  - Regenerative braking potential
  - Speed constraints
- Extends practical EV driving range

---

### LSTM Time-Series Prediction Model
- 3-layer stacked LSTM architecture
- 256 hidden units
- Fully connected prediction layers:
```math
256 \rightarrow 128 \rightarrow 64 \rightarrow 1
