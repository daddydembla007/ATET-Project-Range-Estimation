# Road-Condition-Aware EV Range Prediction

This project implements a topography-aware electric vehicle (EV) range prediction and smart route optimization system using deep learning and live mapping APIs. It predicts real-world EV energy consumption by incorporating road elevation, speed profiles, and regenerative braking potential, enabling users to make smarter navigation decisions through Fast and Eco routing modes.

## Project Overview

Traditional EV range estimators often ignore terrain variation and regenerative braking opportunities, leading to inaccurate range predictions. This project addresses that challenge by integrating:

- LSTM-based energy prediction
- Road gradient and elevation awareness
- Regenerative braking efficiency modeling
- Real-time route analysis using live APIs
- Interactive GUI for user navigation

The system helps users determine:

- Whether their EV can reach the selected destination
- Which route is faster
- Which route is more energy efficient
- Estimated battery depletion or regeneration throughout the journey

## Core Features

### Topography-Aware Range Prediction

Dynamically evaluates:

- Elevation gain/loss
- Road slopes
- Gradient profiles
- Regenerative braking zones

Provides more realistic EV battery consumption estimates.

### Dual-Mode Routing System

**Fast Mode**

- Prioritizes minimum travel time
- Uses standard OSRM shortest-time routing
- Best for urgent travel scenarios

**Eco Mode**

- Prioritizes maximum battery efficiency
- Uses LSTM predictions combined with terrain analysis, regenerative braking potential, and speed constraints
- Extends practical EV driving range

### LSTM Time-Series Prediction Model

- 3-layer stacked LSTM architecture
- 256 hidden units
- Fully connected prediction layers: 256 → 128 → 64 → 1
- R² Score: 0.86
- Approximate Prediction Accuracy: 86%

### Live API Integration

The project integrates multiple external APIs for real-world deployment:

- **OSRM API** — Route generation, distance calculation, duration prediction
- **Open-Elevation API** — Terrain elevation data and gradient profiling
- **Overpass API** — Speed limit extraction and road metadata

### Interactive GUI

Users can input source, destination, and initial State of Charge (SOC).

Outputs include:

- Fastest route
- Eco-optimized route
- Predicted energy consumption
- Remaining battery percentage
- Regenerative braking gains
- Vehicle stopping point if range is insufficient

## Project Structure
├── gui new/                           # Final GUI/web application source code
├── old codes/                        # Legacy scripts and previous versions
├── poster presentation/              # Posters, reports, and slides
├── cleaneVED_sample.zip              # Clean processed EV driving dataset
├── raweVED_sample.zip                # Raw GPS and vehicle telemetry data
├── model_only_select_features.ipynb  # LSTM model development and training
├── regen_only_ev.ipynb               # Regenerative braking and terrain analysis
├── requirements.txt                  # Required dependencies
├── scaler.pkl                        # Feature scaling object for inference
├── training_history.png              # Model loss and validation plots
└── README.md                         # Project documentation

**Terrain Logic:**

- Uphill / Flat Terrain — Positive battery consumption
- Downhill Terrain — Negative effective consumption with regenerative braking energy recovery

**Range Exhaustion:**

If battery reaches 0%, the GUI identifies exact stopping coordinates, displays reachable range, and provides real-world failure prediction.

## Performance Highlights

- R² Accuracy: 0.86
- Terrain-aware real-world predictions
- Smart route optimization
- Enhanced battery utilization
- Sustainable transportation planning

## Applications

- EV navigation systems
- Battery consumption forecasting
- Fleet management
- Smart city mobility
- Sustainable transport solutions
- Research and educational use

## Future Enhancements

- Charging station recommendations
- Real-time traffic integration
- Weather-aware battery prediction
- Mobile deployment
- Cloud optimization
- Multi-vehicle compatibility

## Research Contribution

This project demonstrates how combining machine learning, geographic information systems, regenerative braking analysis, and real-time routing APIs can significantly improve EV route planning and practical range prediction beyond traditional static estimators.
