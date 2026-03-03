# Electric Vehicle Energy Consumption Analysis

This project analyzes electric vehicle energy consumption patterns using LSTM neural networks and explores regenerative braking efficiency.

## Features

- **Data Preprocessing**: Cleans and processes electric vehicle driving data
- **LSTM Model**: Time-series prediction model for energy consumption
- **Regenerative Analysis**: Studies regenerative braking efficiency vs road gradient
- **Exploratory Data Analysis**: Comprehensive visualization of driving patterns

## Project Structure

```
├── regen.ipynb              # Main analysis notebook
├── energy_lstm_checkpoint.pt # Trained model checkpoint
├── clean_eved/              # Processed clean data
├── raw_eved/                # Raw vehicle data
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "ATET Project"
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook:**
   ```bash
   jupyter notebook regen.ipynb
   ```

## Data Processing

The project processes electric vehicle data with the following steps:

1. **Data Cleaning**: Removes invalid entries and handles missing values
2. **Feature Engineering**: 
   - Converts speed to m/s
   - Calculates acceleration from speed differences
   - Detects braking and stop events
   - Processes road gradient information

3. **Time Series Preparation**: Creates sliding windows for LSTM training

## Model Architecture

- **LSTM Network**: 2-layer LSTM with 64 hidden units
- **Features**: Speed, acceleration, gradient, braking flags, speed limits
- **Target**: Energy consumption prediction
- **Window Size**: 20 time steps

## Results

- The model achieves competitive performance in predicting energy consumption
- Analysis reveals correlation between road gradient and regenerative efficiency
- Braking events show distinct energy patterns compared to cruising

## Usage

The main analysis is contained in `regen.ipynb`. Run all cells to:

1. Process raw vehicle data
2. Perform exploratory data analysis
3. Train the LSTM model
4. Analyze regenerative braking efficiency

## Dependencies

- Python 3.8+
- PyTorch
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and research purposes.
