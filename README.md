# Kalman Filter for GPS Trajectory Smoothing

## Overview
This notebook applies a linear Kalman filter to real GPS trajectories from the Microsoft GeoLife dataset. Two motion models are implemented and compared:
Constant Velocity (CV) and Constant Acceleration (CA).

## Requirements
```bash
pip install numpy matplotlib pandas
```

## Project structure
```
.
├── src/
│   ├── kalman_tracker.py    # KalmanFilter class
│   ├── state_logger.py      # StateLogger class
│   └── state_permuter.py    # StatePermuter class
├── data/
│   └── raw_data_trajectories/
│       └── geolife_1.3/
│           └── geolife_trajectories/
│               └── data/
│                   ├── 010/
│                   ├── 020/
│                   └── 021/
└── 01__Kalman_Filter_for_GPS_Theory_and_Model.ipynb
```

## Data
Download the GeoLife dataset from:
https://www.microsoft.com/en-us/download/details.aspx?id=52367

Extract to `data/raw_data_trajectories/geolife_1.3/`.

## Usage
Run the notebook top to bottom. No additional configuration required — all parameters (σ_r, σ_a) are set inline with explanations.