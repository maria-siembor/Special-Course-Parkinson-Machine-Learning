# Gait Analysis: IMU-EMG Feature Fusion for Parkinson's Detection

## Project Overview
This project explores whether time-domain features extracted from synchronized ankle IMU (accelerometer, position, velocity) and lower-leg EMG (tibialis anterior, gastrocnemius) signals can distinguish gait patterns in Parkinson's disease (PD) patients from healthy controls (HC).

Data was collected from 6 healthy controls and 10 PD patients as part of a special course project. Raw IMU and EMG recordings are not included in this repository to protect participant privacy. The processing pipeline, feature extraction methods, and ML models are included so the approach can be reviewed and reused with equivalent data.

## Pipeline

1. **IMU processing** (`src/imu_processing.py`): loads ankle acceleration, position, and velocity data, estimates sampling rate, detects gait cycle boundaries via peak detection on the acceleration signal, and extracts per-cycle features (RMS, mean, standard deviation).

2. **EMG processing** (`src/emg_processing.py`): loads raw EMG from the tibialis anterior and gastrocnemius, applies bandpass filtering (20-450 Hz), a 50 Hz notch filter for powerline interference, rectification, and envelope extraction (4 Hz lowpass). Extracts per-segment features (RMS, MAV, integrated EMG, variance, zero-crossings) aligned to the IMU-derived gait cycles.

3. **Feature fusion** (`src/feature_fusion.py`): merges IMU and EMG features per subject and gait cycle into a single feature table, labeled by subject and group (HC/PD).

4. **Pipeline runner** (`src/run_pipeline.py`): runs the above steps across all subjects and writes the combined feature set to `results/fused_features.csv`.

5. **Classification** (`src/mlpipeline.py`, `src/mlpipeline2.py`): trains and evaluates Logistic Regression, Random Forest, and SVM classifiers on the fused features using 5-fold stratified cross-validation. `mlpipeline2.py` is the more rigorous version, with scaling done inside the CV pipeline to avoid data leakage, ROC/AUC evaluation, a confusion matrix, feature importance ranking, and a grid search over Random Forest hyperparameters.

An exploratory notebook (`notebook.ipynb`) covers the same steps interactively.

## Requirements

See `requirements.txt`. Install with: pip install -r requirements.txt

## Notes

This was an exploratory special course project rather than a validated clinical tool. Sample size is small (16 subjects total), and results should be read as a proof of concept for the feature extraction and fusion approach rather than a diagnostic claim.
