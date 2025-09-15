# PPG Stress Preprocessing

This project focuses on pre-processing photoplethysmography (PPG) signals for stress detection as part of my undergraduate thesis. The pipeline includes filtering, noise removal, segmentation, and signal preparation for machine learning classification using the K-Nearest Neighbors (KNN) algorithm. A graphical user interface (GUI) is also provided to visualize the original and pre-processed signals and display classification results (normal vs. stress).

## Dataset
- **all_data/** : segmented PPG signals (10-second windows).
- **train.csv** and **test.csv** : randomly split subsets from `all_data` for model training and evaluation.

## How to Run the Final Application
To run the GUI for stress classification, only the following files are required:
- `ppg_gui.py` (the GUI code)
- `knn_model.joblib` (trained KNN model)
- `scaler_knn.joblib` (scaler for preprocessing features)
- `test_data/` (example data for testing)

### Steps
1. Install Python (version ≥ 3.7 recommended).
2. Install the required dependencies:
   ```
   pip install numpy scipy pandas scikit-learn matplotlib PyQt5 joblib
   ```
3. Make sure the following files and folder are present in the same directory:
   - `ppg_gui.py`
   - `knn_model.joblib`
   - `scaler_knn.joblib`
   - `test_data/`
4. Run the GUI with:
   ```
   python ppg_gui.py
   ```
5. The application will open, display the original and pre-processed PPG signals, and show the classification results (normal vs. stress).
