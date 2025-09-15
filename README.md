# PPG Stress Preprocessing

This project focuses on **pre-processing photoplethysmography (PPG) signals for stress detection**. The preprocessing pipeline includes segmentation (10-second windows), noise removal, filtering, normalization, and preparation of signals for machine learning classification. This project was developed as part of my **undergraduate final thesis**.

---

## Project Overview

Raw PPG signals are segmented into 10-second chunks and stored in the `All_Data` folder. From this folder, samples are randomly divided into `train_data` and `test_data` for training and evaluation purposes. Several scripts are provided for data cleaning, filtering, and feature preparation, ensuring the dataset is suitable for classification tasks.

While the repository contains many files related to the full preprocessing workflow, **only a few files are necessary to directly run the core system and reproduce the main result of this thesis**:

- `ppg_gui.py` → A graphical user interface (GUI) that loads test PPG signals, visualizes the original and pre-processed signals, and classifies them as either *normal* or *stress*.  
- `knn_model.joblib` → The pre-trained K-Nearest Neighbors (KNN) model.  
- `scaler_knn.joblib` → The scaler used to standardize input features before feeding them into the model.  
- `test_data/` → Sample dataset used for testing and evaluation.  

With these files, anyone can directly run the GUI to observe both the preprocessing steps and the classification output without repeating the entire preprocessing and training pipeline.  

---

## How to Run

1. Install Python (version ≥ 3.7 recommended).  
2. Install the required dependencies:  

   ```bash
   pip install numpy scipy pandas scikit-learn matplotlib PyQt5 joblib
3. Ensure the following files and folder are present in the same directory:

  ppg_gui.py
  
  knn_model.joblib
  
  scaler_knn.joblib
  
  test_data/
  
  Run the GUI with:
  
  python ppg_gui.py


4. The application will open, display the original and pre-processed PPG signals, and show the classification results (normal vs. stress).




Additional Notes

For those interested in the complete preprocessing and training workflow, the repository also includes:

All_Data/ → All segmented PPG signals (10s segments).

train_data/ and test_data/ → Randomly sampled subsets from All_Data.

pre-processing_train.py and pre-processing_test.py → Scripts for preprocessing training and testing datasets.

knn_classifier.ipynb → Notebook for model training and evaluation.

p-value_test.ipynb → Statistical analysis.

These files are primarily intended for documentation of the research process, not for running the final system.



License

This project is distributed under the MIT License. See the LICENSE file for details.
