import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QFileDialog
from PyQt5.QtCore import Qt
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Signal processing parameters
window_size = 2560
sampling_rate = 256
fs = 256
low_cut = 0.5
high_cut = 6.5
nyquist = 0.5 * fs
low_ny = low_cut / nyquist
high_ny = high_cut / nyquist
b, a = butter(2, [low_ny, high_ny], btype='band')

# ========== DETEKSI TITIK SYSTOLIC DAN DIASTOLIC ==========
def detect_systolic_diastolic(filtered_signal, second_derivative, fs):
    systolic_distance = int(fs * 0.45)
    height_systolic = 0.005
    window_size = 10
    range_after_peak = int(fs * 0.3)
    min_gap = 5
    range_after_minimum = int(fs * 0.3)

    systolic = []
    for i in range(1, len(filtered_signal) - 1):
        if filtered_signal[i] > filtered_signal[i - 1] and filtered_signal[i] > filtered_signal[i + 1]:
            if filtered_signal[i] > height_systolic:
                systolic.append(i)

    systolic_distanced = []
    last_systolic = -systolic_distance
    for current in systolic:
        if current - last_systolic >= systolic_distance:
            start = max(0, current - window_size)
            end = min(len(filtered_signal), current + window_size)
            window_vals = [m for m in systolic if start <= m < end]
            min_index = min(window_vals, key=lambda x: filtered_signal[x])
            systolic_distanced.append(min_index)
            last_systolic = min_index

    yellow_points = []
    diastolic = []
    for i in range(len(systolic_distanced) - 1):
        min1 = systolic_distanced[i]
        search_end = min(min1 + range_after_minimum, len(second_derivative) - 1)
        segment = second_derivative[min1:search_end]

        peak_found = False
        highest_peak_value = None
        highest_peak_index = None

        for j in range(1, len(segment) - 1):
            if segment[j] > segment[j - 1] and segment[j] > segment[j + 1]:
                if highest_peak_value is None or segment[j] > highest_peak_value:
                    highest_peak_value = segment[j]
                    highest_peak_index = min1 + j
                    peak_found = True

        if peak_found:
            yellow_points.append(highest_peak_index)
            search_start = highest_peak_index + min_gap
            search_end = min(highest_peak_index + range_after_peak, len(second_derivative) - 1)
            local_segment = second_derivative[search_start:search_end]

            if len(local_segment) >= 3:
                for k in range(1, len(local_segment) - 1):
                    if local_segment[k] < local_segment[k - 1] and local_segment[k] < local_segment[k + 1]:
                        diastolic_point = search_start + k
                        diastolic.append(diastolic_point)
                        break
    return systolic_distanced, diastolic

# ========== FUNGSI EKSTRAKSI FITUR ==========
def extract_features(signal, systolic, diastolic, fs, tolerance=0.5):
    sys_to_d_slopes = []
    auc_cycles = []
    sys_dias_pairs = []
    motion_artifacts_indices = []

    for sys_point in systolic:
        next_dias = [d for d in diastolic if d > sys_point]
        if next_dias:
            closest_dias = min(next_dias)
            sys_dias_pairs.append((sys_point, closest_dias))

    for sys, dias in sys_dias_pairs:
        if dias <= sys:
            continue

        amp = signal[sys] - signal[dias]
        dur = (dias - sys) / fs
        if dur > 0:
            slope = amp / dur
            sys_to_d_slopes.append(slope)

        cycle = signal[sys:dias]
        if len(cycle) > 0:
            auc = np.trapz(cycle)
            auc_cycles.append(auc)

    # Hitung amplitudo untuk semua pasangan
    amps = [signal[sys] - signal[dias] for sys, dias in sys_dias_pairs]

    # Tentukan rentang mayoritas amp
    q1 = np.percentile(amps, 25)
    q3 = np.percentile(amps, 75)
    lower = q1 - tolerance
    upper = q3 + tolerance

    for i, amp in enumerate(amps):
        if amp < lower or amp > upper:
            motion_artifacts_indices.append(i)

    print(f"Jumlah siklus: {len(sys_dias_pairs)}")
    print(f"Jumlah terdeteksi Motion Artifacts: {len(motion_artifacts_indices)}")
    print(f"Indeks siklus terdeteksi Motion Artifacts: {motion_artifacts_indices}")

    features = {
        'slope_mean': np.mean(sys_to_d_slopes) if sys_to_d_slopes else 0,
        'slope_std': np.std(sys_to_d_slopes) if sys_to_d_slopes else 0,
        'auc_mean': np.mean(auc_cycles) if auc_cycles else 0,
        'auc_std': np.std(auc_cycles) if len(auc_cycles) > 1 else 0,
        'jumlah_siklus': len(sys_dias_pairs),
        'jumlah_motion_artifacts': len(motion_artifacts_indices),
        'siklus_motion_artifacts': motion_artifacts_indices
    }

    return list(features.values()), list(features.keys())

class PPGClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG Signal Classifiers")

        self.scroll_area = QScrollArea()
        self.central_widget = QWidget()
        self.scroll_area.setWidget(self.central_widget)
        self.scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self.scroll_area)
        self.layout = QVBoxLayout(self.central_widget)

        self.central_widget.setStyleSheet("""
            background-color: #fff;
            padding: 20px;
        """)

        self.header_label = QLabel("PPG SIGNAL CLASSIFIERS (NORMAL AND STRESS)")
        self.header_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px; text-align: center;")
        self.layout.addWidget(self.header_label)

        self.header_row = QHBoxLayout()
        self.dataset_label = QLabel("Insert PPG Dataset")
        self.dataset_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.header_row.addWidget(self.dataset_label)

        self.classification_label = QLabel("Hasil Klasifikasi: <span style='color: red;'>Belum ada data</span>")
        self.classification_label.setStyleSheet("font-size: 18px; text-align: center; font-weight: bold;")
        self.header_row.addWidget(self.classification_label, alignment=Qt.AlignRight)
        self.layout.addLayout(self.header_row)

        self.file_layout = QHBoxLayout()
        self.upload_button = QPushButton("Choose CSV File")
        self.upload_button.setStyleSheet("""
            padding: 10px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f2f2f2;
        """)
        self.upload_button.clicked.connect(self.upload_file)
        self.file_layout.addWidget(self.upload_button)

        self.submit_button = QPushButton("Submit")
        self.submit_button.setStyleSheet("""
            padding: 10px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f2f2f2;
        """)
        self.submit_button.clicked.connect(self.process_file)
        self.file_layout.addWidget(self.submit_button)

        self.show_plots_button = QPushButton("Tampilkan Gambar Sinyal")
        self.show_plots_button.setStyleSheet("""
            padding: 10px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f2f2f2;
        """)
        self.show_plots_button.clicked.connect(self.show_plots)
        self.file_layout.addWidget(self.show_plots_button)

        self.file_layout.addStretch(1)
        self.layout.addLayout(self.file_layout)

        self.signal_layout = QVBoxLayout()

        # Original signal
        self.original_signal_widget = QWidget()
        self.original_signal_layout = QVBoxLayout(self.original_signal_widget)
        self.original_signal_label = QLabel("Sinyal Asli")
        self.original_signal_label.setStyleSheet("font-size: 16px; text-align: center;")
        self.original_signal_layout.addWidget(self.original_signal_label)
        self.original_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.original_canvas = FigureCanvas(self.original_fig)
        self.original_toolbar = NavigationToolbar(self.original_canvas, self.original_signal_widget)
        self.original_signal_layout.addWidget(self.original_toolbar)
        self.original_signal_layout.addWidget(self.original_canvas)
        self.original_desc_label = QLabel("Sinyal asli")
        self.original_desc_label.setStyleSheet("font-size: 18px; text-align: center; margin-top: 5px;")
        self.original_desc_label.setAlignment(Qt.AlignCenter)
        self.original_signal_layout.addWidget(self.original_desc_label)
        self.signal_layout.addWidget(self.original_signal_widget)

        # Filtered signal
        self.filtered_signal_widget = QWidget()
        self.filtered_signal_layout = QVBoxLayout(self.filtered_signal_widget)
        self.filtered_signal_label = QLabel("Sinyal Setelah Bandpass")
        self.filtered_signal_label.setStyleSheet("font-size: 16px; text-align: center;")
        self.filtered_signal_layout.addWidget(self.filtered_signal_label)
        self.filtered_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.filtered_canvas = FigureCanvas(self.filtered_fig)
        self.filtered_toolbar = NavigationToolbar(self.filtered_canvas, self.filtered_signal_widget)
        self.filtered_signal_layout.addWidget(self.filtered_toolbar)
        self.filtered_signal_layout.addWidget(self.filtered_canvas)
        self.filtered_desc_label = QLabel("Sinyal setelah bandpass")
        self.filtered_desc_label.setStyleSheet("font-size: 18px; text-align: center; margin-top: 5px;")
        self.filtered_desc_label.setAlignment(Qt.AlignCenter)
        self.filtered_signal_layout.addWidget(self.filtered_desc_label)
        self.signal_layout.addWidget(self.filtered_signal_widget)

        # Systolic detection
        self.systolic_widget = QWidget()
        self.systolic_layout = QVBoxLayout(self.systolic_widget)
        self.systolic_label = QLabel("Deteksi Systolic")
        self.systolic_label.setStyleSheet("font-size: 16px; text-align: center;")
        self.systolic_layout.addWidget(self.systolic_label)
        self.systolic_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.systolic_canvas = FigureCanvas(self.systolic_fig)
        self.systolic_toolbar = NavigationToolbar(self.systolic_canvas, self.systolic_widget)
        self.systolic_layout.addWidget(self.systolic_toolbar)
        self.systolic_layout.addWidget(self.systolic_canvas)
        self.systolic_desc_label = QLabel("Deteksi systolic")
        self.systolic_desc_label.setStyleSheet("font-size: 18px; text-align: center; margin-top: 5px;")
        self.systolic_desc_label.setAlignment(Qt.AlignCenter)
        self.systolic_layout.addWidget(self.systolic_desc_label)
        self.signal_layout.addWidget(self.systolic_widget)

        # First derivative
        self.first_derivative_widget = QWidget()
        self.first_derivative_layout = QVBoxLayout(self.first_derivative_widget)
        self.first_derivative_label = QLabel("Turunan Pertama Sinyal PPG")
        self.first_derivative_label.setStyleSheet("font-size: 16px; text-align: center;")
        self.first_derivative_layout.addWidget(self.first_derivative_label)
        self.first_derivative_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.first_derivative_canvas = FigureCanvas(self.first_derivative_fig)
        self.first_derivative_toolbar = NavigationToolbar(self.first_derivative_canvas, self.first_derivative_widget)
        self.first_derivative_layout.addWidget(self.first_derivative_toolbar)
        self.first_derivative_layout.addWidget(self.first_derivative_canvas)
        self.first_derivative_desc_label = QLabel("Turunan pertama sinyal PPG")
        self.first_derivative_desc_label.setStyleSheet("font-size: 18px; text-align: center; margin-top: 5px;")
        self.first_derivative_desc_label.setAlignment(Qt.AlignCenter)
        self.first_derivative_layout.addWidget(self.first_derivative_desc_label)
        self.signal_layout.addWidget(self.first_derivative_widget)

        # Second derivative with diastolic
        self.second_derivative_widget = QWidget()
        self.second_derivative_layout = QVBoxLayout(self.second_derivative_widget)
        self.second_derivative_label = QLabel("Turunan Kedua Sinyal PPG dengan Deteksi Diastolic")
        self.second_derivative_label.setStyleSheet("font-size: 16px; text-align: center;")
        self.second_derivative_layout.addWidget(self.second_derivative_label)
        self.second_derivative_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.second_derivative_canvas = FigureCanvas(self.second_derivative_fig)
        self.second_derivative_toolbar = NavigationToolbar(self.second_derivative_canvas, self.second_derivative_widget)
        self.second_derivative_layout.addWidget(self.second_derivative_toolbar)
        self.second_derivative_layout.addWidget(self.second_derivative_canvas)
        self.second_derivative_desc_label = QLabel("Turunan kedua sinyal PPG dengan deteksi diastolic")
        self.second_derivative_desc_label.setStyleSheet("font-size: 18px; text-align: center; margin-top: 5px;")
        self.second_derivative_desc_label.setAlignment(Qt.AlignCenter)
        self.second_derivative_layout.addWidget(self.second_derivative_desc_label)
        self.signal_layout.addWidget(self.second_derivative_widget)

        self.layout.addLayout(self.signal_layout)

        # Combined plot
        self.combined_widget = QWidget()
        self.combined_layout = QVBoxLayout(self.combined_widget)
        self.combined_signal_label = QLabel("Sinyal Bandpass dengan Deteksi")
        self.combined_signal_label.setStyleSheet("font-size: 16px; text-align: center;")
        self.combined_layout.addWidget(self.combined_signal_label)
        self.combined_fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.combined_canvas = FigureCanvas(self.combined_fig)
        self.combined_toolbar = NavigationToolbar(self.combined_canvas, self.combined_widget)
        self.combined_layout.addWidget(self.combined_toolbar)
        self.combined_layout.addWidget(self.combined_canvas)
        self.combined_desc_label = QLabel("Sinyal setelah bandpass dengan deteksi systolic dan diastolic")
        self.combined_desc_label.setStyleSheet("font-size: 18px; text-align: center; margin-top: 5px;")
        self.combined_desc_label.setAlignment(Qt.AlignCenter)
        self.combined_layout.addWidget(self.combined_desc_label)
        self.layout.addWidget(self.combined_widget)

        # Features labels
        self.features_label = QLabel("Jumlah Siklus: N/A | Jumlah Motion Artifacts: N/A | Indeks Motion Artifacts: N/A")
        self.features_label.setStyleSheet("font-size: 20px; text-align: center; margin-top: 10px;")
        self.features_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.features_label)

        self.extracted_features_label = QLabel("Extracted Features: N/A")
        self.extracted_features_label.setStyleSheet("font-size: 20px; text-align: center; margin-top: 5px;")
        self.extracted_features_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.extracted_features_label)

        self.layout.addStretch(1)

        self.file_path = None
        self.signal = None
        self.filtered_signal = None
        self.first_derivative = None
        self.second_derivative = None
        self.systolic_points = None
        self.diastolic_points = None
        self.jumlah_siklus = None
        self.jumlah_motion_artifacts = None
        self.siklus_motion_artifacts = None

        self.showMaximized()
        QApplication.processEvents()

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.file_path = file_path
            self.upload_button.setText(os.path.basename(file_path))
            logging.info(f"Selected file: {file_path}")
        else:
            logging.info("No file selected")

    def process_file(self):
        if not self.file_path:
            self.classification_label.setText("Hasil Klasifikasi: <span style='color: red;'>No file selected</span>")
            self.clear_plots()
            return

        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"CSV file not found at: {self.file_path}")

            model_path = os.path.abspath('knn_model.joblib')
            scaler_path = os.path.abspath('scaler_knn.joblib')
            logging.info(f"Checking model path: {model_path}")
            logging.info(f"Checking scaler path: {scaler_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

            df = pd.read_csv(self.file_path)
            logging.info(f"Loaded CSV file: {self.file_path}, shape: {df.shape}, columns: {df.columns.tolist()}")
            if df.shape[1] < 1:
                raise ValueError("CSV file has no columns")
            
            self.signal = df.iloc[:, 0].values
            logging.info(f"Signal loaded, length: {len(self.signal)}, first few values: {self.signal[:5]}")
            if not np.all(np.isfinite(self.signal)):
                raise ValueError("CSV contains non-numeric or invalid data in the first column")
            if len(self.signal) < window_size:
                raise ValueError(f"Signal length ({len(self.signal)}) is less than window_size ({window_size})")
            
            self.filtered_signal = filtfilt(b, a, self.signal)
            logging.info(f"Filtered signal, length: {len(self.filtered_signal)}, first few values: {self.filtered_signal[:5]}")

            self.first_derivative = np.diff(self.filtered_signal)
            self.second_derivative = np.diff(self.first_derivative)
            logging.info(f"First derivative length: {len(self.first_derivative)}, Second derivative length: {len(self.second_derivative)}")

            self.systolic_points, self.diastolic_points = detect_systolic_diastolic(self.filtered_signal, self.second_derivative, fs)
            logging.info(f"Systolic points: {self.systolic_points[:5]} (first 5), total: {len(self.systolic_points)}")
            logging.info(f"Diastolic points: {self.diastolic_points[:5]} (first 5), total: {len(self.diastolic_points)}")

            feature_values, feature_names = extract_features(self.filtered_signal, self.systolic_points, self.diastolic_points, fs)
            features_dict = dict(zip(feature_names, feature_values))
            
            self.jumlah_siklus = features_dict['jumlah_siklus']
            self.jumlah_motion_artifacts = features_dict['jumlah_motion_artifacts']
            self.siklus_motion_artifacts = features_dict['siklus_motion_artifacts']
            
            self.features_label.setText(
                f"Jumlah Siklus: {self.jumlah_siklus} | "
                f"Jumlah Motion Artifacts: {self.jumlah_motion_artifacts} | "
                f"Indeks Motion Artifacts: {self.siklus_motion_artifacts}"
            )

            classification_features = ['slope_mean', 'slope_std', 'auc_mean', 'auc_std']
            classification_values = [features_dict[f] for f in classification_features]
            df_features = pd.DataFrame([classification_values], columns=classification_features)
            logging.info(f"Extracted features for classification, shape: {df_features.shape}, values: {classification_values}")

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logging.info("Loaded model and scaler")

            X_scaled = scaler.transform(df_features)
            scaled_features_dict = dict(zip(classification_features, X_scaled[0]))
            logging.info(f"Scaled features: {scaled_features_dict}")

            y_pred = model.predict(X_scaled)
            self.classification = 'Normal' if y_pred[0] == 0 else 'Stress'
            logging.info(f"Prediction: {self.classification}")

            self.classification_label.setText(f"Hasil Klasifikasi: <span style='color: red;'>{self.classification}</span>")
            self.extracted_features_label.setText(
                f"Extracted Features (Scaled): Slope Mean: {scaled_features_dict['slope_mean']:.2f} | "
                f"Slope Std: {scaled_features_dict['slope_std']:.2f} | "
                f"AUC Mean: {scaled_features_dict['auc_mean']:.2f} | "
                f"AUC Std: {scaled_features_dict['auc_std']:.2f}"
            )

        except Exception as e:
            logging.error(f"Error processing file {self.file_path}: {str(e)}")
            self.classification = f"Error: {str(e)}"
            self.classification_label.setText(f"Hasil Klasifikasi: <span style='color: red;'>{self.classification}</span>")
            self.features_label.setText("Jumlah Siklus: N/A | Jumlah Motion Artifacts: N/A | Indeks Motion Artifacts: N/A")
            self.extracted_features_label.setText("Extracted Features: N/A")

    def update_plots(self):
        # Original signal
        self.original_fig.clear()
        ax = self.original_fig.add_subplot(111)
        if self.signal is not None and len(self.signal) >= window_size:
            time = np.arange(window_size) / fs
            ax.plot(time, self.signal[:window_size], color='blue', linewidth=1, label='Sinyal Raw')
            logging.info("Original signal plotted")
        else:
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=12)
            logging.warning("No data for original signal plot")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
        self.original_canvas.draw()
        self.original_canvas.setMinimumSize(800, 600)
        self.original_signal_widget.update()

        # Filtered signal
        self.filtered_fig.clear()
        ax = self.filtered_fig.add_subplot(111)
        if self.filtered_signal is not None and len(self.filtered_signal) >= window_size:
            time = np.arange(window_size) / fs
            ax.plot(time, self.filtered_signal[:window_size], color='seagreen', linewidth=1, label='Sinyal Setelah Bandpass')
            logging.info("Filtered signal plotted")
        else:
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=12)
            logging.warning("No data for filtered signal plot")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
        self.filtered_canvas.draw()
        self.filtered_canvas.setMinimumSize(800, 600)
        self.filtered_signal_widget.update()

        # Systolic detection
        self.systolic_fig.clear()
        ax = self.systolic_fig.add_subplot(111)
        if self.filtered_signal is not None and len(self.filtered_signal) >= window_size:
            time = np.arange(window_size) / fs
            ax.plot(time, self.filtered_signal[:window_size], color='seagreen', linewidth=1, label='Sinyal Setelah Bandpass')
            if self.systolic_points:
                valid_systolic = [p for p in self.systolic_points if p < window_size]
                if valid_systolic:
                    systolic_time = [t / fs for t in valid_systolic]
                    ax.scatter(systolic_time, self.filtered_signal[valid_systolic], color='red', s=50, marker='o', label='Systolic', edgecolors='black')
                    logging.info("Systolic points plotted")
            logging.info("Systolic signal plotted")
        else:
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=12)
            logging.warning("No data for systolic plot")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
        self.systolic_canvas.draw()
        self.systolic_canvas.setMinimumSize(800, 600)
        self.systolic_widget.update()

        # First derivative
        self.first_derivative_fig.clear()
        ax = self.first_derivative_fig.add_subplot(111)
        if self.first_derivative is not None and len(self.first_derivative) >= window_size-1:
            time = np.arange(window_size-1) / fs
            ax.plot(time, self.first_derivative[:window_size-1], color='purple', linewidth=1, label='Turunan Pertama')
            logging.info("First derivative plotted")
        else:
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=12)
            logging.warning("No data for first derivative plot")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
        self.first_derivative_canvas.draw()
        self.first_derivative_canvas.setMinimumSize(800, 600)
        self.first_derivative_widget.update()

        # Second derivative with diastolic
        self.second_derivative_fig.clear()
        ax = self.second_derivative_fig.add_subplot(111)
        if self.second_derivative is not None and len(self.second_derivative) >= window_size-2:
            time = np.arange(window_size-2) / fs
            ax.plot(time, self.second_derivative[:window_size-2], color='red', linewidth=1, label='Turunan Kedua')
            if self.diastolic_points:
                valid_diastolic = [p for p in self.diastolic_points if p < window_size-2]
                if valid_diastolic:
                    diastolic_time = [t / fs for t in valid_diastolic]
                    ax.scatter(diastolic_time, self.second_derivative[valid_diastolic], color='orange', s=50, marker='o', label='Diastolic', edgecolors='black')
                    logging.info("Diastolic points plotted")
            logging.info("Second derivative plotted")
        else:
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=12)
            logging.warning("No data for second derivative plot")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
        self.second_derivative_canvas.draw()
        self.second_derivative_canvas.setMinimumSize(800, 600)
        self.second_derivative_widget.update()

        # Combined plot
        self.combined_fig.clear()
        ax = self.combined_fig.add_subplot(111)
        if self.filtered_signal is not None and len(self.filtered_signal) >= window_size:
            time = np.arange(window_size) / fs
            ax.plot(time, self.filtered_signal[:window_size], color='seagreen', linewidth=1, label='Sinyal Setelah Bandpass')
            if self.systolic_points:
                valid_systolic = [p for p in self.systolic_points if p < window_size]
                if valid_systolic:
                    systolic_time = [t / fs for t in valid_systolic]
                    ax.scatter(systolic_time, self.filtered_signal[valid_systolic], color='red', s=50, marker='o', label='Systolic', edgecolors='black')
            if self.diastolic_points:
                valid_diastolic = [p for p in self.diastolic_points if p < window_size]
                if valid_diastolic:
                    diastolic_time = [t / fs for t in valid_diastolic]
                    ax.scatter(diastolic_time, self.filtered_signal[valid_diastolic], color='orange', s=50, marker='o', label='Diastolic', edgecolors='black')
            logging.info("Combined signal plotted")
        else:
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=12)
            logging.warning("No data for combined plot")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True)
        self.combined_canvas.draw()
        self.combined_canvas.setMinimumSize(800, 600)
        self.combined_widget.update()

        self.central_widget.update()
        QApplication.processEvents()

    def update_plots_with_message(self, message):
        # Show plots with error message
        self.original_fig.clear()
        ax = self.original_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        self.original_canvas.draw()
        self.original_canvas.setMinimumSize(800, 600)
        self.original_signal_widget.update()

        self.filtered_fig.clear()
        ax = self.filtered_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        self.filtered_canvas.draw()
        self.filtered_canvas.setMinimumSize(800, 600)
        self.filtered_signal_widget.update()

        self.systolic_fig.clear()
        ax = self.systolic_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        self.systolic_canvas.draw()
        self.systolic_canvas.setMinimumSize(800, 600)
        self.systolic_widget.update()

        self.first_derivative_fig.clear()
        ax = self.first_derivative_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        self.first_derivative_canvas.draw()
        self.first_derivative_canvas.setMinimumSize(800, 600)
        self.first_derivative_widget.update()

        self.second_derivative_fig.clear()
        ax = self.second_derivative_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        self.second_derivative_canvas.draw()
        self.second_derivative_canvas.setMinimumSize(800, 600)
        self.second_derivative_widget.update()

        self.combined_fig.clear()
        ax = self.combined_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        self.combined_canvas.draw()
        self.combined_canvas.setMinimumSize(800, 600)
        self.combined_widget.update()

        self.central_widget.update()
        QApplication.processEvents()

    def show_plots(self):
        # Ensure plots are updated
        if self.signal is None:
            self.update_plots_with_message("No data to display. Please submit a file first.")
            logging.warning("Show plots called but no data available")
        else:
            self.update_plots()
            logging.info("Displayed interactive signal plots successfully")

    def clear_plots(self):
        self.original_fig.clear()
        self.filtered_fig.clear()
        self.systolic_fig.clear()
        self.first_derivative_fig.clear()
        self.second_derivative_fig.clear()
        self.combined_fig.clear()
        self.original_canvas.draw()
        self.filtered_canvas.draw()
        self.systolic_canvas.draw()
        self.first_derivative_canvas.draw()
        self.second_derivative_canvas.draw()
        self.combined_canvas.draw()
        self.central_widget.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PPGClassifierGUI()
    window.show()
sys.exit(app.exec_())

