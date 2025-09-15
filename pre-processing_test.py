import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ========== DETEKSI SISTOLIK DAN DIASTOLIK ==========
def detect_systolic_diastolic(filtered_signal, second_derivative, fs):
    systolic_distance = int(fs * 0.4)
    height_systolic = 0.35
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
def extract_features(signal, systolic, diastolic, fs, tolerance=1):
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

    features = {
        'slope_mean': np.mean(sys_to_d_slopes) if sys_to_d_slopes else 0,
        'slope_std': np.std(sys_to_d_slopes) if sys_to_d_slopes else 0,
        'auc_mean': np.mean(auc_cycles) if auc_cycles else 0,
        'auc_std': np.std(auc_cycles) if len(auc_cycles) > 1 else 0,
        'jumlah_motion_artifacts': len(motion_artifacts_indices),
        'siklus_motion_artifacts': (motion_artifacts_indices)
    }

    return list(features.values()), list(features.keys())

# ========== PARAMETER FILTER ==========
fs = 256
low_cut = 0.5
high_cut = 6.5
nyquist = 0.5 * fs
low_ny = low_cut / nyquist
high_ny = high_cut / nyquist
b, a = butter(2, [low_ny, high_ny], btype='band')

# ====== LOOP PROSES SEMUA FILE DI train_data ======
all_features = []
file_ids = []
labels = []

# Ambil semua file di train_data
train_data_path = 'test_data'
for filename in os.listdir(train_data_path):
    if filename.endswith('.csv') and (filename.startswith('normal_') or filename.startswith('stress_')):
        file_path = os.path.join(train_data_path, filename)
        try:
            # Baca file
            df = pd.read_csv(file_path)
            signal_raw = df.iloc[:, 0].values
            filtered_signal = filtfilt(b, a, signal_raw)

            # Hitung turunan kedua untuk deteksi
            first_d = np.diff(filtered_signal)
            second_d = np.diff(first_d)
            fsecond_d = filtfilt(b, a, second_d)

            # Deteksi sistolik dan diastolik
            systolic_points, diastolic_points = detect_systolic_diastolic(filtered_signal, second_d, fs)

            # Ekstraksi fitur
            feats, column_names = extract_features(filtered_signal, systolic_points, diastolic_points, fs, tolerance=0.5)
            if feats is not None:
                # Ambil Set_ID dari nama file (misal, normal_set_76.csv → 76)
                set_id = int(filename.split('_set_')[1].split('.csv')[0])
                all_features.append(feats)
                file_ids.append(set_id)

                # Tentukan label berdasarkan nama file
                label = 0 if filename.startswith('normal_') else 1
                labels.append(label)

        except Exception as e:
            print(f"Error memproses file {file_path}: {e}")

# ====== SIMPAN KE CSV ======
column_names = ['slope_mean', 'slope_std', 'auc_mean', 'auc_std', 'jumlah_motion_artifacts', 'siklus_motion_artifacts']
df_features = pd.DataFrame(all_features, columns=column_names)
df_features['Set_ID'] = file_ids
df_features['Label'] = labels
df_features.to_csv('test_features.csv', index=False)

print(f"Ekstraksi fitur selesai. Hasil disimpan ke 'test_features.csv'.")