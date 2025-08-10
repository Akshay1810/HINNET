import os
import h5py
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.signal import butter, filtfilt
from ahrs.filters import Madgwick
from ahrs.common import Quaternion
import random

# --- Constants ---
SAMPLING_FREQUENCY = 60.0
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(WINDOW_SECONDS * SAMPLING_FREQUENCY)
G = 9.81


def apply_low_pass_filter(data, cutoff_hz, fs, order=4):
    """
    Applies a low-pass Butterworth filter to the data.

    Args:
        data (np.ndarray): The 1D signal to filter.
        cutoff_hz (float): The cutoff frequency for the filter.
        fs (float): The sampling frequency of the signal.
        order (int): The order of the Butterworth filter.

    Returns:
        np.ndarray: The filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # Apply the filter forwards and backwards to remove phase shift
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def check_and_convert_gyro_units(df_imu):
    """
    Converts gyroscope data from degrees/s to radians/s if necessary.
    A heuristic is used to check if the values are too high for rad/s.
    """
    gyro_data = df_imu[["gyro_x", "gyro_y", "gyro_z"]].values
    max_abs_val = np.max(np.abs(gyro_data))

    if max_abs_val > 10.0:
        df_imu[["gyro_x", "gyro_y", "gyro_z"]] = np.deg2rad(gyro_data)

    return df_imu


def find_optimal_cutoff(data, fs):
    """
    Return the cut-off frequency (Hz) that preserves 99 % of the
    signal’s energy. Always returns a float, never an array.
    """
    N = len(data)
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, 1 / fs)

    pos_freqs = xf[: N // 2]
    energy_spec = np.abs(yf[: N // 2]) ** 2
    total_E = np.sum(energy_spec)
    cum_E = np.cumsum(energy_spec)

    idx = np.where(cum_E >= 0.99 * total_E)[0]
    if len(idx):
        # Return a Python float (or NumPy scalar) instead of a 1-element array
        return float(pos_freqs[idx[0]])
    else:
        return float(fs / 2)  # Nyquist if nothing found


# def roll_pitch_compensation(df_imu, fs=SAMPLING_FREQUENCY):
#     """
#     HINNet roll-pitch compensation as per Section 2.1 of the paper.
#     It directly computes roll (phi) and pitch (theta) from the accelerometer
#     data to align the IMU with the gravity vector.
#     """
#     acc = df_imu[["acc_x", "acc_y", "acc_z"]].values
#     gyr = df_imu[["gyro_x", "gyro_y", "gyro_z"]].values
#     num_samples = len(df_imu)

#     # cutoff_frequency_x_gyr = find_optimal_cutoff(gyr[:, 0], fs)
#     # cutoff_frequency_y_gyr = find_optimal_cutoff(gyr[:, 1], fs)
#     # cutoff_frequency_z_gyr = find_optimal_cutoff(gyr[:, 2], fs)

#     # gyr[:, 0] = apply_low_pass_filter(gyr[:, 0], cutoff_frequency_x_gyr, fs)
#     # gyr[:, 1] = apply_low_pass_filter(gyr[:, 1], cutoff_frequency_y_gyr, fs)
#     # gyr[:, 2] = apply_low_pass_filter(gyr[:, 2], cutoff_frequency_z_gyr, fs)

#     acc_norm = np.zeros_like(acc)
#     gyr_norm = np.zeros_like(gyr)

#     for t in range(num_samples):
#         phi = np.arctan2(acc[t, 1], acc[t, 2])
#         theta = np.arctan2(-acc[t, 0], np.sqrt(acc[t, 1] ** 2 + acc[t, 2] ** 2))

#         c_phi, s_phi = np.cos(phi), np.sin(phi)
#         c_theta, s_theta = np.cos(theta), np.sin(theta)

#         R_a = np.array(
#             [
#                 [c_theta, 0.0, -s_theta],
#                 [s_phi * s_theta, c_phi, s_phi * c_theta],
#                 [c_phi * s_theta, -s_phi, c_phi * c_theta],
#             ]
#         )

#         R_w = np.array(
#             [
#                 [1.0, 0.0, -s_theta],
#                 [0.0, c_phi, s_phi * c_theta],
#                 [0.0, -s_phi, c_phi * c_theta],
#             ]
#         )

#         acc_norm[t] = R_a.T @ acc[t]
#         gyr_norm[t] = R_w.T @ gyr[t]

#     return np.hstack((acc_norm, gyr_norm))
def roll_pitch_compensation(df_imu, mag_df=None, fs=SAMPLING_FREQUENCY, beta=0.1):
    """
    HINNet roll-pitch compensation per the paper:
    1) Run Madgwick → quaternion q[t]
    2) Extract roll (ϕ), pitch (θ) from q[t]
    3) Build R_a, R_w and apply R_a^T to acc, R_w^T to gyro
    Returns (N,6) array: [acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z]
    """
    acc = df_imu[["acc_x", "acc_y", "acc_z"]].values
    gyr = df_imu[["gyro_x", "gyro_y", "gyro_z"]].values
    use_mag = mag_df is not None and not mag_df.empty
    if use_mag:
        mag = mag_df[["mag_x", "mag_y", "mag_z"]].values

    # 1) Madgwick → quaternion
    mad = Madgwick(frequency=fs, beta=beta)
    q = np.zeros((len(df_imu), 4))
    # ─── Seed initial tilt from the first accel sample ───
    ax0, ay0, az0 = acc[0]
    phi0 = np.arctan2(ay0, az0)
    theta0 = np.arctan2(-ax0, np.sqrt(ay0**2 + az0**2))

    qw = np.cos(phi0 / 2) * np.cos(theta0 / 2)
    qx = np.sin(phi0 / 2) * np.cos(theta0 / 2)
    qy = np.cos(phi0 / 2) * np.sin(theta0 / 2)
    qz = -np.sin(phi0 / 2) * np.sin(theta0 / 2)
    q[0] = [qw, qx, qy, qz]
    for t in range(1, len(q)):
        if use_mag:
            q[t] = mad.updateMARG(q[t - 1], gyr=gyr[t], acc=acc[t], mag=mag[t])
        else:
            q[t] = mad.updateIMU(q[t - 1], gyr=gyr[t], acc=acc[t])
        q[t] /= np.linalg.norm(q[t])

    # 2) Euler angles from quaternion
    #    to_angles() gives (roll, pitch, yaw)
    eul = np.array([Quaternion(qi).to_angles() for qi in q])
    roll = eul[:, 0]
    pitch = eul[:, 1]

    # 3) allocate outputs
    acc_norm = np.zeros_like(acc)
    gyr_norm = np.zeros_like(gyr)

    # 4) apply R_a^T and R_w^T
    for t in range(len(df_imu)):
        ϕ = roll[t]
        θ = pitch[t]
        cϕ, sϕ = np.cos(ϕ), np.sin(ϕ)
        cθ, sθ = np.cos(θ), np.sin(θ)

        # HINNet’s R_a, R_w
        R_a = np.array(
            [[cθ, 0.0, -sθ], [sϕ * sθ, cϕ, sϕ * cθ], [cϕ * sθ, -sϕ, cϕ * cθ]]
        )
        R_w = np.array([[1.0, 0.0, -sθ], [0.0, cϕ, sϕ * cθ], [0.0, -sϕ, cϕ * cθ]])

        acc_norm[t] = R_a.T @ acc[t]
        gyr_norm[t] = R_w.T @ gyr[t]

    return np.hstack((acc_norm, gyr_norm))

def calculate_peak_ratio(acc_x_data, fs):
    """
    Calculates the peak ratio from the normalized accelerometer x-axis data using FFT.
    This function implements the core idea from the "Walking Compass" and "HINNet" papers.

    Args:
        acc_x_data (np.ndarray): The normalized accelerometer x-axis data for a window.
        fs (float): The sampling frequency.

    Returns:
        float: The peak ratio P_swing / P_stepping. Returns 0 if no peaks are found.
    """
    N = len(acc_x_data)
    if N == 0:
        return 0.0
    yf = fft(acc_x_data)
    xf = fftfreq(N, 1 / fs)

    # Find the indices for positive frequencies
    pos_freq_indices = np.where(xf > 0)
    pos_freqs = xf[pos_freq_indices]
    magnitudes = 2.0 / N * np.abs(yf[pos_freq_indices])

    # Find the main peaks, corresponding to stepping and swinging.
    # The stepping frequency is typically higher than the swinging frequency.
    # We look for the two most prominent peaks.
    peaks, _ = signal.find_peaks(magnitudes, height=0.1)

    # Sort peaks by magnitude in descending order
    sorted_peaks = peaks[np.argsort(magnitudes[peaks])][::-1]

    if len(sorted_peaks) < 2:
        return 0.0

    # Heuristic to identify stepping and swinging peaks
    # Stepping peak is generally at a higher frequency than swinging peak
    peak_1_idx = sorted_peaks[0]
    peak_2_idx = sorted_peaks[1]

    # P_stepping should have a higher frequency.
    if pos_freqs[peak_1_idx] > pos_freqs[peak_2_idx]:
        p_stepping = magnitudes[peak_1_idx]
        p_swinging = magnitudes[peak_2_idx]
    else:
        p_stepping = magnitudes[peak_2_idx]
        p_swinging = magnitudes[peak_1_idx]

    if p_stepping == 0:
        return 0.0

    return p_swinging / p_stepping


# --- 2. PyTorch Dataset ---
# The HINNetDataset class is simplified to only use 6 features (acc_xyz, gyro_xyz)


class HINNetDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing the HINNet data.
    Uses a sliding window (stride=1) for data augmentation during training.
    """

    def __init__(self, file_paths, stride, scaler):
        self.feature_windows = []
        self.labels = []
        self.file_paths = file_paths
        self.stride = stride
        self.scaler = scaler
        self._prepare_data()

    def _prepare_data(self):
        print(f"Processing {len(self.file_paths)} files for dataset...")
        for filepath in self.file_paths:
            print(f"Loading data from: {os.path.basename(filepath)}")
            with h5py.File(filepath, "r") as f:
                acc, gyro, pos, ori = (
                    f["acc"][:],
                    f["gyro"][:],
                    f["pos"][:],
                    f["ori"][:],
                )

            CALIB_SAMPLES = 1000

            if len(acc) <= CALIB_SAMPLES + 2 * WINDOW_SIZE:
                print(
                    f"Skipping file {filepath}, too short for calibration and peak ratio calculation."
                )
                continue

            acc = acc[CALIB_SAMPLES:]
            gyro = gyro[CALIB_SAMPLES:]
            pos = pos[CALIB_SAMPLES:]
            ori = ori[CALIB_SAMPLES:]

            df_imu = pd.DataFrame(
                {
                    "acc_x": acc[:, 0],
                    "acc_y": acc[:, 1],
                    "acc_z": acc[:, 2],
                    "gyro_x": gyro[:, 0],
                    "gyro_y": gyro[:, 1],
                    "gyro_z": gyro[:, 2],
                }
            )
            df_imu = check_and_convert_gyro_units(df_imu)
            print("RAW Acc:\n", df_imu[["acc_x", "acc_y", "acc_z"]].head())
            print("RAW Gyr(in rad):\n", df_imu[["gyro_x", "gyro_y", "gyro_z"]].head())
            norm_imu_data = roll_pitch_compensation(df_imu)
            print("NORM Acc:\n", norm_imu_data[:5, 0:3])
            print("NORM Gyr:\n", norm_imu_data[:5, 3:6])
            # Separate out normalized accelerometer x,y,z for feature calculation
            norm_acc_x = norm_imu_data[:, 0]

            num_samples = len(norm_imu_data)

            # Start loop from a point where a 'before' window is available
            start_index = WINDOW_SIZE
            # End loop at a point where an 'after' window is available
            end_index = num_samples - 2 * WINDOW_SIZE

            for i in range(start_index, end_index, self.stride):
                window_end = i + WINDOW_SIZE

                # --- Calculate peak ratio features for EACH sample in the current window ---
                peak_ratio_features = np.zeros((WINDOW_SIZE, 2))
                for j in range(WINDOW_SIZE):
                    sample_index = i + j

                    # P_ratio_before is from the window BEFORE the current sample
                    acc_x_before_window = norm_acc_x[
                        sample_index - WINDOW_SIZE : sample_index
                    ]
                    p_ratio_before = calculate_peak_ratio(
                        acc_x_before_window, SAMPLING_FREQUENCY
                    )

                    # P_ratio_after is from the window AFTER the current sample
                    acc_x_after_window = norm_acc_x[
                        sample_index : sample_index + WINDOW_SIZE
                    ]
                    p_ratio_after = calculate_peak_ratio(
                        acc_x_after_window, SAMPLING_FREQUENCY
                    )

                    peak_ratio_features[j, :] = [p_ratio_before, p_ratio_after]

                # Combine IMU data and peak ratio features
                imu_window = norm_imu_data[i:window_end]
                all_features = np.hstack((imu_window, peak_ratio_features))

                if self.scaler:
                    all_features = self.scaler.transform(all_features)

                pos_start, pos_end = pos[i], pos[i + WINDOW_SIZE - 1]
                ori_start, ori_end = ori[i], ori[i + WINDOW_SIZE - 1]

                delta_l = np.linalg.norm(pos_end[[0, 2]] - pos_start[[0, 2]])

                yaw_start_deg = ori_start[1]
                yaw_end_deg = ori_end[1]

                yaw_start_rad = np.deg2rad(yaw_start_deg)
                yaw_end_rad = np.deg2rad(yaw_end_deg)

                delta_psi_raw = yaw_end_rad - yaw_start_rad
                delta_psi = np.arctan2(np.sin(delta_psi_raw), np.cos(delta_psi_raw))

                if delta_l > 5.0:
                    continue
                if abs(delta_psi) > np.pi:
                    continue
                self.feature_windows.append(all_features)
                self.labels.append(np.array([delta_l, delta_psi]))

    def __len__(self):
        return len(self.feature_windows)

    def __getitem__(self, idx):
        features = torch.tensor(self.feature_windows[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, labels


# --- 3. PyTorch Model ---
# The model's input dimension is changed to 8


class HINNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=2, num_layers=2):
        super(HINNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        trans_err_sq = (pred[:, 0] - target[:, 0]) ** 2
        head_err_sq = (pred[:, 1] - target[:, 1]) ** 2
        loss = trans_err_sq + self.alpha * head_err_sq
        return loss.mean()


# --- 4. Training and Evaluation ---


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, sum_l2, sum_h2, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            diff = outputs - labels
            sum_l2 += torch.sum(diff[:, 0] ** 2).item()
            sum_h2 += torch.sum(diff[:, 1] ** 2).item()
            n += labels.size(0)

    mean_loss = total_loss / n
    mse_l = sum_l2 / n
    mse_h = sum_h2 / n
    return mean_loss, mse_l, mse_h


# --- 5. Trajectory Reconstruction and Metrics ---


def reconstruct_trajectory(predictions, initial_pos, initial_heading):
    positions = [initial_pos]
    heading = initial_heading
    for delta_l, delta_psi in predictions:
        # Corrected integration: update heading, then calculate step
        heading += delta_psi
        dx_step = delta_l * np.cos(heading)
        dz_step = delta_l * np.sin(heading)
        positions.append(positions[-1] + np.array([dx_step, dz_step]))
    return np.array(positions)


def trajectory_from_deltas(delta_array, init_pos, init_heading):
    return reconstruct_trajectory(delta_array, init_pos, init_heading)


def calculate_ate(gt_traj, est_traj):
    error = gt_traj - est_traj
    squared_error = np.sum(error**2, axis=1)
    ate = np.sqrt(np.mean(squared_error))
    return ate


def calculate_rte(gt_traj, est_traj, interval_sec=60, fs=SAMPLING_FREQUENCY):
    interval = int(interval_sec / WINDOW_SECONDS)
    if interval == 0:
        return 0

    errors = []
    num_intervals = (len(gt_traj) - 1) // interval
    for i in range(num_intervals):
        start_index = i * interval
        end_index = start_index + interval
        gt_segment = gt_traj[start_index:end_index]
        est_segment = est_traj[start_index:end_index]
        gt_segment_aligned = gt_segment - gt_segment[0]
        est_segment_aligned = est_segment - est_segment[0]
        segment_ate = calculate_ate(gt_segment_aligned, est_segment_aligned)
        errors.append(segment_ate)
    return np.mean(errors) if errors else 0


def calculate_distance_error(gt_traj, est_traj):
    gt_dist = np.sum(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1))
    est_dist = np.sum(np.linalg.norm(np.diff(est_traj, axis=0), axis=1))
    return (np.abs(gt_dist - est_dist) / gt_dist) * 100 if gt_dist > 0 else 0


# --- 6. Plotting Functions ---


def plot_delta_comparison(gts, preds, save_path):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(gts[:, 0], label="Ground Truth", color="darkorange")
    plt.plot(preds[:, 0], label="HINNet Prediction", color="dodgerblue")
    plt.title("Delta Distance Comparison (per 2s window)")
    plt.ylabel("Delta Distance (m)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(gts[:, 1], label="Ground Truth", color="darkorange")
    plt.plot(preds[:, 1], label="HINNet Prediction", color="dodgerblue")
    plt.title("Delta Orientation Comparison (per 2s window)")
    plt.xlabel("Samples (2s Windows)")
    plt.ylabel("Delta Orientation (rad)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_trajectory_comparison(gt_traj, est_traj, save_path):
    plt.figure(figsize=(8, 8))
    plt.plot(
        gt_traj[:, 0],
        gt_traj[:, 1],
        label="Original Ground Truth",
        color="lightblue",
        linewidth=4,
    )
    plt.plot(
        est_traj[:, 0], est_traj[:, 1], label="HINNet Trajectory", color="limegreen"
    )
    plt.title("Trajectory Comparison")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(save_path)
    plt.close()


# --- 7. Main Execution Block ---

if __name__ == "__main__":
    DATA_DIRECTORY = "./data2/walk"
    PLOT_DIRECTORY = "./plots_hinnet"
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 60
    MODEL_PATH = "hinnet_model_lp_pr_fine.pth"

    os.makedirs(PLOT_DIRECTORY, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_files = [
        "./data2/walk/w1.hdf5",
        "./data2/walk/w68.hdf5",
        "./data2/walk/w3.hdf5",
        "./data2/walk/w4.hdf5",
        "./data2/walk/w5.hdf5",
        "./data2/walk/w7.hdf5",
        "./data2/walk/w6.hdf5",
        "./data2/walk/w9.hdf5",
        "./data2/walk/w10.hdf5",
        "./data2/walk/w50.hdf5",
        "./data2/walk/w12.hdf5",
        "./data2/walk/w63.hdf5",
        "./data2/walk/w66.hdf5",
        "./data2/walk/w67.hdf5",
        "./data2/walk/w69.hdf5",
        "./data2/walk/w71.hdf5",
        "./data2/walk/w73.hdf5",
        "./data2/walk/w74.hdf5",
        "./data2/walk/w42.hdf5",
        "./data2/walk/w79.hdf5",
        "./data2/walk/w39.hdf5",
        "./data2/walk/w40.hdf5",
        "./data2/walk/w41.hdf5",
        "./data2/walk/w43.hdf5",
        "./data2/walk/w44.hdf5",
        "./data2/walk/w45.hdf5",
        "./data2/walk/w46.hdf5",
        "./data2/walk/w48.hdf5",
        "./data2/walk/w49.hdf5",
        "./data2/walk/w47.hdf5",
        "./data2/walk/w52.hdf5",
        "./data2/walk/w15.hdf5",
        "./data2/walk/w26.hdf5",
        "./data2/walk/w27.hdf5",
        "./data2/walk/w28.hdf5",
        "./data2/walk/w32.hdf5",
        "./data2/walk/w31.hdf5",
        "./data2/walk/w16.hdf5",
        "./data2/walk/w17.hdf5",
        "./data2/walk/w18.hdf5",
        "./data2/walk/w22.hdf5",
        "./data2/walk/w20.hdf5",
        "./data2/walk/w61.hdf5",
        "./data2/walk/w33.hdf5",
        "./data2/walk/w34.hdf5",
        "./data2/walk/w35.hdf5",
        "./data2/walk/w25.hdf5",
        "./data2/walk/w53.hdf5",
        "./data2/walk/w37.hdf5",
        "./data2/walk/w55.hdf5",
        "./data2/walk/w56.hdf5",
        "./data2/walk/w57.hdf5",
        "./data2/walk/w38.hdf5",
        "./data2/walk/w59.hdf5",
        "./data2/walk/w60.hdf5",
    ]
    val_files = [
        "./data2/walk/w8.hdf5",
        "./data2/walk/w13.hdf5",
        "./data2/walk/w14.hdf5",
        "./data2/walk/w78.hdf5",
        "./data2/walk/w75.hdf5",
        "./data2/walk/w51.hdf5",
        "./data2/walk/w11.hdf5",
        "./data2/walk/w62.hdf5",
        "./data2/walk/w64.hdf5",
        "./data2/walk/w2.hdf5",
        "./data2/walk/w70.hdf5",
        "./data2/walk/w72.hdf5",
        "./data2/walk/w76.hdf5",
        "./data2/walk/w65.hdf5",
        "./data2/walk/w77.hdf5",
        "./data2/walk/w30.hdf5",
        "./data2/walk/w29.hdf5",
        "./data2/walk/w22.hdf5",
        "./data2/walk/w23.hdf5",
        "./data2/walk/w24.hdf5",
        "./data2/walk/w36.hdf5",
        "./data2/walk/w54.hdf5",
        "./data2/walk/w58.hdf5",
        "./data2/walk/w21.hdf5",
    ]
    random.shuffle(train_files)
    random.shuffle(val_files)

    print(f"Total training files: {len(train_files) + len(val_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    all_features_to_fit = []

    for train_file in tqdm(train_files):
        with h5py.File(train_file, "r") as f:
            acc, gyro = f["acc"][:], f["gyro"][:]

        CALIB_SAMPLES = 1000

        if len(acc) <= CALIB_SAMPLES + 2 * WINDOW_SIZE:
            continue

        acc = acc[CALIB_SAMPLES:]
        gyro = gyro[CALIB_SAMPLES:]

        df_imu = pd.DataFrame(
            {
                "acc_x": acc[:, 0],
                "acc_y": acc[:, 1],
                "acc_z": acc[:, 2],
                "gyro_x": gyro[:, 0],
                "gyro_y": gyro[:, 1],
                "gyro_z": gyro[:, 2],
            }
        )
        df_imu = check_and_convert_gyro_units(df_imu)
        norm_imu_data = roll_pitch_compensation(df_imu)

        norm_acc_x = norm_imu_data[:, 0]
        num_samples = len(norm_imu_data)

        start_index = WINDOW_SIZE
        end_index = num_samples - 2 * WINDOW_SIZE

        for i in range(start_index, end_index, 120):
            window_end = i + WINDOW_SIZE

            peak_ratio_features = np.zeros((WINDOW_SIZE, 2))
            for j in range(WINDOW_SIZE):
                sample_index = i + j
                acc_x_before_window = norm_acc_x[
                    sample_index - WINDOW_SIZE : sample_index
                ]
                p_ratio_before = calculate_peak_ratio(
                    acc_x_before_window, SAMPLING_FREQUENCY
                )
                acc_x_after_window = norm_acc_x[
                    sample_index : sample_index + WINDOW_SIZE
                ]
                p_ratio_after = calculate_peak_ratio(
                    acc_x_after_window, SAMPLING_FREQUENCY
                )
                peak_ratio_features[j, :] = [p_ratio_before, p_ratio_after]
            # Combine IMU data and peak ratio features
            window_features = norm_imu_data[i:window_end]
            features_to_fit = np.hstack(
                (window_features, peak_ratio_features)
            )
            all_features_to_fit.append(features_to_fit)

    all_features_to_fit = np.concatenate(all_features_to_fit, axis=0)
    scaler.fit(all_features_to_fit)
    print("Scaler fitted.")
    del all_features_to_fit

    train_dataset = HINNetDataset(file_paths=train_files, stride=30, scaler=scaler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataset = HINNetDataset(file_paths=val_files, stride=120, scaler=scaler)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    model = HINNet(input_dim=8).to(device)

    if os.path.exists(MODEL_PATH):
        print(f"Found pre-trained model at {MODEL_PATH}. Skipping training.")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"No pre-trained model found at {MODEL_PATH}. Starting training...")
        criterion = WeightedMSELoss(alpha=1.0).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
        print("\n--- Starting Training ---")
        best_val_loss = float("inf")
        for epoch in range(NUM_EPOCHS):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_l_mse, val_h_mse = evaluate_model(
                model, val_loader, criterion, device
            )
            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  ✓ New best model saved (val loss {val_loss:.6f})")
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Train {train_loss:.6f} | Val {val_loss:.6f} | "
                f"Val Δl {val_l_mse:.4f} | Val Δψ {val_h_mse:.4f} | "
                f"LR {current_lr:.6f}"
            )

        print("\n--- Training Finished ---")

    print("\n--- Running Final Evaluation on Validation Set ---")
    if not val_files:
        print("No validation files to evaluate. Exiting.")
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        for test_file_path in val_files:
            file_basename = os.path.splitext(os.path.basename(test_file_path))[0]
            print(f"\n--- Evaluating: {file_basename} ---")

            CALIB_SAMPLES = 1000
            with h5py.File(test_file_path, "r") as f:
                acc, gyro, pos, ori = (
                    f["acc"][:],
                    f["gyro"][:],
                    f["pos"][:],
                    f["ori"][:],
                )

            if len(acc) <= CALIB_SAMPLES + 2 * WINDOW_SIZE:
                print(
                    f"File {file_basename} is too short for peak ratio calculation. Skipping."
                )
                continue

            acc = acc[CALIB_SAMPLES:]
            gyro = gyro[CALIB_SAMPLES:]
            pos = pos[CALIB_SAMPLES:]
            ori = ori[CALIB_SAMPLES:]

            df_imu = pd.DataFrame(
                {
                    "acc_x": acc[:, 0],
                    "acc_y": acc[:, 1],
                    "acc_z": acc[:, 2],
                    "gyro_x": gyro[:, 0],
                    "gyro_y": gyro[:, 1],
                    "gyro_z": gyro[:, 2],
                }
            )
            df_imu = check_and_convert_gyro_units(df_imu)
            norm_imu_data = roll_pitch_compensation(df_imu)

            all_preds, all_gts = [], []

            norm_acc_x = norm_imu_data[:, 0]
            num_samples = len(norm_imu_data)

            start_index = WINDOW_SIZE
            end_index = num_samples - 2 * WINDOW_SIZE

            for i in range(start_index, end_index, WINDOW_SIZE):
                window_end = i + WINDOW_SIZE

                peak_ratio_features = np.zeros((WINDOW_SIZE, 2))
                for j in range(WINDOW_SIZE):
                    sample_index = i + j
                    acc_x_before_window = norm_acc_x[
                        sample_index - WINDOW_SIZE : sample_index
                    ]
                    p_ratio_before = calculate_peak_ratio(
                        acc_x_before_window, SAMPLING_FREQUENCY
                    )
                    acc_x_after_window = norm_acc_x[
                        sample_index : sample_index + WINDOW_SIZE
                    ]
                    p_ratio_after = calculate_peak_ratio(
                        acc_x_after_window, SAMPLING_FREQUENCY
                    )
                    peak_ratio_features[j, :] = [p_ratio_before, p_ratio_after]

                window_features = np.hstack(
                    (norm_imu_data[i:window_end], peak_ratio_features)
                )

                window_features_scaled = scaler.transform(window_features)

                pos_start, pos_end = pos[i], pos[i + WINDOW_SIZE - 1]
                ori_start, ori_end = ori[i], ori[i + WINDOW_SIZE - 1]
                delta_l = np.linalg.norm(pos_end[[0, 2]] - pos_start[[0, 2]])
                yaw_start_rad = np.deg2rad(ori_start[1])
                yaw_end_rad = np.deg2rad(ori_end[1])
                delta_psi_raw = yaw_end_rad - yaw_start_rad
                delta_psi = np.arctan2(np.sin(delta_psi_raw), np.cos(delta_psi_raw))
                all_gts.append([delta_l, delta_psi])

                with torch.no_grad():
                    window_tensor = (
                        torch.tensor(window_features_scaled, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                    prediction = model(window_tensor).squeeze().cpu().numpy()
                    all_preds.append(prediction)

            all_preds = np.array(all_preds)
            all_gts = np.array(all_gts)
            REMOVE_SOME = 15

            # The initial position and heading must be taken from within the valid data range
            initial_pos = pos[start_index + REMOVE_SOME * WINDOW_SIZE, [2, 0]]
            initial_displacement_vector = (
                pos[start_index + REMOVE_SOME * WINDOW_SIZE + WINDOW_SIZE - 1, [2, 0]]
                - pos[start_index + REMOVE_SOME * WINDOW_SIZE, [2, 0]]
            )
            dx = initial_displacement_vector[0]
            dz = initial_displacement_vector[1]
            initial_heading_rad = np.arctan2(dz, dx)

            all_gts = all_gts[REMOVE_SOME:]
            all_preds = all_preds[REMOVE_SOME:]

            gt_traj = trajectory_from_deltas(all_gts, initial_pos, initial_heading_rad)
            est_traj = trajectory_from_deltas(
                all_preds, initial_pos, initial_heading_rad
            )
            ate = calculate_ate(gt_traj, est_traj)
            rte = calculate_rte(gt_traj, est_traj)
            dist_err = calculate_distance_error(gt_traj, est_traj)
            print(
                f"ATE: {ate:.4f} m | RTE: {rte:.4f} m | Distance Error: {dist_err:.2f} %"
            )

            plot_delta_comparison(
                all_gts,
                all_preds,
                os.path.join(PLOT_DIRECTORY, f"{file_basename}_deltas.png"),
            )
            plot_trajectory_comparison(
                gt_traj,
                est_traj,
                os.path.join(PLOT_DIRECTORY, f"{file_basename}_trajectory.png"),
            )
