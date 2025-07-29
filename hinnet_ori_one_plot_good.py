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
from ahrs.filters import Madgwick
from ahrs.common import Quaternion
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR


# --- Constants ---
# These constants are derived from the HINNet paper.
SAMPLING_FREQUENCY = 60.0
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(WINDOW_SECONDS * SAMPLING_FREQUENCY)
G = 9.81
KAPPA_MIN = 0.1  # lower bound for κ
KAPPA_MAX = 20.0  # upper bound for κ


# Frequency ranges for peak detection, based on typical human gait.
SWING_FREQ_RANGE = (0.5, 1.25)
STEP_FREQ_RANGE = (1.5, 2.5)


# --- 1. Preprocessing and Feature Engineering ---


def check_and_convert_gyro_units(df_imu):
    gyro_data = df_imu[["gyro_x", "gyro_y", "gyro_z"]].values
    max_abs_val = np.max(np.abs(gyro_data))

    if max_abs_val > 10.0:
        df_imu[["gyro_x", "gyro_y", "gyro_z"]] = np.deg2rad(gyro_data)

    return df_imu


def roll_pitch_compensation(df_imu, fs=SAMPLING_FREQUENCY, beta=0.1):
    """
    Performs roll and pitch compensation on raw IMU data to align the z-axis with gravity.
    """
    gyro_rad = df_imu[["gyro_x", "gyro_y", "gyro_z"]].values
    acc = df_imu[["acc_x", "acc_y", "acc_z"]].values

    madgwick = Madgwick(frequency=fs, beta=beta)
    q = np.zeros((len(df_imu), 4))
    q[0] = [1.0, 0.0, 0.0, 0.0]

    for t in range(1, len(df_imu)):
        q[t] = madgwick.updateIMU(q[t - 1], gyr=gyro_rad[t], acc=acc[t])

    eul = np.array([Quaternion(qi).to_angles() for qi in q])
    roll, pitch = eul[:, 0], eul[:, 1]

    acc_norm = np.zeros_like(acc)
    gyro_norm = np.zeros_like(gyro_rad)

    for t in range(len(df_imu)):
        phi, theta = roll[t], pitch[t]
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)

        R_a = np.array(
            [
                [ctheta, 0, -stheta],
                [sphi * stheta, cphi, sphi * ctheta],
                [cphi * stheta, -sphi, cphi * ctheta],
            ]
        )
        R_w = np.array(
            [[1, 0, -stheta], [0, cphi, sphi * ctheta], [0, -sphi, cphi * ctheta]]
        )

        acc_norm[t] = R_a.T @ acc[t]
        gyro_norm[t] = R_w.T @ gyro_rad[t]

    return np.hstack((acc_norm, gyro_norm))


def calculate_peak_ratios(acc_norm_x, fs=SAMPLING_FREQUENCY, window_size=WINDOW_SIZE):
    """
    Calculates the 'peak ratio' feature as described in Section 2.2 of the paper.
    """
    num_samples = len(acc_norm_x)
    peak_ratios = np.zeros((num_samples, 2))
    freqs = fftfreq(window_size, 1 / fs)

    swing_mask = (freqs >= SWING_FREQ_RANGE[0]) & (freqs <= SWING_FREQ_RANGE[1])
    step_mask = (freqs >= STEP_FREQ_RANGE[0]) & (freqs <= STEP_FREQ_RANGE[1])

    for t in range(num_samples):
        start_before = max(0, t - window_size)
        window_before = acc_norm_x[start_before:t]
        if len(window_before) == window_size:
            fft_vals = np.abs(fft(window_before))
            p_swing = np.max(fft_vals[swing_mask]) if np.any(swing_mask) else 0
            p_stepping = np.max(fft_vals[step_mask]) if np.any(step_mask) else 0
            peak_ratios[t, 0] = p_swing / (p_stepping + 1e-9)

        start_after = t + 1
        window_after = acc_norm_x[start_after : start_after + window_size]
        if len(window_after) == window_size:
            fft_vals = np.abs(fft(window_after))
            p_swing = np.max(fft_vals[swing_mask]) if np.any(swing_mask) else 0
            p_stepping = np.max(fft_vals[step_mask]) if np.any(step_mask) else 0
            peak_ratios[t, 1] = p_swing / (p_stepping + 1e-9)

    return peak_ratios


# --- 2. PyTorch Dataset ---


class HINNetDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing the HINNet data.
    Uses a sliding window (stride=1) for data augmentation during training.
    """

    def __init__(self, file_paths, stride=120):
        self.feature_windows = []
        self.labels = []
        self.file_paths = file_paths
        self.stride = stride
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
                peak_ratios = calculate_peak_ratios(norm_imu_data[:, 0])
                all_features = np.hstack((norm_imu_data, peak_ratios))

                num_samples = len(all_features)
                # Create overlapping windows (stride=1)
                for i in range(0, num_samples - WINDOW_SIZE, self.stride):
                    window_end = i + WINDOW_SIZE
                    self.feature_windows.append(all_features[i:window_end])
                    pos_start, pos_end = pos[i], pos[i + WINDOW_SIZE - 1]
                    ori_start, ori_end = ori[i], ori[i + WINDOW_SIZE - 1]

                    delta_l = np.linalg.norm(pos_end[[0, 2]] - pos_start[[0, 2]])

                    yaw_start_deg = ori_start[1]
                    yaw_end_deg = ori_end[1]

                    yaw_start_rad = np.deg2rad(yaw_start_deg)
                    yaw_end_rad = np.deg2rad(yaw_end_deg)

                    delta_psi_raw = yaw_end_rad - yaw_start_rad
                    delta_psi = np.arctan2(np.sin(delta_psi_raw), np.cos(delta_psi_raw))

                    self.labels.append(np.array([delta_l, delta_psi]))

    def __len__(self):
        return len(self.feature_windows)

    def __getitem__(self, idx):
        features = torch.tensor(self.feature_windows[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, labels


# --- 3. PyTorch Model ---


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

class TrainableKappaMSE(nn.Module):
    """
    κ-weighted MSE loss for (Δl, Δψ) with κ as a learnable parameter.
    ▸ κ is kept strictly positive by optimising log(κ).
    """

    def __init__(self, init_kappa: float = 15.0):
        super().__init__()
        # we optimise log κ so that κ = exp(log_κ) is always > 0
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(init_kappa)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : (B, 2) – (Δl̂, Δψ̂)
        target : (B, 2) – (Δl , Δψ )
        """
        trans_err = (pred[:, 0] - target[:, 0]) ** 2  # Δl error
        head_err = (pred[:, 1] - target[:, 1]) ** 2  # Δψ error
        kappa = torch.exp(self.log_kappa)  # ensure κ > 0
        loss = trans_err*kappa + head_err
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
        # ---------- NEW: hard-clamp κ ---------------------------------
        if hasattr(criterion, "log_kappa"):
            with torch.no_grad():
                criterion.log_kappa.clamp_(np.log(KAPPA_MIN), np.log(KAPPA_MAX))
        # --------------------------------------------------------------
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
    """
    Reconstructs a 2D trajectory using the corrected logic.
    """
    positions = [initial_pos]
    heading = initial_heading
    for delta_l, delta_psi in predictions:
        # --- FIX: Use average heading for smoother, more accurate turns ---
        avg_heading = heading + delta_psi / 2.0
        dx_step = delta_l * np.sin(avg_heading)
        dz_step = delta_l * np.cos(avg_heading)

        positions.append(positions[-1] + np.array([dx_step, dz_step]))

        # --- FIX: Update heading for the *next* window using the correct convention ---
        heading += delta_psi

    return np.array(positions)


def calculate_ate(gt_traj, est_traj):
    error = gt_traj - est_traj
    squared_error = np.sum(error**2, axis=1)
    ate = np.sqrt(np.mean(squared_error))
    return ate


def calculate_rte(gt_traj, est_traj, interval_sec=60, fs=SAMPLING_FREQUENCY):
    # The interval is now based on the number of non-overlapping windows
    interval = int(interval_sec / WINDOW_SECONDS)
    if interval == 0:
        return 0  # Avoid division by zero if interval is too short

    errors = []
    for i in range(0, len(gt_traj) - interval, interval):
        gt_segment = gt_traj[i : i + interval]
        est_segment = est_traj[i : i + interval]
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
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 80
    MODEL_PATH = "hinnet_model.pth"

    os.makedirs(PLOT_DIRECTORY, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = [
        os.path.join(DATA_DIRECTORY, f)
        for f in os.listdir(DATA_DIRECTORY)
        if f.endswith(".hdf5")
    ]
    np.random.shuffle(all_files)
    split_idx = int(0.75 * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    train_dataset = HINNetDataset(file_paths=train_files, stride=10)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    val_dataset = HINNetDataset(file_paths=val_files, stride=10)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation data
        num_workers=4,
        pin_memory=True,
    )
    model = HINNet().to(device)
    # FREEZE_EPOCHS = 10
    if os.path.exists(MODEL_PATH):
        print(f"Found pre-trained model at {MODEL_PATH}. Skipping training.")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"No pre-trained model found at {MODEL_PATH}. Starting training...")
        criterion = TrainableKappaMSE(init_kappa=15.0).to(device)
        optim_params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.Adam(optim_params, lr=LEARNING_RATE)
        scheduler = MultiStepLR(optimizer, milestones=[20, 50], gamma=0.1)
        print("\n--- Starting Training ---")
        best_val_loss = float("inf")          # keep track of the best value
        for epoch in range(NUM_EPOCHS):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_l_mse, val_h_mse = evaluate_model(
                model, val_loader, criterion, device
            )
            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  ✓ New best model saved (val loss {val_loss:.6f})")
            current_lr = optimizer.param_groups[0]["lr"]
            kappa_val = (
                torch.exp(criterion.log_kappa).item()
                if hasattr(criterion, "log_kappa")
                else 0.0
            )
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Train {train_loss:.6f} | Val {val_loss:.6f} | "
                f"Val Δl {val_l_mse:.4f} | Val Δψ {val_h_mse:.4f} | "
                f"κ = {kappa_val:5.2f} | LR {current_lr:.6f}"
            )

        print("\n--- Training Finished ---")
        # torch.save(model.state_dict(), MODEL_PATH)
        # print(f"Model saved to {MODEL_PATH}")

    print("\n--- Running Final Evaluation on Validation Set ---")
    if not val_files:
        print("No validation files to evaluate. Exiting.")
    else:
        # eval_model = HINNet().to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        for test_file_path in val_files:
            file_basename = os.path.splitext(os.path.basename(test_file_path))[0]
            print(f"\n--- Evaluating: {file_basename} ---")

            CALIB_SAMPLES = 2000
            # --- FIX: Manually process the test file for non-overlapping windows ---
            with h5py.File(test_file_path, "r") as f:
                acc, gyro, pos, ori = (
                    f["acc"][:],
                    f["gyro"][:],
                    f["pos"][:],
                    f["ori"][:],
                )

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
            peak_ratios = calculate_peak_ratios(norm_imu_data[:, 0])
            all_features = np.hstack((norm_imu_data, peak_ratios))

            all_preds, all_gts = [], []

            # Create non-overlapping windows
            for i in range(0, len(all_features) - WINDOW_SIZE, WINDOW_SIZE):
                window_features = all_features[i : i + WINDOW_SIZE]

                # Get ground truth label for this non-overlapping window
                pos_start, pos_end = pos[i], pos[i + WINDOW_SIZE - 1]
                ori_start, ori_end = ori[i], ori[i + WINDOW_SIZE - 1]
                delta_l = np.linalg.norm(pos_end[[0, 2]] - pos_start[[0, 2]])
                yaw_start_rad = np.deg2rad(ori_start[1])
                yaw_end_rad = np.deg2rad(ori_end[1])
                delta_psi_raw = yaw_end_rad - yaw_start_rad
                delta_psi = np.arctan2(np.sin(delta_psi_raw), np.cos(delta_psi_raw))
                all_gts.append([delta_l, delta_psi])

                # Run model prediction
                with torch.no_grad():
                    window_tensor = (
                        torch.tensor(window_features, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                    prediction = model(window_tensor).squeeze().cpu().numpy()
                    all_preds.append(prediction)

            all_preds = np.array(all_preds)
            all_gts = np.array(all_gts)

            initial_pos = pos[0, [0, 2]]
            initial_displacement_vector = pos[WINDOW_SIZE - 1, [0, 2]] - pos[0, [0, 2]]
            dx = initial_displacement_vector[0]
            dz = initial_displacement_vector[1]
            initial_heading_rad = np.arctan2(dx, dz)

            estimated_trajectory = reconstruct_trajectory(
                all_preds, initial_pos, initial_heading_rad
            )

            # For metrics, compare against the GT path downsampled to the same 2s interval
            gt_path_for_metrics = pos[::WINDOW_SIZE, [0, 2]]
            # Ensure paths have same length for comparison
            min_len = min(len(gt_path_for_metrics), len(estimated_trajectory))
            gt_path_for_metrics = gt_path_for_metrics[:min_len]
            estimated_trajectory_for_metrics = estimated_trajectory[:min_len]

            ate = calculate_ate(gt_path_for_metrics, estimated_trajectory_for_metrics)
            rte = calculate_rte(gt_path_for_metrics, estimated_trajectory_for_metrics)
            dist_err = calculate_distance_error(
                gt_path_for_metrics, estimated_trajectory_for_metrics
            )

            print(
                f"ATE: {ate:.4f} m | RTE: {rte:.4f} m | Distance Error: {dist_err:.2f} %"
            )

            delta_plot_path = os.path.join(
                PLOT_DIRECTORY, f"{file_basename}_deltas.png"
            )
            traj_plot_path = os.path.join(
                PLOT_DIRECTORY, f"{file_basename}_trajectory.png"
            )

            plot_delta_comparison(all_gts, all_preds, delta_plot_path)

            # Plot the full original GT path for visual comparison
            original_gt_plot_path = pos[:, [0, 2]]
            plot_trajectory_comparison(
                original_gt_plot_path, estimated_trajectory, traj_plot_path
            )
            print(f"Saved plots to {delta_plot_path} and {traj_plot_path}")