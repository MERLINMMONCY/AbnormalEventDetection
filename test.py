import os
import re
import numpy as np
from keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import cv2

# Initialize mean squared error function
mse = MeanSquaredError()


def mean_squared_loss(x1, x2):
    """Compute Euclidean Distance Loss between input frame and the reconstructed frame."""
    diff = x1 - x2
    sq_diff = diff ** 2
    Sum = sq_diff.sum()
    mean_dist = np.sqrt(Sum) / np.prod(diff.shape)
    return mean_dist


def get_dynamic_threshold(test_data, model):
    """Calculate dynamic threshold based on reconstruction loss across all test data."""
    losses = []
    for batch in test_data:
        output = model.predict(np.expand_dims(batch, axis=0))
        loss = mean_squared_loss(batch, output)
        losses.append(loss)
    dynamic_threshold = np.mean(losses) + np.std(losses)
    print(f"Dynamic threshold calculated: {dynamic_threshold}")
    return dynamic_threshold


def parse_metadata(metadata_path):
    """Parse metadata file to extract frame ranges indicating anomalies."""
    video_frame_ranges = []
    with open(metadata_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "gt_frame" in line:
            frame_ranges = []
            match = re.search(r'\{end\+1\}\.gt_frame\s*=\s*\[(.*?)\];', line)
            if match:
                ranges = match.group(1).split(',')
                for r in ranges:
                    if ':' in r:
                        start, end = map(int, r.split(':'))
                        frame_ranges.extend(range(start, end + 1))
                    else:
                        frame_ranges.append(int(r.strip()))
            video_frame_ranges.append(frame_ranges)

    return video_frame_ranges


def get_frame_level_gt(video_root_path, metadata_path):
    """Construct frame-level ground truth array from metadata."""
    all_gt = []
    video_dirs = sorted([d for d in os.listdir(video_root_path) if
                         not d.endswith('_gt') and os.path.isdir(os.path.join(video_root_path, d))])
    video_frame_ranges = parse_metadata(metadata_path)

    for i, video_dir in enumerate(video_dirs):
        expected_anomalous_frames = video_frame_ranges[i] if i < len(video_frame_ranges) else []
        video_length = len(os.listdir(os.path.join(video_root_path, video_dir)))
        video_gt = np.zeros(video_length)

        for frame in expected_anomalous_frames:
            if frame <= len(video_gt):
                video_gt[frame - 1] = 1

        all_gt.append(video_gt)
        print(f"Processed ground truth for video: {video_dir}")

    return np.concatenate(all_gt)


def compute_eer(far, frr):
    """Compute Equal Error Rate (EER) from FAR and FRR."""
    min_dist = float('inf')
    eer = 0
    for item_far, item_frr in zip(far, frr):
        dist = abs(item_far - item_frr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_far + item_frr) / 2
    return eer


def calc_auc_overall(all_gt, all_pred):
    """Calculate and plot AUC and EER for the entire dataset."""
    all_gt = np.asarray(all_gt).ravel()
    all_pred = np.asarray(all_pred).ravel()

    if np.all(all_gt == 0):
        print("No anomalies in ground truth, skipping AUC and EER calculation.")
        return None, None

    if len(all_gt) != len(all_pred):
        if len(all_pred) > len(all_gt):
            all_pred = all_pred[:len(all_gt)]
        else:
            all_pred = np.pad(all_pred, (0, len(all_gt) - len(all_pred)), 'constant')

    fpr, tpr, _ = roc_curve(all_gt, all_pred, pos_label=1)
    overall_auc = auc(fpr, tpr)
    frr = 1 - tpr
    overall_eer = compute_eer(fpr, frr)

    print(f"Overall AUC: {overall_auc * 100:.2f}%, Overall EER: {overall_eer * 100:.2f}%")

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [1, 0], '--')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.title(f'AUC: {overall_auc:.3f}, EER: {overall_eer:.3f}')
    plt.savefig('auc_eer_plot_ucsd.png')
    plt.close()

    return overall_auc, overall_eer


# Load trained model
model = load_model(
    '/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/trained_models/AnomalyDetector_UCSD_Ped1_tds.h5',
    custom_objects={'mse': mse})

# Load test data
X_test = np.load('/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/Processed_Data/test.npy')
print(f"Original X_test shape: {X_test.shape}")

# Dynamic threshold calculation
dynamic_threshold = get_dynamic_threshold(X_test, model)

# Define paths for ground truth
video_root_path = '/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
metadata_path = os.path.join(video_root_path, 'UCSDped1.m')

# Get frame-level ground truth
all_gt = get_frame_level_gt(video_root_path, metadata_path)

# Anomaly detection and prediction
all_pred = []
max_batches = len(all_gt) // 10

for number, batch in enumerate(X_test):
    if number >= max_batches:
        print(f"Stopping at batch {number + 1} as it exceeds available ground truth data.")
        break

    output = model.predict(np.expand_dims(batch, axis=0))
    loss = mean_squared_loss(batch, output)
    print(f"Batch {number + 1}: Loss = {loss}")
    all_pred.extend([loss] * 10)

    # Compare each frame's prediction to ground truth
    for j in range(10):
        index = number * 10 + j
        if index >= len(all_gt):
            print(f"Skipping out-of-bounds index {index}")
            continue
        is_anomaly = loss > dynamic_threshold
        print(f"Frame {index + 1}: Predicted = {loss}, Ground Truth = {all_gt[index]}, Is Anomaly: {is_anomaly}")

# Process any remaining frames that do not fill a full batch
remaining_frames = len(all_gt) % 10
if remaining_frames > 0:
    print(f"Processing remaining {remaining_frames} frames.")
    last_batch = np.zeros((1, 10, 256, 256, 1), dtype=np.float32)
    for j in range(remaining_frames):
        frame = X_test[-1, j, :, :, :].squeeze()
        resized_frame = cv2.resize(frame, (256, 256))
        last_batch[0, j, :, :, 0] = resized_frame

    output = model.predict(last_batch)
    loss = mean_squared_loss(last_batch, output)
    all_pred.extend([loss] * remaining_frames)

# Calculate and log AUC and EER
calc_auc_overall(all_gt, all_pred[:len(all_gt)])

print("Anomaly detection completed.")
