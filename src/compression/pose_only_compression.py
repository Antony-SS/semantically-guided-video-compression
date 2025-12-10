from analysis_core.extract_rgb_frames import extract_rgb_frames
from dataset_stream.dataset_stream import DatasetStream
from dataset_stream.dataset_writer import DatasetWriter
import numpy as np
from tqdm import tqdm

def pose_only_compression(original_dataset_path: str, compressed_dataset_path: str, xy_pose_resolution: float = 0.5, yaw_pose_resolution: float = 45.0):
    """
    Compress a dataset by only keeping the poses.
    Parameters
    ----------
    original_dataset_path: str
    compressed_dataset_path: str
    xy_pose_resolution: float
    yaw_pose_resolution: float
    """

    original_dataset = DatasetStream(original_dataset_path)
    compressed_dataset = DatasetWriter(compressed_dataset_path)
    original_poses = np.array(original_dataset.poses)

    # only using original to determine good bounds, in true online, you could just adjust the bounds as you go
    max_x = max(original_poses[:, 0])
    max_y = max(original_poses[:, 1])
    min_x = min(original_poses[:, 0])
    min_y = min(original_poses[:, 1])
    
    total_bins_x = int((max_x - min_x) / xy_pose_resolution)
    total_bins_y = int((max_y - min_y) / xy_pose_resolution)
    total_bins_yaw = int(360 / yaw_pose_resolution)

    x_edges = np.linspace(min_x, max_x, total_bins_x + 1)
    y_edges = np.linspace(min_y, max_y, total_bins_y + 1)
    yaw_edges = np.linspace(-180, 180, total_bins_yaw + 1)
    compressed_histogram = np.zeros((total_bins_x, total_bins_y, total_bins_yaw))

    for instance in tqdm(original_dataset.iterate(), total=len(original_dataset), desc="Compressing dataset"):
        pose = instance.pose
        x_bin = np.digitize(pose[0], x_edges) - 1
        y_bin = np.digitize(pose[1], y_edges) - 1
        yaw_bin = np.digitize(pose[2], yaw_edges) - 1
        
        # Clamp to valid range (digitize can return len(edges) for values >= max edge)
        x_bin = np.clip(x_bin, 0, total_bins_x - 1)
        y_bin = np.clip(y_bin, 0, total_bins_y - 1)
        yaw_bin = np.clip(yaw_bin, 0, total_bins_yaw - 1)
        
        if compressed_histogram[x_bin, y_bin, yaw_bin] == 0:
            compressed_histogram[x_bin, y_bin, yaw_bin] = 1
            compressed_dataset.write_instance(instance)
        else:
            continue

    return compressed_dataset

