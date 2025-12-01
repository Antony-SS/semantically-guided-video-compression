from dataset_stream.dataset_stream import DatasetStream
import numpy as np
import json
import os


def compare_geometric_coverage(original_dataset_path: str, compressed_dataset_path: str, xy_pose_resolution: float = 0.05, output_path: str = "evaluation_outputs/") -> float:
    """
    Compare the ratio between the geometric coverage of the original and compressed datasets.
    Geometric coverage is defined as the ratio of the number of bins covered by the original dataset to the number of bins covered by the compressed dataset.
    A bin is considered covered if there exists at least one pose in that bin.
    Parameters
    ----------
    original_dataset_path: str
        The path to the original dataset.
    compressed_dataset_path: str
        The path to the compressed dataset.
    xy_pose_resolution: float
        The resolution of the xy poses.
    yaw_pose_resolution: float
        The resolution of the yaw poses.
    Returns
    -------
    geometric_coverage_ratio: float
        The ratio between the geometric coverage of the original dataset and the compressed dataset.
    """
    original_dataset_stream = DatasetStream(original_dataset_path)
    compressed_dataset_stream = DatasetStream(compressed_dataset_path)

    original_base_path = os.path.basename(original_dataset_path)
    compressed_base_path = os.path.basename(compressed_dataset_path)

    output_dir = os.path.join(output_path, original_base_path, compressed_base_path, "geometric_coverage")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get the poses from the datasets
    original_poses = np.array(original_dataset_stream.poses)
    compressed_poses = np.array(compressed_dataset_stream.poses)

    # find min and max x and y, only have to do this for original b/c compressed is a subset
    min_x = min(original_poses[:, 0])
    max_x = max(original_poses[:, 0])
    min_y = min(original_poses[:, 1])
    max_y = max(original_poses[:, 1])

    total_x_bins = int((max_x - min_x) / xy_pose_resolution)
    total_y_bins = int((max_y - min_y) / xy_pose_resolution)
 
    x_edges = np.linspace(min_x, max_x, total_x_bins + 1)
    y_edges = np.linspace(min_y, max_y, total_y_bins + 1)

    # discretize yaw into 8 bins, consider all angles in a bin 'covered' if there exists one pose in that bin
    yaw_edges = np.linspace(-180, 180, 8) # this is a good default

    # 3D histogram of poses, each bin is x,y,yaw, make them the same shape for comparison
    original_histogram = discretized_poses(original_poses, x_edges, y_edges, yaw_edges)
    compressed_histogram = discretized_poses(compressed_poses, x_edges, y_edges, yaw_edges)
    
    # compare coverage of original and compressed histograms
    original_coverage = np.sum(original_histogram > 0)
    compressed_coverage = np.sum(compressed_histogram > 0)
    geometric_coverage_ratio = original_coverage / compressed_coverage

    metrics = {
        "original_dataset_path": original_dataset_path,
        "compressed_dataset_path": compressed_dataset_path,
        "xy_pose_resolution": xy_pose_resolution,
        "geometric_coverage_ratio": geometric_coverage_ratio,
    }

    log_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    return metrics

def log_metrics(metrics: dict, log_file: str = "metrics.json"):
    """
    Log metrics to a file.
    """
    with open(log_file, "w") as f:
        json.dump(metrics, f)
    print(f"Logged metrics to {log_file}")


def discretized_poses(poses: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, yaw_edges: np.ndarray) -> np.ndarray:
    """
    Discretize the poses into a 3D histogram, each bin is x,y,yaw.
    Parameters
    ----------
    poses: np.ndarray
        The poses to discretize.
    x_edges: np.ndarray
    y_edges: np.ndarray
    yaw_edges: np.ndarray
        The edges of the yaw bins.
    Returns
    -------
    histogram: np.ndarray
        The 3D histogram of the poses.
    """
    histogram, _ = np.histogramdd(poses, bins=(x_edges, y_edges, yaw_edges))
    return histogram

