from dataset_stream.dataset_stream import DatasetStream
import numpy as np
import cv2
from visualization_helpers.fov_histogram import draw_origin_axes
from tqdm import tqdm


def create_trajectory_histogram(dataset_path : str, 
                                output_histogram_path : str, 
                                robot_radius : float = 0.25, 
                                resolution : float = 0.05, 
                                padding : float = 2.0, 
                                exponential_scaling : bool = True,
                                clip_max : float = 1000) -> None:

    dataset_stream = DatasetStream(dataset_path)
    poses = np.array(dataset_stream.poses)

    max_x = max(poses[:, 0]) + padding
    max_y = max(poses[:, 1]) + padding 
    min_x = min(poses[:, 0]) - padding
    min_y = min(poses[:, 1]) - padding

    total_bins_x = int((max_x - min_x) / resolution)
    total_bins_y = int((max_y - min_y) / resolution)

    # Create histogram grid
    x_left_edges = np.linspace(min_x, max_x, total_bins_x + 1)
    y_left_edges = np.linspace(min_y, max_y, total_bins_y + 1)
    x_centers = (x_left_edges[:-1] + x_left_edges[1:]) / 2.0
    y_centers = (y_left_edges[:-1] + y_left_edges[1:]) / 2.0

    X_grid, Y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')

    histogram = np.zeros((total_bins_x, total_bins_y))

    for i, pose in tqdm(enumerate(poses), total=len(poses), desc="Creating trajectory histogram"):
        cam_x, cam_y = pose[0], pose[1]
        distance = np.sqrt((X_grid - cam_x)**2 + (Y_grid - cam_y)**2)
        within_radius = distance <= robot_radius
        histogram[within_radius] += 1.0

    histogram = np.fliplr(np.flipud(histogram))
    x_left_edges = np.flip(x_left_edges)
    y_left_edges = np.flip(y_left_edges)

    origin_idx = np.argmin(np.abs(x_left_edges - 0))
    origin_idy = np.argmin(np.abs(y_left_edges - 0))

    histogram = np.clip(histogram, 0, clip_max)

    if exponential_scaling:
        histogram = np.log(histogram + 1)

    histogram = (histogram - np.min(histogram)) / (np.max(histogram) - np.min(histogram)) * 255
    histogram = histogram.astype(np.uint8)

    histo_image = cv2.applyColorMap(histogram, cv2.COLORMAP_JET)
    draw_origin_axes(histo_image, origin_idx, origin_idy, resolution)
    print(f"Writing trajectory histogram to {output_histogram_path}")
    cv2.imwrite(output_histogram_path, histo_image)