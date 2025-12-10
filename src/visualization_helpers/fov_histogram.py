
import numpy as np
import cv2
from dataset_stream.dataset_stream import DatasetStream


def create_fov_histogram(dataset_path : str, 
                           output_histogram_path : str, 
                           resolution : float = 0.05, 
                           padding : float = 2.0,
                           exponential_scaling : bool = True, fov_degrees : float = 69.4, 
                           max_viewing_distance : float = 2.5) -> None:
    """
    Creates a histogram of the poses at which frames in the dataset are captured.
    Each frame contributes to all bins within its camera viewing cone.
    
    Parameters
    ----------
    dataset_path : str
        The path to the dataset.
    output_histogram_path : str
        The path to the output histogram.
    resolution : float, optional
        The resolution of the histogram in meters (default: 0.05).
    exponential_scaling : bool, optional
        Whether to use exponential scaling for the histogram (default: True).
    fov_degrees : float, optional
        Horizontal field of view in degrees (default: 69.4 for RealSense D435i RGB).
    max_viewing_distance : float, optional
        Maximum viewing distance in meters (default: 3.0).
    """

    dataset_stream = DatasetStream(dataset_path)
    poses = np.array(dataset_stream.poses)

    max_x = max(poses[:, 0]) + padding
    max_y = max(poses[:, 1]) + padding 
    min_x = min(poses[:, 0]) - padding
    min_y = min(poses[:, 1]) - padding

    padding_bins = int(padding / resolution)
    total_bins_x = int((max_x - min_x) / resolution) + 2 * padding_bins
    total_bins_y = int((max_y - min_y) / resolution) + 2 * padding_bins

    # Create histogram grid
    histogram = np.zeros((total_bins_x, total_bins_y))
    
    # Calculate bin centers
    x_edges = np.linspace(min_x, max_x, total_bins_x + 1)
    y_edges = np.linspace(min_y, max_y, total_bins_y + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    
    # Create meshgrid of bin centers
    X_grid, Y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    # FOV half-angle in radians
    fov_half_rad = np.radians(fov_degrees / 2.0)
    cos_fov_half = np.cos(fov_half_rad)
    

    for pose in poses:
        cam_x, cam_y, yaw = pose[0], pose[1], pose[2]
        
        yaw_rad = np.radians(yaw)
        forward_vec = np.array([np.cos(yaw_rad), np.sin(yaw_rad)]) # forward vector in FLU frame
        
        # Vector from camera to each bin center
        dx = X_grid - cam_x
        dy = Y_grid - cam_y
        dist_sq = dx**2 + dy**2
        
        # Distance check: within max_viewing_distance
        within_distance = dist_sq <= max_viewing_distance**2
        
        # Angle check: within FOV cone
        # Normalize direction vectors and compute dot product
        dist = np.sqrt(dist_sq + 1e-10)  # Add small epsilon to avoid division by zero
        to_bin_vec = np.stack([dx / dist, dy / dist], axis=-1)
        
        # Dot product with forward vector
        dot_product = (to_bin_vec[..., 0] * forward_vec[0] + 
                      to_bin_vec[..., 1] * forward_vec[1])
        
        # Check if angle is within FOV (dot product >= cos(half_fov))
        within_fov = dot_product >= cos_fov_half
        
        # Combine both conditions
        in_cone = within_distance & within_fov
        
        # Add contribution to bins within cone
        histogram[in_cone] += 1.0

    histogram = np.fliplr(np.flipud(histogram))
    x_edges = np.flip(x_edges)
    y_edges = np.flip(y_edges)

    origin_idx = np.argmin(np.abs(x_edges - 0))
    origin_idy = np.argmin(np.abs(y_edges - 0))

    clip_max = max(histogram.max(), 1000)
    histogram = np.clip(histogram, 0, clip_max)

    print(f"histogram max: {histogram.max()}, min: {histogram.min()}, clip_max: {clip_max}")

    if exponential_scaling:
        histogram = np.log(histogram + 1)

    histogram = (histogram - np.min(histogram)) / (np.max(histogram) - np.min(histogram)) * 255
    histogram = histogram.astype(np.uint8)

    print(f"histogram max: {histogram.max()}, min: {histogram.min()}")

    histo_image = cv2.applyColorMap(histogram, cv2.COLORMAP_JET)
    draw_origin_axes(histo_image, origin_idx, origin_idy, resolution)
    cv2.imwrite(output_histogram_path, histo_image)


def draw_origin_axes(histogram_image : np.ndarray, origin_idx : int, origin_idy : int, resolution : float = 0.05) -> None:
    size = 1.5 # meters
    step = size / resolution

    cv2.arrowedLine(histogram_image, (origin_idy, origin_idx), (origin_idy, int(origin_idx - step)), (0, 0, 255), 1)
    cv2.arrowedLine(histogram_image, (origin_idy, origin_idx), (int(origin_idy + step), origin_idx), (0, 0, 255), 1)

    return histogram_image
