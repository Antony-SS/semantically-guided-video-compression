import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset_stream.dataset_stream import DatasetStream


def create_cuboid_density_visualization(dataset_path: str, output_path: str = None, 
                                       bins: int = 15, min_point_size: float = 10.0, 
                                       max_point_size: float = 200.0) -> None:
    """
    Creates a 3D histogram visualization showing density hotspots where the robot took pictures.
    
    Parameters
    ----------
    dataset_path : str
        The path to the dataset.
    output_path : str, optional
        The path to save the output visualization. If None, displays interactively.
    bins : int, optional
        Number of bins for each dimension (default: 15).
    min_point_size : float, optional
        Minimum size of points in the scatter plot (default: 10.0).
    max_point_size : float, optional
        Maximum size of points in the scatter plot (default: 200.0). Size scales with density.
    """
    dataset_stream = DatasetStream(dataset_path)
    poses = np.array(dataset_stream.poses)
    
    x = poses[:, 0]
    y = poses[:, 1]
    yaw = poses[:, 2]
    
    # Create 3D histogram
    hist, edges = np.histogramdd(poses, bins=bins)
    
    # Get bin centers
    x_edges = edges[0]
    y_edges = edges[1]
    yaw_edges = edges[2]
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    yaw_centers = (yaw_edges[:-1] + yaw_edges[1:]) / 2.0
    
    # Find non-zero bins and their counts
    non_zero_indices = np.nonzero(hist)
    counts = hist[non_zero_indices]
    
    # Get coordinates for non-zero bins
    x_coords = x_centers[non_zero_indices[0]]
    y_coords = y_centers[non_zero_indices[1]]
    yaw_coords = yaw_centers[non_zero_indices[2]]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use percentile-based normalization to emphasize outliers
    count_min = counts.min()
    count_max = counts.max()
    
    # Normalize counts
    counts_normalized = (counts - count_min) / (count_max - count_min + 1e-8)
    counts_normalized = np.clip(counts_normalized, 0, 1)
    
    # Scale point sizes based on density (linear interpolation between min and max)
    if counts.max() > counts.min():
        point_sizes = min_point_size + (max_point_size - min_point_size) * (counts - counts.min()) / (counts.max() - counts.min())
    else:
        point_sizes = np.full_like(counts, min_point_size)
    
    # Scatter plot with color and size based on density
    # Use counts_for_color for colormap mapping, but display actual counts in colorbar
    scatter = ax.scatter(x_coords, y_coords, yaw_coords, 
                        c=counts_normalized, cmap='viridis', 
                        s=point_sizes, 
                        alpha=0.8, edgecolors='none', vmin=0, vmax=1)
    
    # Create colorbar with actual count values
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels([f'{val:.0f}' for val in np.linspace(count_min, count_max, 6)])
    cbar.set_label('Number of Pictures', rotation=270, labelpad=20)
    
    ax.set_xlabel('X (meters)', labelpad=10)
    ax.set_ylabel('Y (meters)', labelpad=10)
    ax.set_zlabel('Yaw (degrees)', labelpad=10)
    ax.set_title('3D Pose Density Histogram (Hotspots)')
    
    # Set equal aspect ratio for xy plane only
    xy_max_range = np.array([x.max()-x.min(), y.max()-y.min()]).max() / 2.0
    yaw_range = (yaw.max() - yaw.min()) / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_yaw = (yaw.max()+yaw.min()) * 0.5
    ax.set_xlim(mid_x - xy_max_range, mid_x + xy_max_range)
    ax.set_ylim(mid_y - xy_max_range, mid_y + xy_max_range)
    ax.set_zlim(mid_yaw - yaw_range, mid_yaw + yaw_range)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

