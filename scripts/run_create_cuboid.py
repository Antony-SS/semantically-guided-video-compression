from visualization.cuboid_density import create_cuboid_density_visualization
import argparse
import os


def run_create_cuboid_density(dataset_path: str, output_path: str = "analysis_outputs/", 
                             bins: int = 15, min_point_size: float = 10.0, 
                             max_point_size: float = 200.0, color_power: float = 2.0,
                             percentile_clip: float = 95.0) -> None:
    """
    Creates a 3D cuboid density histogram visualization for the dataset.
    
    Parameters
    ----------
    dataset_path : str
        The path to the dataset.
    output_path : str, optional
        The output directory path (default: "analysis_outputs/").
    bins : int, optional
        Number of bins for each dimension (default: 15).
    min_point_size : float, optional
        Minimum size of points in the scatter plot (default: 10.0).
    max_point_size : float, optional
        Maximum size of points in the scatter plot (default: 200.0).
    color_power : float, optional
        Power scaling for colorization to emphasize hotspots (default: 2.0).
    percentile_clip : float, optional
        Percentile to use as maximum for color scaling (default: 95.0).
    """
    output_dir = os.path.join(output_path, os.path.basename(dataset_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, "cuboid_density_visualization.png")
    create_cuboid_density_visualization(
        dataset_path=dataset_path,
        output_path=output_file,
        bins=bins,
        min_point_size=min_point_size,
        max_point_size=max_point_size,
        color_power=color_power,
        percentile_clip=percentile_clip
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False, default="analysis_outputs/")
    parser.add_argument("--bins", type=int, required=False, default=15)
    parser.add_argument("--min_point_size", type=float, required=False, default=10.0)
    parser.add_argument("--max_point_size", type=float, required=False, default=200.0)
    parser.add_argument("--color_power", type=float, required=False, default=2.0)
    parser.add_argument("--percentile_clip", type=float, required=False, default=95.0)
    return parser.parse_args()


def main():
    args = parse_args()
    run_create_cuboid_density(
        args.dataset_path, 
        args.output_path, 
        args.bins, 
        args.min_point_size,
        args.max_point_size,
        args.color_power,
        args.percentile_clip
    )


if __name__ == "__main__":
    main()
