from evaluation.geometric_coverage import compare_geometric_coverage
from visualization_helpers.fov_histogram import create_fov_histogram
from visualization_helpers.trajectory_histogram import create_trajectory_histogram
import argparse
import os

def run_geometric_compression_ratio(original_dataset_path: str, target_path: str, xy_pose_resolution: float = 0.5, yaw_pose_resolution: float = 45.0, output_path: str = "evaluation_outputs/") -> float:
    
    metrics = compare_geometric_coverage(original_dataset_path, target_path, xy_pose_resolution, yaw_pose_resolution, output_path)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--xy_pose_resolution", type=float, required=False, default=0.5)
    parser.add_argument("--yaw_pose_resolution", type=float, required=False, default=45.0)
    parser.add_argument("--output_path", type=str, required=False, default="evaluation_outputs/")
    args = parser.parse_args()
    run_geometric_compression_ratio(args.original_dataset_path, args.target_path, args.xy_pose_resolution, args.yaw_pose_resolution, args.output_path)

if __name__ == "__main__":
    main()