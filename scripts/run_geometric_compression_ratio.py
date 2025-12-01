from evaluation.geometric_coverage import compare_geometric_coverage
import argparse
import os

def run_geometric_compression_ratio(original_dataset_path: str, compressed_dataset_path: str, xy_pose_resolution: float = 0.5, output_path: str = "evaluation_outputs/") -> float:
    return compare_geometric_coverage(original_dataset_path, compressed_dataset_path, xy_pose_resolution, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--compressed_dataset_path", type=str, required=True)
    parser.add_argument("--xy_pose_resolution", type=float, required=False, default=0.05)
    parser.add_argument("--output_path", type=str, required=False, default="evaluation_outputs/")
    args = parser.parse_args()
    run_geometric_compression_ratio(args.original_dataset_path, args.compressed_dataset_path, args.xy_pose_resolution, args.output_path)

if __name__ == "__main__":
    main()