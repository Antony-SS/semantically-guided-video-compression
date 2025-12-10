from evaluation.geometric_coverage import compare_geometric_coverage
from evaluation.semantic_coverage_CLIP import evaluate_semantic_coverage_CLIP
from evaluation.compression_ratio import framewise_compression_ratio
from visualization_helpers.fov_histogram import create_fov_histogram
from visualization_helpers.trajectory_histogram import create_trajectory_histogram
import os
import argparse
import matplotlib.pyplot as plt

def run_full_comparison_suite(original_dataset_path: str, compressed_dataset_path: str, xy_pose_resolution: float = 1.0, yaw_pose_resolution: float = 45.0, n_clusters: int = 80, output_path: str = "evaluation_outputs/"):

    output_dir = os.path.join(output_path, f"{os.path.basename(original_dataset_path)}_vs_{os.path.basename(compressed_dataset_path)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # FOR THESE PASS THE OUTPUT PATH, NOT THE FULL DIR BECAUSE THEY ALREADY DO THIS IN THEIR FUNCTIONS
    geometric_coverage_metrics = compare_geometric_coverage(original_dataset_path, compressed_dataset_path, xy_pose_resolution, yaw_pose_resolution, output_path)
    compression_ratio_metrics = framewise_compression_ratio(original_dataset_path, compressed_dataset_path, output_path)
    semantic_coverage_metrics = evaluate_semantic_coverage_CLIP(original_dataset_path, compressed_dataset_path, n_clusters, output_path)

    # now run visualizations use output dir
    create_fov_histogram(original_dataset_path, os.path.join(output_dir, "fov_histogram_original.png"))
    create_trajectory_histogram(original_dataset_path, os.path.join(output_dir, "trajectory_histogram_original.png"))
    create_fov_histogram(compressed_dataset_path, os.path.join(output_dir, "fov_histogram_compressed.png"))
    create_trajectory_histogram(compressed_dataset_path, os.path.join(output_dir, "trajectory_histogram_compressed.png"))
    
    return geometric_coverage_metrics, semantic_coverage_metrics, compression_ratio_metrics

def plot_compression_ratio_vs_geometric_coverage(geometric_coverage_metrics: dict, compression_ratio_metrics: dict, output_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(geometric_coverage_metrics, compression_ratio_metrics, marker='o', linestyle='-', color='b')
    plt.xlabel("Geometric Coverage")
    plt.ylabel("Compression Ratio")
    plt.title("Compression Ratio vs Geometric Coverage")
    plt.savefig(os.path.join(output_dir, "compression_ratio_vs_geometric_coverage.png"))
    plt.close()

def plot_compression_ratio_vs_semantic_coverage(semantic_coverage_metrics: dict, compression_ratio_metrics: dict, output_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(semantic_coverage_metrics, compression_ratio_metrics, marker='o', linestyle='-', color='b')
    plt.xlabel("Semantic Coverage")
    plt.ylabel("Compression Ratio")
    plt.title("Compression Ratio vs Semantic Coverage")
    plt.savefig(os.path.join(output_dir, "compression_ratio_vs_semantic_coverage.png"))
    plt.close()

def plot_geometric_coverage_vs_semantic_coverage(geometric_coverage_metrics: dict, semantic_coverage_metrics: dict, output_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(geometric_coverage_metrics, semantic_coverage_metrics, marker='o', linestyle='-', color='b')
    plt.xlabel("Geometric Coverage")
    plt.ylabel("Semantic Coverage")
    plt.title("Geometric Coverage vs Semantic Coverage")
    plt.savefig(os.path.join(output_dir, "geometric_coverage_vs_semantic_coverage.png"))
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--compressed_dataset_path", type=str, required=True)
    parser.add_argument("--xy_pose_resolution", type=float, required=False, default=1.0)
    parser.add_argument("--yaw_pose_resolution", type=float, required=False, default=45.0)
    parser.add_argument("--n_clusters", type=int, required=False, default=80)
    parser.add_argument("--output_path", type=str, required=False, default="evaluation_outputs/")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_full_comparison_suite(args.original_dataset_path, args.compressed_dataset_path, args.xy_pose_resolution, args.yaw_pose_resolution, args.n_clusters, args.output_path)