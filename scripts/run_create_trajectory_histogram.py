from visualization.trajectory_histogram import create_trajectory_histogram
import argparse
import os


def run_create_trajectory_histogram(dataset_path: str, output_path: str = "analysis_outputs/", robot_radius: float = 0.25, resolution: float = 0.05, padding: float = 2.0, exponential_scaling: bool = True, clip_max: float = 1000) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, os.path.basename(dataset_path), f"trajectory_histogram_{resolution}_res.png")
    create_trajectory_histogram(dataset_path, output_path, robot_radius, resolution, padding, exponential_scaling, clip_max)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False, default="analysis_outputs/")
    parser.add_argument("--robot_radius", type=float, required=False, default=0.3)
    parser.add_argument("--resolution", type=float, required=False, default=0.05)
    parser.add_argument("--padding", type=float, required=False, default=2.0)
    parser.add_argument("--exponential_scaling", type=bool, required=False, default=True)
    parser.add_argument("--clip_max", type=float, required=False, default=1000)
    return parser.parse_args()

def main():
    args = parse_args()
    run_create_trajectory_histogram(args.dataset_path, args.output_path, args.robot_radius, args.resolution, args.padding, args.exponential_scaling, args.clip_max)

if __name__ == "__main__":
    main()