from visualization.frame_histogram import create_frame_histogram
import argparse

import argparse
import os


def run_create_capture_histogram(dataset_path : str, output_histogram_path : str, resolution : float = 0., exponential_scaling : bool = True, padding : float = 2.0) -> None:

    output_histogram_path = os.path.join(output_histogram_path, os.path.basename(dataset_path))
    if not os.path.exists(output_histogram_path):
        os.makedirs(output_histogram_path)

    create_frame_histogram(dataset_path, os.path.join(output_histogram_path, f"capture_histogram_{resolution}_res.png"), resolution=resolution, exponential_scaling=exponential_scaling, padding=padding)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False, default="analysis_outputs/")
    parser.add_argument("--resolution", type=float, required=False, default=0.05)
    parser.add_argument("--exponential_scaling", action="store_true", required=False, default=True)
    parser.add_argument("--padding", type=float, required=False, default=2.0)
    return parser.parse_args()

def main():
    args = parse_args()
    run_create_capture_histogram(args.dataset_path, args.output_path, args.resolution, args.exponential_scaling, args.padding)

if __name__ == "__main__":
    main()