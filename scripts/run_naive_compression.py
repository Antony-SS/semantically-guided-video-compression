from compression.naive_compression import naive_compression
import argparse
import os
import json

def run_naive_compression(dataset_path: str, output_path: str, downsample_factor: int = 10):
    discard_frames = naive_compression(dataset_path, output_path, downsample_factor)
    metrics = {
        "original_dataset_path": dataset_path,
        "compressed_dataset_path": output_path,
        "downsample_factor": downsample_factor,
    }
    log_metrics(metrics, os.path.join(output_path, "compression_metadata.json"))
    log_discard_frames(discard_frames, os.path.join(output_path, "discarded_frames.txt"))
    return metrics

def log_metrics(metrics: dict, log_file: str = "metrics.json"):
    """
    Log metrics to a file.
    """
    with open(log_file, "w") as f:
        json.dump(metrics, f)

def log_discard_frames(discard_frames: list, log_file: str = "discarded_frames.txt"):
    """
    Log discarded frames to a file.
    """
    with open(log_file, "w") as f:
        for frame_path in discard_frames:
            f.write(f"{frame_path}\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--downsample_factor", type=int, required=False, default=30)
    return parser.parse_args()

def main():
    args = parse_args()
    run_naive_compression(args.dataset_path, args.output_path, args.downsample_factor)

if __name__ == "__main__":
    main()