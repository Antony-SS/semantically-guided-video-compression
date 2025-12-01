from evaluation.semantic_coverage import evaluate_semantic_coverage
import argparse
import os

def run_semantic_compression_ratio(original_dataset_path: str, compressed_dataset_path: str, n_clusters: int) -> float:
    return evaluate_semantic_coverage(original_dataset_path, compressed_dataset_path, n_clusters)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--compressed_dataset_path", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, required=False, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    run_semantic_compression_ratio(args.original_dataset_path, args.compressed_dataset_path, args.n_clusters)

if __name__ == "__main__":
    main()