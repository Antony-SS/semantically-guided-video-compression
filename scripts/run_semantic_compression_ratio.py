from evaluation.semantic_coverage_CLIP import evaluate_semantic_coverage_CLIP
import argparse

def run_semantic_compression_ratio(original_dataset_path: str, compressed_dataset_path: str, n_clusters: int, output_path: str = "evaluation_outputs/") -> float:
    return evaluate_semantic_coverage_CLIP(original_dataset_path, compressed_dataset_path, n_clusters, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--compressed_dataset_path", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, required=False, default=90)
    parser.add_argument("--output_path", type=str, required=False, default="evaluation_outputs/")
    return parser.parse_args()

def main():
    args = parse_args()
    run_semantic_compression_ratio(args.original_dataset_path, args.compressed_dataset_path, args.n_clusters, args.output_path)

if __name__ == "__main__":
    main()