from compression.semantic_only_compression import semantic_only_compression

import argparse

def run_semantic_only_compression(dataset_path: str, output_path: str, similarity_threshold: float = 0.95, model_type: str = "DINO"):
    return semantic_only_compression(dataset_path, output_path, similarity_threshold, model_type)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--similarity_threshold", type=float, required=False, default=0.90)
    parser.add_argument("--model_type", type=str, required=False, default="DINO")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_semantic_only_compression(args.dataset_path, args.output_path, args.similarity_threshold, args.model_type)