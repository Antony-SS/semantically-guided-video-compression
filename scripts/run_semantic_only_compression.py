from compression.semantic_only_compression import semantic_only_compression

import argparse

def run_semantic_only_compression(original_dataset_path: str, target_path: str, similarity_threshold: float = 0.95, model_type: str = "CLIP"):
    return semantic_only_compression(original_dataset_path, target_path, similarity_threshold, model_type)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--similarity_threshold", type=float, required=False, default=0.90)
    parser.add_argument("--model_type", type=str, required=False, default="DINO")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_semantic_only_compression(args.original_dataset_path, args.target_path, args.similarity_threshold, args.model_type)