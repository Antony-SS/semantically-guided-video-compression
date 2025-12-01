from evaluation.compression_ratio import framewise_compression_ratio

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--compressed_dataset_path", type=str, required=True)
    args = parser.parse_args()
    framewise_compression_ratio(args.original_dataset_path, args.compressed_dataset_path)