from dataset_stream.dataset_stream import DatasetStream
import os
import json


def framewise_compression_ratio(original_dataset_path: str, compressed_dataset_path: str, output_path: str = "evaluation_outputs/") -> float:
    original_dataset = DatasetStream(original_dataset_path)
    compressed_dataset = DatasetStream(compressed_dataset_path)
    len_original_frames = len(original_dataset)
    len_compressed_frames = len(compressed_dataset)
    compression_ratio = len_compressed_frames / len_original_frames
    original_base_path = os.path.basename(original_dataset_path)
    compressed_base_path = os.path.basename(compressed_dataset_path)
    output_dir = os.path.join(output_path, f"{original_base_path}_vs_{compressed_base_path}", "framewise_compression_ratio")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metrics = {
        "original_dataset_path": original_dataset_path,
        "compressed_dataset_path": compressed_dataset_path,
        "framewise_compression_ratio": compression_ratio,
        "len_original_frames": len_original_frames,
        "len_compressed_frames": len_compressed_frames,
    }
    
    log_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    return metrics

def log_metrics(metrics: dict, log_file: str = "metrics.json"):
    """
    Log metrics to a file.
    """
    with open(log_file, "w") as f:
        json.dump(metrics, f)