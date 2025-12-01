from dataset_stream.dataset_stream import DatasetStream


def framewise_compression_ratio(original_dataset_path: str, compressed_dataset_path: str) -> float:
    original_dataset = DatasetStream(original_dataset_path)
    compressed_dataset = DatasetStream(compressed_dataset_path)
    return len(compressed_dataset) / len(original_dataset) # framewise compression ratio