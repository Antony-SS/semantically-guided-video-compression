from analysis_core.extract_rgb_frames import extract_rgb_frames
from dataset_stream.dataset_stream import DatasetStream
from dataset_stream.dataset_writer import DatasetWriter
from tqdm import tqdm


def naive_compression(original_dataset_path: str, target_compressed_path: str, downsample_factor: int = 30) -> float:
    """
    Naive compression algorithm which downsamples the original dataset by the given factor.
    Parameters
    ----------
    original_dataset_path: str
    target_compressed_path: str
    downsample_factor: int
    Returns
    -------
    compressed_dataset_path: str
    """
    original_dataset_stream = DatasetStream(original_dataset_path)
    compressed_dataset_writer = DatasetWriter(target_compressed_path)
    print(f"Compressing dataset from {original_dataset_path} to {target_compressed_path} with downsample factor {downsample_factor}")
    discard_frames = []
    for i, instance in enumerate(tqdm(original_dataset_stream.iterate(), total=len(original_dataset_stream), desc="Compressing dataset")):
        if i % downsample_factor == 0:
            compressed_dataset_writer.write_instance(instance)
        else:
            discard_frames.append(instance.frame_path)

    print(f"Compressed dataset to {target_compressed_path}")
    
    return discard_frames