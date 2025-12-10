from dataset_stream.dataset_stream import DatasetStream
from dataset_stream.dataset_writer import DatasetWriter
import numpy as np
import cv2
from inference.CLIP import _load_clip_model, get_clip_embedding
from tqdm import tqdm
import hashlib
import os
from inference.DINO import _load_dino_model, get_dino_embedding

def semantic_only_compression(original_dataset_path: str, target_path: str, similarity_threshold: float = 0.95, model_type: str = "CLIP"):
    if model_type not in ["CLIP", "DINO"]:
        raise ValueError(f"Invalid model type: {model_type}")
    """
    Compress a dataset by only keeping the semantic information.
    Each new item is compared against all current saved items. If similarity is greater than
    the threshold, the item is thrown out. Otherwise, it is added to the new dataset.
    
    Parameters
    ----------
    original_dataset_path : str
        Path to the original dataset
    target_path : str
        Path where the compressed dataset will be saved
    similarity_threshold : float
        Similarity threshold (0-1). Items with similarity > threshold to any saved item are discarded.
        Default: 0.95
    model_type : str
        Type of model to use for embedding extraction. Must be "CLIP" or "DINO".
        Default: "CLIP"
    """
    
    original_dataset_stream = DatasetStream(original_dataset_path)
    compressed_dataset_writer = DatasetWriter(target_path)
    if model_type == "CLIP":
        model, preprocess = _load_clip_model()
    elif model_type == "DINO":
        model, preprocess = _load_dino_model()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    saved_embeddings = []

    # Check if embeddings are cached for the original dataset
    cache_dir = f".cache/embeddings/original_{model_type}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5(original_dataset_path.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
    
    # Load or compute all embeddings
    if os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        all_embeddings = np.load(cache_path)
    else:
        print(f"Computing embeddings (will cache to {cache_path})")
        # Get first embedding to infer the embedding dimension
        first_snapshot = next(original_dataset_stream.iterate())
        first_image = cv2.imread(first_snapshot.frame_path)
        if model_type == "CLIP":
            first_embedding = get_clip_embedding(first_image, model, preprocess)
        elif model_type == "DINO":
            first_embedding = get_dino_embedding(first_image, model, preprocess)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        embedding_dim = len(first_embedding)
        all_embeddings = np.zeros((len(original_dataset_stream), embedding_dim))
        all_embeddings[0] = first_embedding
        for i, snapshot in tqdm(enumerate(original_dataset_stream.iterate()), total=len(original_dataset_stream), desc="Computing embeddings"):
            if i == 0:
                continue  # Already processed first embedding
            image = cv2.imread(snapshot.frame_path)
            if model_type == "CLIP":
                embedding = get_clip_embedding(image, model, preprocess)
            elif model_type == "DINO":
                embedding = get_dino_embedding(image, model, preprocess)
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            all_embeddings[i] = embedding
        np.save(cache_path, all_embeddings)
        print(f"Cached embeddings to {cache_path}")

    # Now iterate through the dataset using cached embeddings
    for i, snapshot in tqdm(enumerate(original_dataset_stream.iterate()), total=len(original_dataset_stream), desc="Compressing (semantic only)"):
        embedding = all_embeddings[i]
        
        # If no saved embeddings yet, save the first one
        if len(saved_embeddings) == 0:
            compressed_dataset_writer.write_instance(snapshot)
            saved_embeddings.append(embedding)
            continue

        saved_embeddings_array = np.array(saved_embeddings)
        similarities = np.dot(saved_embeddings_array, embedding)
        max_similarity = np.max(similarities)
        
        if max_similarity > similarity_threshold:
            print(f"Skipping item {snapshot.frame_path} because similarity {max_similarity} is greater than threshold {similarity_threshold}")
            continue
        
        compressed_dataset_writer.write_instance(snapshot)
        saved_embeddings.append(embedding)
