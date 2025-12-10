from dataset_stream.dataset_stream import DatasetStream
from dataset_stream.dataset_writer import DatasetWriter
import numpy as np
from tqdm import tqdm
import cv2
from inference.CLIP import _load_clip_model, get_clip_embedding
from inference.DINO import _load_dino_model, get_dino_embedding
import os
import hashlib

def pose_and_semantic_compression(original_dataset_path: str, target_path: str, xy_pose_resolution: float = 1.0, yaw_pose_resolution: float = 45.0, similarity_threshold: float = 0.95, model_type: str = "CLIP"):
    if model_type not in ["CLIP", "DINO"]:
        raise ValueError(f"Invalid model type: {model_type}")
    """
    Compress a dataset by enforcing geometric coverage, then gating by semantic similarity.
    Parameters
    ----------
    original_dataset_path: str
    target_path: str
    similarity_threshold: float
    model_type: str
    """

    original_dataset_stream = DatasetStream(original_dataset_path)
    compressed_dataset_writer = DatasetWriter(target_path)
    if model_type == "CLIP":
        model, preprocess = _load_clip_model()
    elif model_type == "DINO":
        model, preprocess = _load_dino_model()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    poses = np.array(original_dataset_stream.poses)

    min_x = min(poses[:, 0])
    min_y = min(poses[:, 1])
    max_x = max(poses[:, 0])
    max_y = max(poses[:, 1])

    total_bins_x = int((max_x - min_x) / xy_pose_resolution)
    total_bins_y = int((max_y - min_y) / xy_pose_resolution)
    total_bins_yaw = int(360 / yaw_pose_resolution)

    x_edges = np.linspace(min_x, max_x, total_bins_x + 1)
    y_edges = np.linspace(min_y, max_y, total_bins_y + 1)
    yaw_edges = np.linspace(-180, 180, total_bins_yaw + 1)

    # Check if embeddings are cached for the original dataset
    cache_dir = f".cache/embeddings/original_{model_type}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5(original_dataset_path.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
    
    # Load or compute all embeddings
    if os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        embeddings = np.load(cache_path)
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
        embeddings = np.zeros((len(original_dataset_stream), embedding_dim))
        embeddings[0] = first_embedding
        for i, snapshot in tqdm(enumerate(original_dataset_stream.iterate()), total=len(original_dataset_stream), desc="Computing embeddings"):
            if i == 0:
                continue  # Already processed first embedding
            image = cv2.imread(snapshot.frame_path)
            if model_type == "CLIP":
                embedding = get_clip_embedding(image, model, preprocess)
            elif model_type == "DINO":
                embedding = get_dino_embedding(image, model, preprocess)
            embeddings[i] = embedding
        np.save(cache_path, embeddings)
        print(f"Cached embeddings to {cache_path}")

    compressed_histogram = np.zeros((total_bins_x, total_bins_y, total_bins_yaw))
    histo_bin_to_embeddings = {}

    repeated_bins_save_dir = os.path.join(target_path, "repeated_bin_pictures")
    os.makedirs(repeated_bins_save_dir, exist_ok=True)

    for i, snapshot in tqdm(enumerate(original_dataset_stream.iterate()), total=len(original_dataset_stream), desc="Compressing (pose and semantic)"):
        print(f"Processing snapshot {i}")
        instance_pose = np.array(snapshot.pose)
        embedding = embeddings[i]
        x_bin = np.digitize(instance_pose[0], x_edges) - 1
        y_bin = np.digitize(instance_pose[1], y_edges) - 1
        yaw_bin = np.digitize(instance_pose[2], yaw_edges) - 1

        x_bin = np.clip(x_bin, 0, total_bins_x - 1)
        y_bin = np.clip(y_bin, 0, total_bins_y - 1)
        yaw_bin = np.clip(yaw_bin, 0, total_bins_yaw - 1)

        if compressed_histogram[x_bin, y_bin, yaw_bin] == 0:
            compressed_histogram[x_bin, y_bin, yaw_bin] += 1
            histo_bin_to_embeddings[(x_bin, y_bin, yaw_bin)] = {"embeddings": [embedding], "frame_paths": [snapshot.frame_path]}
            compressed_dataset_writer.write_instance(snapshot)
        else:
            # get the embeddings of the snapshot
            current_bin_embeddings = histo_bin_to_embeddings[(x_bin, y_bin, yaw_bin)]["embeddings"]
            similarities = np.dot(current_bin_embeddings, embedding)
            max_similarity = np.max(similarities)
            if max_similarity < similarity_threshold:
                compressed_dataset_writer.write_instance(snapshot)
                histo_bin_to_embeddings[(x_bin, y_bin, yaw_bin)]["embeddings"].append(embedding)
                histo_bin_to_embeddings[(x_bin, y_bin, yaw_bin)]["frame_paths"].append(snapshot.frame_path)
                print(f"Added embedding to bin {x_bin, y_bin, yaw_bin} for a total of {len(histo_bin_to_embeddings[(x_bin, y_bin, yaw_bin)])} embeddings")
                # save the picture of the snapshot
            else:
                print(f"Skipping snapshot {snapshot.frame_path} because similarity {max_similarity} is greater than threshold {similarity_threshold}")
                continue

    # Find all bins with more than 1 item stored, and visualize
    bins_with_multiple_items = []
    for bin_key, info in histo_bin_to_embeddings.items():
        num_items = len(info["embeddings"])
        if num_items > 1:
            vis_dir = os.path.join(repeated_bins_save_dir, f"bin_{bin_key[0]}_{bin_key[1]}_{bin_key[2]}")
            os.makedirs(vis_dir, exist_ok=True)
            for i, frame_path in enumerate(info["frame_paths"]):
                cv2.imwrite(os.path.join(vis_dir, os.path.basename(frame_path)), cv2.imread(frame_path))

    return compressed_dataset_writer