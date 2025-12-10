from dataset_stream.dataset_stream import DatasetStream
import numpy as np
import cv2
import torch
import clip
from PIL import Image
from typing import Any
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import random
from inference.CLIP import _load_clip_model, get_clip_embedding


def evaluate_semantic_coverage_CLIP(original_dataset_path: str, compressed_dataset_path: str, n_clusters: int = 80, output_path: str = "evaluation_outputs/") -> float:
    """
    Evaluate the semantic coverage of the compressed dataset.
    Semantic coverage is defined as the ratio of the number of bins covered by the original dataset to the number of bins covered by the compressed dataset.
    A bin is considered covered if there exists at least one pose in that bin.
    Parameters
    ----------
    original_dataset_path: str
    compressed_dataset_path: str
    n_clusters: int
        The number of clusters to use for k-means
    model_type: str
        Type of model to use for embedding extraction. Must be "CLIP" or "DINO".
        Default: "CLIP"
    Returns
    -------
    metrics: dict
        The ratio of the number of bins covered by the original dataset to the number of bins covered by the compressed dataset.
    """

    original_base_path = os.path.basename(original_dataset_path)
    compressed_base_path = os.path.basename(compressed_dataset_path)

    output_dir = os.path.join(output_path, f"{original_base_path}_vs_{compressed_base_path}", "semantic_coverage")

    model, preprocess = _load_clip_model()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_dataset_stream = DatasetStream(original_dataset_path)
    compressed_dataset_stream = DatasetStream(compressed_dataset_path)

    # cache_dir = os.path.join(original_dataset_path, ".cache/embeddings")
    original_embeddings = get_all_embeddings_from_dataset(original_dataset_stream, 
                                                          original_dataset_path, 
                                                          model, 
                                                          preprocess, 
                                                          cache_dir=f".cache/embeddings/original_CLIP")

    compressed_embeddings = get_all_embeddings_from_dataset(compressed_dataset_stream, 
                                                            compressed_dataset_path, 
                                                            model, 
                                                            preprocess, 
                                                            cache_dir=f".cache/embeddings/compressed_CLIP")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++", n_init=10).fit(original_embeddings)
    
    C = cosine_similarity(kmeans.cluster_centers_)

    plot_cosine_similarity_matrix(C, n_clusters, output_dir) # plot the cosine similarity matrix

    # Now run predictions on the compressed dataset
    compressed_clusters = kmeans.predict(compressed_embeddings)
    coverage_ratio = unweighted_coverage(kmeans.labels_, compressed_clusters, n_clusters)
    rare_cluster_recall_ratio = rare_cluster_recall(kmeans.labels_, compressed_clusters, n_clusters)
    distribution_L1_ratio, histogram_original, histogram_compressed = distribution_L1(kmeans.labels_, compressed_clusters, n_clusters)

    # plot images of clusters
    dropped_clusters = np.where((histogram_original > 0) & (histogram_compressed == 0))[0]
    visualize_cluster_pictures(kmeans.labels_, original_dataset_stream, dropped_clusters, os.path.join(output_dir, "cluster_images"))

    # write dropped clusters to a file
    output_dropped_clusters(histogram_original, histogram_compressed, os.path.join(output_dir, "dropped_clusters.txt"))

    # Plot histograms
    plot_cluster_histograms(n_clusters, histogram_original, histogram_compressed, output_dir)

    # Log metrics
    metrics = {
        "original_dataset_path": original_dataset_path,
        "compressed_dataset_path": compressed_dataset_path,
        "n_clusters": n_clusters,
        "silhouette_score_original_dataset": silhouette_score(original_embeddings, kmeans.labels_, metric='cosine'),
        "min_cluster_size_original_dataset": np.min(np.bincount(kmeans.labels_)) / len(original_embeddings),
        "semantic_coverage_ratio": coverage_ratio,
        "rare_cluster_recall_ratio": rare_cluster_recall_ratio,
        "distribution_L1_ratio": distribution_L1_ratio,
    }

    log_metrics(metrics, os.path.join(output_dir, "metrics.json"))

    return metrics

def plot_cosine_similarity_matrix(cosine_similarities_matrix: np.ndarray, k: int, output_dir: str):
    """
    Plot the cosine similarity matrix.
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(cosine_similarities_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"Cosine Similarity Matrix (k={k})")
    plt.savefig(os.path.join(output_dir, f"cosine_similarity_matrix_k_{k}.png"))
    plt.close()

def plot_cosine_similarity_matrix(cosine_similarities_matrix: np.ndarray, k: int, output_dir: str):
    """
    Plot the cosine similarity matrix.
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(cosine_similarities_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"Cosine Similarity Matrix (k={k})")
    plt.savefig(os.path.join(output_dir, f"cosine_similarity_matrix_k_{k}.png"))
    plt.close()

def log_metrics(metrics: dict, log_file: str = "metrics.json"):
    """
    Log metrics to a file.
    """
    with open(log_file, "w") as f:
        json.dump(metrics, f)

def plot_cluster_histograms(k: int, histogram_original: np.ndarray, histogram_compressed: np.ndarray, output_dir: str):
    """
    Plot the histograms of the original and compressed clusters.
    """
    plt.bar(range(len(histogram_original)), histogram_original, alpha=0.5, color="blue")
    plt.title(f"Histogram of Original Dataset Visual Modes (k={k})")
    plt.xlabel("Cluster Index")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"histogram_original_plot_k_{k}.png"))
    plt.close()
    plt.bar(range(len(histogram_compressed)), histogram_compressed, alpha=0.5, color="red")
    plt.title(f"Histogram of Compressed Dataset Visual Modes (k={k})")
    plt.xlabel("Cluster Index")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"histogram_compressed_plot_k_{k}.png"))
    plt.close()

def output_dropped_clusters(histogram_original: np.ndarray, histogram_compressed: np.ndarray, output_file_name: str):
    """
    Output the dropped clusters to a file.
    """
    dropped_clusters = np.where((histogram_original > 0) & (histogram_compressed == 0))[0]
    with open(output_file_name, "w") as f:
        f.write(f"Dropped clusters: {len(dropped_clusters)}\n")
        for cluster_idx_dropped in dropped_clusters:
            f.write(f"{cluster_idx_dropped}: original cluster size: {histogram_original[cluster_idx_dropped]}\n")

def visualize_cluster_pictures(k_means_labels: np.ndarray, dataset_stream: DatasetStream, dropped_clusters: np.ndarray, output_dir: str):
    """
    Visualize the pictures of the clusters.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Visualizing cluster pictures for {output_dir}")
    
    for cluster_idx in range(k_means_labels.max() + 1):

        # only visualize the dropped clusters
        if cluster_idx not in dropped_clusters:
            continue

        indices_in_cluster = np.where(k_means_labels == cluster_idx)[0]
        if len(indices_in_cluster) == 0:
            continue
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}")
        os.makedirs(cluster_dir, exist_ok=True)
        for n, idx in enumerate(indices_in_cluster):
            picture = cv2.imread(dataset_stream.get_instance(idx).frame_path)
            cv2.imwrite(os.path.join(cluster_dir, f"picture_{idx}.png"), picture)

    print(f"Visualized cluster pictures for {output_dir}")

def compute_cluster_hist(clusters, k):
    """
    clusters: 1D array of ints in [0, k-1]
    returns: probs p[c] over clusters (sum p = 1), and counts
    """
    counts = np.bincount(clusters, minlength=k).astype(np.float64)
    total = counts.sum()
    if total == 0:
        # avoid divide-by-zero; return zeros
        return np.zeros(k, dtype=np.float64), counts
    return counts / total, counts


def unweighted_coverage(clusters_full, clusters_comp, k):
    """
    Fraction of distinct clusters (visual modes) from full
    that are hit by at least one compressed frame.
    """
    present_full = set(np.unique(clusters_full))
    present_comp = set(np.unique(clusters_comp))
    if len(present_full) == 0:
        return 0.0
    return len(present_full & present_comp) / len(present_full)


def rare_cluster_recall(clusters_full, clusters_comp, k, rare_fraction=0.25):
    """
    Recall on 'rare' clusters: bottom rare_fraction (by size) of clusters in full.
    E.g. rare_fraction=0.25 -> bottom 25% of clusters (by count).
    """
    _, counts_full = compute_cluster_hist(clusters_full, k)

    # sort clusters by frequency ascending
    cluster_ids = np.arange(k)
    sort_idx = np.argsort(counts_full)
    sorted_clusters = cluster_ids[sort_idx]

    # select bottom rare_fraction of clusters that actually appear
    nonzero_sorted = [c for c in sorted_clusters if counts_full[c] > 0]
    if len(nonzero_sorted) == 0:
        return 0.0

    num_rare = max(1, int(len(nonzero_sorted) * rare_fraction))
    rare_clusters_full = set(nonzero_sorted[:num_rare])

    present_comp = set(np.unique(clusters_comp))
    rare_preserved = rare_clusters_full & present_comp

    return len(rare_preserved) / len(rare_clusters_full)


def distribution_L1(clusters_full, clusters_comp, k):
    """
    L1 distance between cluster distributions in full vs compressed.
    0 = identical distributions, 1 = completely disjoint.
    """
    p_full, _ = compute_cluster_hist(clusters_full, k)
    p_comp, _ = compute_cluster_hist(clusters_comp, k)

    # 0.5 * L1 norm of difference
    return 0.5 * np.abs(p_full - p_comp).sum(), p_full, p_comp

def get_all_embeddings_from_dataset(dataset_stream: DatasetStream, dataset_path: str, model: clip.model.CLIP, preprocess: Any, cache_dir: str = ".cache/embeddings") -> np.ndarray:
    """
    Get all embeddings from a dataset, using cache if available.
    
    Parameters
    ----------
    dataset_stream : DatasetStream
        The dataset stream to get embeddings from
    dataset_path : str
        The path to the dataset
    model : clip.model.CLIP
        The CLIP model to use for inference
    preprocess : Any
        The preprocessing function
    cache_dir : str
        Directory to store cached embeddings (default: ".cache/embeddings")
    
    Returns
    -------
    all_embeddings : np.ndarray
        Array of embeddings of shape (n_samples, 1024)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key from dataset path
    cache_key = hashlib.md5(dataset_path.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        return np.load(cache_path)
    
    # Compute embeddings
    print(f"Computing embeddings (will cache to {cache_path})")
    all_embeddings = np.zeros((len(dataset_stream), 1024)) # embeddings are 1024d vectors
    for i, snapshot in tqdm(enumerate(dataset_stream.iterate()), total=len(dataset_stream), desc="Getting embeddings from dataset"):
        image = cv2.imread(snapshot.frame_path)
        embedding = get_clip_embedding(image, model, preprocess)
        all_embeddings[i] = embedding
    
    # Save to cache
    np.save(cache_path, all_embeddings)
    print(f"Cached embeddings to {cache_path}")
    
    return all_embeddings


