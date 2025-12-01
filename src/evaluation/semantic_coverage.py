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
from open_clip import create_model_and_transforms
import hashlib
import os
import json
import matplotlib.pyplot as plt


def evaluate_semantic_coverage(original_dataset_path: str, compressed_dataset_path: str, n_clusters: int = 10) -> float:
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
    Returns
    -------
    semantic_coverage: float
        The ratio of the number of bins covered by the original dataset to the number of bins covered by the compressed dataset.
    """

    original_base_path = os.path.basename(original_dataset_path)
    compressed_base_path = os.path.basename(compressed_dataset_path)

    output_dir = f"evaluation_outputs/{original_base_path}_vs_{compressed_base_path}/semantic_coverage"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_dataset_stream = DatasetStream(original_dataset_path)
    compressed_dataset_stream = DatasetStream(compressed_dataset_path)

    model, preprocess = _load_clip_model()
    # cache_dir = os.path.join(original_dataset_path, ".cache/embeddings")
    original_embeddings = get_all_embeddings_from_dataset(original_dataset_stream, 
                                                          original_dataset_path, 
                                                          model, 
                                                          preprocess, 
                                                          cache_dir=".cache/embeddings/original")

    print(f"Original embeddings shape: {original_embeddings.shape}")

    # now we need to run k-means on the embeddings to get the clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++", n_init=10).fit(original_embeddings)
    C = cosine_similarity(kmeans.cluster_centers_)
    np.fill_diagonal(C, -1)

    # cache_dir = os.path.join(compressed_dataset_path, ".cache/embeddings")
    compressed_embeddings = get_all_embeddings_from_dataset(compressed_dataset_stream, 
                                                            compressed_dataset_path, 
                                                            model, 
                                                            preprocess, 
                                                            cache_dir=".cache/embeddings/compressed")
    compressed_clusters = kmeans.predict(compressed_embeddings)

    coverage_ratio = unweighted_coverage(kmeans.labels_, compressed_clusters, n_clusters)
    rare_cluster_recall_ratio = rare_cluster_recall(kmeans.labels_, compressed_clusters, n_clusters)
    distribution_L1_ratio, histogram_original, histogram_compressed = distribution_L1(kmeans.labels_, compressed_clusters, n_clusters)

    print(f"Coverage ratio: {coverage_ratio}")
    print(f"Rare cluster recall ratio: {rare_cluster_recall_ratio}")
    print(f"Distribution L1 ratio: {distribution_L1_ratio}")

    metrics = {
        "original_dataset_path": original_dataset_path,
        "compressed_dataset_path": compressed_dataset_path,
        "n_clusters": n_clusters,
        "cluster_coverage_ratio": coverage_ratio,
        "rare_cluster_recall_ratio": rare_cluster_recall_ratio,
        "distribution_L1_ratio": distribution_L1_ratio,
    }

    log_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    plot_cluster_histograms(histogram_original, histogram_compressed, output_dir)
    return metrics

def log_metrics(metrics: dict, log_file: str = "metrics.json"):
    """
    Log metrics to a file.
    """
    with open(log_file, "w") as f:
        json.dump(metrics, f)

def plot_cluster_histograms(histogram_original: np.ndarray, histogram_compressed: np.ndarray, output_dir: str):
    """
    Plot the histograms of the original and compressed clusters.
    """
    plt.bar(range(len(histogram_original)), histogram_original, alpha=0.5, color="blue")
    plt.title("Histogram of Original Clusters")
    plt.xlabel("Cluster Index")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "histogram_original_plot.png"))
    plt.close()
    plt.bar(range(len(histogram_compressed)), histogram_compressed, alpha=0.5, color="red")
    plt.title("Histogram of Compressed Clusters")
    plt.xlabel("Cluster Index")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "histogram_compressed_plot.png"))
    plt.close()


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

def get_clip_embedding(image: np.ndarray, model: clip.model.CLIP, preprocess: Any) -> np.ndarray:
    """
    Get CLIP embedding for an image.
    
    Parameters
    ----------
    image : np.ndarray
        Image in BGR format (as returned by cv2.imread)
    
    Returns
    -------
    embedding : np.ndarray
        CLIP embedding vector (1024-dimensional for ViT-g-14)
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Preprocess and get embedding
    image_tensor = preprocess(pil_image).unsqueeze(0).to(_get_device())
    
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
    
    return embedding.cpu().numpy().flatten()

def _load_clip_model():
    """Load and cache CLIP model."""
    device = _get_device()
    model, preprocess_train, preprocess_eval = create_model_and_transforms(
        "ViT-g-14",
        pretrained="laion2b_s34b_b88k"
    )
    model.to(device)
    model.eval()
    return model, preprocess_eval


def _get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

