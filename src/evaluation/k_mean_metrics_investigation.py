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
from sklearn.metrics import silhouette_score


def evaluate_semantic_coverage_CLIP(original_dataset_path: str, compressed_dataset_path: str, n_clusters: int = 10) -> float:
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

    compressed_embeddings = get_all_embeddings_from_dataset(compressed_dataset_stream, 
                                                            compressed_dataset_path, 
                                                            model, 
                                                            preprocess, 
                                                            cache_dir=".cache/embeddings/compressed")

    print(f"Original embeddings shape: {original_embeddings.shape}")

    # now we need to run k-means on the embeddings to get the clusters
    cosine_similarities_means = []
    silhouette_scores = []
    min_cluster_sizes = []

    coverage_ratios = []
    rare_cluster_recall_ratios = []
    distribution_L1_ratios = []
    original_histograms = []
    compressed_histograms = []
    
    range_to_use = range(20, 121, 10)

    for i in range_to_use:
        kmeans = KMeans(n_clusters=i, random_state=0, init="k-means++", n_init=10).fit(original_embeddings)
        C = cosine_similarity(kmeans.cluster_centers_)
        # Plot cosine similarity matrix as a heatmap, similar to a confusion matrix
        np.fill_diagonal(C, 1.0)
        plt.figure(figsize=(8, 7))
        im = plt.imshow(C, cmap='viridis', aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f'Cosine Similarity Matrix (k={i})')
        plt.xlabel('Cluster Center Index')
        plt.ylabel('Cluster Center Index')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cosine_similarity_matrix_k_{i}.png"))
        plt.close()
        cosine_similarities_mean = np.mean(C[np.triu_indices_from(C, k=1)])
        print(f"FOR K={i}:")
        print(f"Cosine similarity mean: {cosine_similarities_mean}")
        print(f"Silhouette score: {silhouette_score(original_embeddings, kmeans.labels_, metric='cosine')}")
        min_cluster_size = np.min(np.bincount(kmeans.labels_))
        print(f"Min cluster size: {min_cluster_size / len(original_embeddings)}")
        cosine_similarities_means.append(cosine_similarities_mean)
        silhouette_scores.append(silhouette_score(original_embeddings, kmeans.labels_, metric='cosine'))
        min_cluster_sizes.append(min_cluster_size / len(original_embeddings))

        # Now run predictions on the compressed dataset
        compressed_clusters = kmeans.predict(compressed_embeddings)
        coverage_ratios.append(unweighted_coverage(kmeans.labels_, compressed_clusters, i))
        rare_cluster_recall_ratios.append(rare_cluster_recall(kmeans.labels_, compressed_clusters, i))
        distribution_L1_ratio, histogram_original, histogram_compressed = distribution_L1(kmeans.labels_, compressed_clusters, i)
        distribution_L1_ratios.append(distribution_L1_ratio)
        original_histograms.append(histogram_original)
        compressed_histograms.append(histogram_compressed)
        print(f"Coverage ratio: {coverage_ratios[-1]}")
        print(f"Rare cluster recall ratio: {rare_cluster_recall_ratios[-1]}")
        print(f"Distribution L1 ratio: {distribution_L1_ratios[-1]}")

    cosine_similarities_means = np.array(cosine_similarities_means)
    silhouette_scores = np.array(silhouette_scores)
    min_cluster_sizes = np.array(min_cluster_sizes)
    coverage_ratios = np.array(coverage_ratios)
    rare_cluster_recall_ratios = np.array(rare_cluster_recall_ratios)
    distribution_L1_ratios = np.array(distribution_L1_ratios)

    plot_k_means_metrics(range_to_use, cosine_similarities_means, silhouette_scores, min_cluster_sizes, output_dir)
    plot_compression_metrics_across_clusters(range_to_use, coverage_ratios, rare_cluster_recall_ratios, distribution_L1_ratios, original_histograms, compressed_histograms, output_dir)

    clusters_to_stats_dict = {}
    for k in range(len(range_to_use)):
        clusters_to_stats_dict[range_to_use[k]] = {
            "coverage_ratio": coverage_ratios[k],
            "rare_cluster_recall_ratio": rare_cluster_recall_ratios[k],
            "distribution_L1_ratio": distribution_L1_ratios[k],
        }
    
    metrics = {
        "original_dataset_path": original_dataset_path,
        "compressed_dataset_path": compressed_dataset_path,
        "n_clusters": n_clusters,
        "cluster_range": list(range_to_use),
        "clusters_to_stats_dict": clusters_to_stats_dict,
    }

    log_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    return metrics

def plot_compression_metrics_across_clusters(range_to_use: range, 
                                             coverage_ratios: np.ndarray, 
                                             rare_cluster_recall_ratios: np.ndarray, 
                                             distribution_L1_ratios: np.ndarray, 
                                             original_histograms: list[np.ndarray], 
                                             compressed_histograms: list[np.ndarray], 
                                             output_dir: str):
    """
    Plot the coverage ratio across number of clusters.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range_to_use, coverage_ratios, marker='o', linestyle='-', color='b', label="Coverage ratio")
    plt.plot(range_to_use, rare_cluster_recall_ratios, marker='o', linestyle='-', color='g', label="Rare cluster recall ratio")
    plt.plot(range_to_use, distribution_L1_ratios, marker='o', linestyle='-', color='r', label="Distribution L1 ratio")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Value")
    plt.title("Compression Metrics Across Number of Clusters")
    plt.grid(True)
    plt.xticks(range_to_use)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compression_metrics_across_number_of_clusters.png"))
    plt.close()

    for i in range(len(range_to_use)):
        plot_cluster_histograms(range_to_use[i], original_histograms[i], compressed_histograms[i], output_dir)

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

def plot_k_means_metrics(cluster_ticks: np.ndarray, cosine_similarities_means: np.ndarray, silhouette_scores: np.ndarray, min_cluster_sizes: np.ndarray, output_dir: str):
    """
    Plot the k-means metrics on the uncompressed dataset.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(cluster_ticks, cosine_similarities_means, marker='o', linestyle='-', color='r', label="cosine similarity mean")
    plt.plot(cluster_ticks, silhouette_scores, marker='o', linestyle='-', color='b', label="silhouette score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Value")
    plt.title("Mean Similarity and Silhouette Score Across Number of Clusters")
    plt.grid(True)
    plt.xticks(cluster_ticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_means_metrics_across_number_of_clusters.png"))
    plt.close()
    plt.figure(figsize=(12, 6))
    plt.plot(cluster_ticks, min_cluster_sizes, marker='o', linestyle='-', color='g', label="Min cluster size (%)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Min Cluster Size (fraction of data)")
    plt.title("Minimum Cluster Size (Normalized) Across Number of Clusters")
    plt.grid(True)
    plt.xticks(cluster_ticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "min_cluster_size_normalized_across_number_of_clusters.png"))
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
    plt.title(f"Histogram of Original Clusters (k={k})")
    plt.xlabel("Cluster Index")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"histogram_original_plot_k_{k}.png"))
    plt.close()
    plt.bar(range(len(histogram_compressed)), histogram_compressed, alpha=0.5, color="red")
    plt.title(f"Histogram of Compressed Clusters (k={k})")
    plt.xlabel("Cluster Index")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"histogram_compressed_plot_k_{k}.png"))
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

