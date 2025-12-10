import numpy as np
import cv2
import torch
from PIL import Image
from typing import Any
from open_clip import create_model_and_transforms

DEVICE_NUMBER = 0


def _get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu"


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


def get_clip_embedding(image: np.ndarray, model, preprocess: Any) -> np.ndarray:
    """
    Get CLIP embedding for an image.
    
    Parameters
    ----------
    image : np.ndarray
        Image in BGR format (as returned by cv2.imread)
    model
        The CLIP model to use for inference
    preprocess : Any
        The preprocessing function
    
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

