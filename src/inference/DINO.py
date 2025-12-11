
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Any
# dinov2 is typically used via torch.hub, not as an installed package
import torch.hub

DEVICE_NUMBER = 0

def _get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu"

def _load_dino_model():
    """
    Load and cache DINOv2 model and preprocessing
    
    Returns
    -------
    model : torch.nn.Module
        The DINO model (ViT-L/14)
    preprocess : Callable
        Preprocessing function to use on images
    """
    device = _get_device()
    # Load DINOv2 model via torch.hub
    print("Loading DINOv2 model (ViT-L/14)...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    model.eval()
    model.to(device)
    
    # DINOv2 uses standard ImageNet normalization
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return model, preprocess

def get_dino_embedding(image: np.ndarray, model: Any, preprocess: Any) -> np.ndarray:
    """
    Get DINOv2 embedding for an image.
    
    Parameters
    ----------
    image : np.ndarray
        Image in BGR format (as returned by cv2.imread)
    model
        The DINOv2 model to use for inference
    preprocess : Any
        The preprocessing function
    
    Returns
    -------
    embedding : np.ndarray
        DINOv2 embedding vector
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Preprocess and get embedding
    image_tensor = preprocess(pil_image).unsqueeze(0).to(_get_device())
    
    with torch.no_grad():
        embedding = model(image_tensor)
        if isinstance(embedding, (tuple, list)):
            embedding = embedding[0]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        
    return embedding.cpu().numpy().flatten()
