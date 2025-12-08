"""
Utility functions for the Fashion Compatibility System
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from src import fclip_config as config


def get_device() -> str:
    """Get the appropriate device (CUDA or CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_json(file_path: Path) -> dict:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, file_path: Path) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_metadata() -> Dict:
    """Load item metadata"""
    return load_json(config.METADATA_PATH)


def get_item_text(item_meta: Dict) -> str:
    """Extract text description from item metadata"""
    # Priority: title -> description -> url_name
    for key in ['title', 'description', 'url_name']:
        value = item_meta.get(key, '').strip()
        if value:
            return value
    return 'No description available'


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length for cosine similarity"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return embeddings / norms


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    vec1_norm = normalize_embeddings(vec1.reshape(1, -1))
    vec2_norm = normalize_embeddings(vec2.reshape(1, -1))
    return float(np.dot(vec1_norm, vec2_norm.T)[0, 0])


def load_image(image_path: Path) -> Optional[Image.Image]:
    """Load image from path, return None if not found"""
    try:
        img = Image.open(image_path)
        img.load()  # Load image data into memory
        return img
    except (FileNotFoundError, OSError):
        return None

