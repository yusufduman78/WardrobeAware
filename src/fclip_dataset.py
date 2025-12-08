"""
Dataset Loader for Fashion Compatibility Training
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from src import fclip_config as config
from utils import fclip_utils as utils

class PolyvoreEmbeddingDataset(Dataset):
    """
    Dataset that serves FashionCLIP embeddings for training.
    Used in conjunction with a specialized Sampler or Miner.
    """
    def __init__(self, 
                 embeddings_path: Path = config.IMAGE_EMBEDDINGS_PATH, 
                 item_ids_path: Path = config.ITEM_IDS_PATH,
                 metadata: Optional[Dict] = None):
        """
        Args:
            embeddings_path: Path to .npy file containing embeddings
            item_ids_path: Path to .json file containing item IDs corresponding to embeddings
            metadata: Optional metadata dictionary
        """
        print(f"Loading embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path)
        self.item_ids = utils.load_json(item_ids_path)
        self.metadata = metadata or {}
        
        assert len(self.embeddings) == len(self.item_ids), \
            f"Mismatch: {len(self.embeddings)} embeddings vs {len(self.item_ids)} IDs"
            
        print(f"Loaded {len(self.embeddings)} embeddings.")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": torch.from_numpy(self.embeddings[idx]).float(),
            "item_id": self.item_ids[idx],
            "index": idx
        }

class TripletDataset(Dataset):
    """
    Dataset that yields triplets (anchor, positive, negative).
    """
    def __init__(self, triplets: List[Tuple[int, int, int]], embeddings: np.ndarray):
        self.triplets = triplets
        self.embeddings = embeddings

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_idx, p_idx, n_idx = self.triplets[idx]
        return (
            torch.from_numpy(self.embeddings[a_idx]).float(), # Anchor
            torch.from_numpy(self.embeddings[p_idx]).float(), # Positive
            torch.from_numpy(self.embeddings[n_idx]).float()  # Negative
        )
