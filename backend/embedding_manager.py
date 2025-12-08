import numpy as np
import json
import torch
from pathlib import Path
from typing import Dict, Optional
from src import fclip_config as config

class EmbeddingManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.embeddings = None
        self.id_to_idx = {}
        self.device = "cpu" # Keep embeddings on CPU to save GPU memory for model
        self.load_data()
        self.initialized = True

    def load_data(self):
        try:
            print("Loading embeddings and IDs...")
            if not config.IMAGE_EMBEDDINGS_PATH.exists() or not config.ITEM_IDS_PATH.exists():
                print("WARNING: Embeddings or Item IDs not found. Recommendations will be random.")
                return

            # Load Embeddings (mmap_mode='r' allows accessing parts of array without loading all to RAM)
            # But for 140MB, loading to RAM is faster.
            self.embeddings = np.load(config.IMAGE_EMBEDDINGS_PATH)
            
            # Load IDs
            with open(config.ITEM_IDS_PATH, 'r') as f:
                item_ids = json.load(f)
                
            # Create Mapping
            self.id_to_idx = {id_: i for i, id_ in enumerate(item_ids)}
            
            print(f"Loaded {len(item_ids)} embeddings.")
            
        except Exception as e:
            print(f"Error loading embedding data: {e}")

    def get_embedding(self, item_id: str) -> Optional[torch.Tensor]:
        if self.embeddings is None or item_id not in self.id_to_idx:
            return None
            
        idx = self.id_to_idx[item_id]
        emb = self.embeddings[idx]
        return torch.tensor(emb, dtype=torch.float32)

    def get_all_embeddings(self):
        return self.embeddings

# Global Instance
embedding_manager = EmbeddingManager()
