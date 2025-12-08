"""
Faiss Index Management for Efficient Similarity Search
"""
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional
import json

from src import fclip_config as config
from utils import fclip_utils as utils


class FaissIndexManager:
    """Manage Faiss index for efficient similarity search"""

    def __init__(self, embedding_dim: int = config.EMBEDDING_DIM, index_type: str = "L2"):
        """
        Initialize Faiss index manager

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ("L2", "IP", or "Cosine")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.item_ids: List[str] = []
        self.id_to_index: dict = {}  # Map item_id to index position

    def create_index(self, num_vectors: int, use_gpu: bool = False) -> None:
        """
        Create a new Faiss index

        Args:
            num_vectors: Expected number of vectors
            use_gpu: Whether to use GPU for index
        """
        if self.index_type == "L2":
            # L2 distance index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IP":
            # Inner product index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "Cosine":
            # Cosine similarity (normalized inner product)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"Moving index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("Index moved to GPU")

        print(f"Created {self.index_type} index with dimension {self.embedding_dim}")

    def add_vectors(self, embeddings: np.ndarray, item_ids: List[str]) -> None:
        """
        Add vectors to the index

        Args:
            embeddings: numpy array of shape (num_items, embedding_dim)
            item_ids: List of item IDs corresponding to embeddings
        """
        if self.index is None:
            self.create_index(len(embeddings))

        # Normalize for cosine similarity if needed
        if self.index_type == "Cosine":
            embeddings = utils.normalize_embeddings(embeddings)

        # Add to index
        self.index.add(embeddings.astype('float32'))

        # Update item_ids and mapping
        start_idx = len(self.item_ids)
        self.item_ids.extend(item_ids)

        for i, item_id in enumerate(item_ids):
            self.id_to_index[item_id] = start_idx + i

        print(f"Added {len(item_ids)} vectors to index. Total: {len(self.item_ids)}")

    def search(
            self,
            query_vector: np.ndarray,
            k: int = config.TOP_K_SIMILAR
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors

        Args:
            query_vector: Query vector of shape (embedding_dim,)
            k: Number of neighbors to retrieve

        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() or load_index() first.")

        # Normalize for cosine similarity if needed
        if self.index_type == "Cosine" or self.index_type == "IP":
            query_vector = utils.normalize_embeddings(query_vector.reshape(1, -1))[0]

        query_vector = query_vector.astype('float32').reshape(1, -1)

        distances, indices = self.index.search(query_vector, k)

        return distances[0], indices[0]

    def get_similar_items(
            self,
            item_id: str,
            k: int = config.TOP_K_SIMILAR,
            exclude_self: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get k most similar items to a given item

        Args:
            item_id: ID of the query item
            k: Number of similar items to retrieve
            exclude_self: Whether to exclude the query item itself

        Returns:
            List of tuples (item_id, similarity_score)
        """
        if item_id not in self.id_to_index:
            raise ValueError(f"Item ID {item_id} not found in index")

        # Get the item's embedding from the index
        item_idx = self.id_to_index[item_id]
        item_vector = self.index.reconstruct(item_idx)

        # Search for k+1 neighbors (to account for excluding self)
        search_k = k + 1 if exclude_self else k
        distances, indices = self.search(item_vector, search_k)

        # Convert indices to item IDs and filter out self
        similar_items = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.item_ids):
                similar_item_id = self.item_ids[idx]
                if exclude_self and similar_item_id == item_id:
                    continue

                # Convert distance to similarity score
                if self.index_type == "L2":
                    # Convert L2 distance to similarity (lower distance = higher similarity)
                    similarity = 1.0 / (1.0 + dist)
                elif self.index_type == "IP" or self.index_type == "Cosine":
                    # For IP/Cosine, higher value = higher similarity
                    similarity = float(dist)
                else:
                    similarity = float(dist)

                similar_items.append((similar_item_id, similarity))

                if len(similar_items) >= k:
                    break

        return similar_items

    def save_index(self, index_path: Path = None, ids_path: Path = None) -> None:
        """Save index and item IDs to disk"""
        if self.index is None:
            raise ValueError("No index to save")

        index_path = index_path or config.FAISS_INDEX_PATH
        ids_path = ids_path or config.FAISS_ITEM_IDS_PATH

        # Convert GPU index to CPU before saving (if using GPU version)
        index_to_save = self.index

        # Check if index might be on GPU (only if faiss-gpu is available)
        try:
            index_type_str = str(type(self.index))
            if 'Gpu' in index_type_str and hasattr(faiss, 'index_gpu_to_cpu'):
                index_to_save = faiss.index_gpu_to_cpu(self.index)
        except (AttributeError, RuntimeError):
            # Index is on CPU (faiss-cpu) or conversion not needed
            pass

        faiss.write_index(index_to_save, str(index_path))
        utils.save_json(self.item_ids, ids_path)

        print(f"Index saved to {index_path}")
        print(f"Item IDs saved to {ids_path}")

    def load_index(self, index_path: Path = None, ids_path: Path = None) -> None:
        """Load index and item IDs from disk"""
        index_path = index_path or config.FAISS_INDEX_PATH
        ids_path = ids_path or config.FAISS_ITEM_IDS_PATH

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"Item IDs file not found: {ids_path}")

        self.index = faiss.read_index(str(index_path))
        self.item_ids = utils.load_json(ids_path)

        # Rebuild id_to_index mapping
        self.id_to_index = {item_id: i for i, item_id in enumerate(self.item_ids)}

        print(f"Loaded index with {len(self.item_ids)} vectors from {index_path}")

