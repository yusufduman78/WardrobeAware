"""
Similarity Search Module for Finding Similar Products
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

from src.fclip_faiss_manager import FaissIndexManager
from src.fclip_embedding_extractor import EmbeddingExtractor
from src import fclip_config as config
from utils import fclip_utils as utils
import torch
from src.fclip_metric_learning import EmbeddingProjection


class SimilaritySearch:
    """Similarity search system for finding similar fashion items"""

    def __init__(
        self,
        faiss_manager: Optional[FaissIndexManager] = None,

        embedding_extractor: Optional[EmbeddingExtractor] = None,
        model_path: Optional[Path] = None
    ):
        """
        Initialize similarity search system

        Args:
            faiss_manager: Faiss index manager instance
            embedding_extractor: Embedding extractor instance
            model_path: Path to trained projection model
        """
        self.faiss_manager = faiss_manager or FaissIndexManager()
        self.embedding_extractor = embedding_extractor or EmbeddingExtractor()
        self.metadata = utils.load_metadata()
        self.projection_model = None

        if model_path and model_path.exists():
            self.load_projection_model(model_path)
    def load_projection_model(self, model_path: Path) -> None:
        """Load trained projection model"""
        print(f"Loading projection model from {model_path}...")
        # We need to know input dim. For now assume it matches config
        input_dim = config.EMBEDDING_DIM

        self.projection_model = EmbeddingProjection(input_dim=input_dim, output_dim=512)
        self.projection_model.load_state_dict(torch.load(model_path, map_location=utils.get_device()))
        self.projection_model.to(utils.get_device())
        self.projection_model.eval()
        print("Projection model loaded.")

    def build_index_from_embeddings(
        self,
        embeddings_path: Path = config.IMAGE_EMBEDDINGS_PATH,
        item_ids_path: Path = config.ITEM_IDS_PATH,
        save_index: bool = True
    ) -> None:
        """
        Build Faiss index from pre-computed embeddings

        Args:
            embeddings_path: Path to embeddings numpy file
            item_ids_path: Path to item IDs JSON file
            save_index: Whether to save the index to disk
        """
        print("Building Faiss index from embeddings...")

        # Load embeddings and item IDs
        embeddings = np.load(embeddings_path)
        item_ids = utils.load_json(item_ids_path)

        if len(embeddings) != len(item_ids):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(item_ids)} item IDs")

        # Create and populate index
        # Create and populate index

        # If projection model exists, project embeddings before indexing
        if self.projection_model:
            print("Projecting embeddings with trained model...")
            projected_list = []
            batch_size = 1024
            with torch.no_grad():
                for i in range(0, len(embeddings), batch_size):
                    batch = torch.FloatTensor(embeddings[i:i+batch_size]).to(utils.get_device())
                    proj = self.projection_model(batch)
                    projected_list.append(proj.cpu().numpy())
            embeddings = np.concatenate(projected_list)

        self.faiss_manager.create_index(len(embeddings))
        self.faiss_manager.add_vectors(embeddings, item_ids)

        if save_index:
            self.faiss_manager.save_index()

        print(f"Index built with {len(item_ids)} items")

    def load_index(self) -> None:
        """Load pre-built Faiss index from disk"""
        self.faiss_manager.load_index()
        print(f"Loaded index with {len(self.faiss_manager.item_ids)} items")

    def find_similar_products(
        self,
        item_id: str,
        top_k: int = config.TOP_K_SIMILAR,
        return_metadata: bool = True
    ) -> List[Dict]:
        """
        Find top-k similar products to a given item

        Args:
            item_id: ID of the query item
            top_k: Number of similar products to return
            return_metadata: Whether to include item metadata in results

        Returns:
            List of dictionaries containing similar item information
        """
        if len(self.faiss_manager.item_ids) == 0:
            raise ValueError("Index is empty. Build or load index first.")

        # Get similar items from Faiss
        similar_items = self.faiss_manager.get_similar_items(item_id, k=top_k)

        # Format results
        results = []
        for similar_item_id, similarity_score in similar_items:
            result = {
                'item_id': similar_item_id,
                'similarity_score': float(similarity_score)
            }

            if return_metadata:
                item_meta = self.metadata.get(similar_item_id, {})
                result['title'] = item_meta.get('title', 'N/A')
                result['description'] = item_meta.get('description', 'N/A')
                result['category'] = item_meta.get('semantic_category', 'N/A')
                result['image_path'] = str(config.IMAGE_DIR / f"{similar_item_id}.jpg")

            results.append(result)

        return results

    def find_similar_by_image(
        self,
        image_path: Path,
        top_k: int = config.TOP_K_SIMILAR,
        return_metadata: bool = True
    ) -> List[Dict]:
        """
        Find similar products by querying with an image

        Args:
            image_path: Path to query image
            top_k: Number of similar products to return
            return_metadata: Whether to include item metadata in results

        Returns:
            List of dictionaries containing similar item information
        """
        # Load and extract embedding from query image
        image = utils.load_image(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        query_embedding = self.embedding_extractor.extract_image_embeddings([image])[0]

        # Project query if model exists
        if self.projection_model:
            with torch.no_grad():
                query_tensor = torch.FloatTensor(query_embedding).unsqueeze(0).to(utils.get_device())
                query_embedding = self.projection_model(query_tensor).cpu().numpy()[0]

        # Search in Faiss index
        distances, indices = self.faiss_manager.search(query_embedding, k=top_k)

        # Format results
        results = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.faiss_manager.item_ids):
                item_id = self.faiss_manager.item_ids[idx]

                # Convert distance to similarity
                if self.faiss_manager.index_type == "L2":
                    similarity = 1.0 / (1.0 + dist)
                else:
                    similarity = float(dist)

                result = {
                    'item_id': item_id,
                    'similarity_score': similarity
                }

                if return_metadata:
                    item_meta = self.metadata.get(item_id, {})
                    result['title'] = item_meta.get('title', 'N/A')
                    result['description'] = item_meta.get('description', 'N/A')
                    result['category'] = item_meta.get('semantic_category', 'N/A')
                    result['image_path'] = str(config.IMAGE_DIR / f"{item_id}.jpg")

                results.append(result)

        return results

    def find_similar_by_text(
        self,
        text_query: str,
        top_k: int = config.TOP_K_SIMILAR,
        return_metadata: bool = True
    ) -> List[Dict]:
        """
        Find similar products by text query

        Args:
            text_query: Text description to search for
            top_k: Number of similar products to return
            return_metadata: Whether to include item metadata in results

        Returns:
            List of dictionaries containing similar item information
        """
        # Extract text embedding
        # Note: extract_text_embeddings returns a batch, we take the first one
        text_embedding = self.embedding_extractor.extract_text_embeddings([text_query])[0]

        # Project query if model exists
        if self.projection_model:
            with torch.no_grad():
                text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0).to(utils.get_device())
                text_embedding = self.projection_model(text_tensor).cpu().numpy()[0]

        # Search in Faiss index
        distances, indices = self.faiss_manager.search(text_embedding, k=top_k)

        # Format results
        results = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.faiss_manager.item_ids):
                item_id = self.faiss_manager.item_ids[idx]

                # Convert distance to similarity
                if self.faiss_manager.index_type == "L2":
                    similarity = 1.0 / (1.0 + dist)
                else:
                    similarity = float(dist)

                result = {
                    'item_id': item_id,
                    'similarity_score': similarity
                }

                if return_metadata:
                    item_meta = self.metadata.get(item_id, {})
                    result['title'] = item_meta.get('title', 'N/A')
                    result['description'] = item_meta.get('description', 'N/A')
                    result['category'] = item_meta.get('semantic_category', 'N/A')
                    result['image_path'] = str(config.IMAGE_DIR / f"{item_id}.jpg")

                results.append(result)

        return results

    def display_similar_products(
        self,
        query: str,
        top_k: int = config.TOP_K_SIMILAR
    ) -> None:
        """
        Display top-k similar products for a given item ID or text query

        Args:
            query: Item ID or text description
            top_k: Number of similar products to display
        """
        print(f"\n{'='*60}")

        # Check if query is an existing item_id
        is_item_id = query in self.faiss_manager.item_ids

        if is_item_id:
            print(f"Finding {top_k} similar products for Item ID: {query}")
            print(f"{'='*60}\n")

            # Get query item metadata
            query_meta = self.metadata.get(query, {})
            print(f"Query Item:")
            print(f"  ID: {query}")
            print(f"  Title: {query_meta.get('title', 'N/A')}")
            print(f"  Category: {query_meta.get('semantic_category', 'N/A')}")
            print(f"\n{'-'*60}\n")

            # Find similar products by ID
            similar_products = self.find_similar_products(query, top_k=top_k)

        else:
            print(f"Finding {top_k} similar products for Text Query: '{query}'")
            print(f"{'='*60}\n")

            # Find similar products by Text
            similar_products = self.find_similar_by_text(query, top_k=top_k)

        print(f"Top {len(similar_products)} Similar Products:\n")
        for i, product in enumerate(similar_products, 1):
            print(f"{i}. Item ID: {product['item_id']}")
            print(f"   Title: {product.get('title', 'N/A')}")
            print(f"   Category: {product.get('category', 'N/A')}")
            print(f"   Similarity Score: {product['similarity_score']:.4f}")
            print(f"   Image: {product.get('image_path', 'N/A')}")
            print()


