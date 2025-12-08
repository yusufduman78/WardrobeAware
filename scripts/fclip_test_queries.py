"""
Test script to evaluate model performance on specific queries
"""
import argparse
import json
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict
import pandas as pd

from src import fclip_config as config
from utils import fclip_utils as utils
from src.fclip_embedding_extractor import EmbeddingExtractor
from src.fclip_faiss_manager import FaissIndexManager
from src.fclip_metric_learning import EmbeddingProjection


def load_projection_model(model_path: Path, input_dim: int) -> EmbeddingProjection:
    """Load trained projection model"""
    model = EmbeddingProjection(input_dim=input_dim, output_dim=512)
    model.load_state_dict(torch.load(model_path, map_location=utils.get_device()))
    model.to(utils.get_device())
    model.eval()
    return model


def run_queries(
        queries: List[str],
        top_k: int = 5,
        model_path: str = None,
        output_file: str = None
):
    """Run queries and print results"""
    print(f"Running {len(queries)} queries...")

    # Initialize components
    extractor = EmbeddingExtractor()
    metadata = utils.load_metadata()

    # Load embeddings to determine dimension
    embeddings_path = config.IMAGE_EMBEDDINGS_PATH
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")

    sample = np.load(embeddings_path, mmap_mode='r')
    embedding_dim = sample.shape[1]

    print(f"Embedding dimension: {embedding_dim}")

    # Load Faiss index
    faiss_manager = FaissIndexManager(embedding_dim=512 if model_path else embedding_dim,
                                      index_type="IP")  # Use IP for cosine similarity

    # If using a projection model, we need to rebuild the index with projected embeddings
    # For simplicity in this test script, we'll just project the query and search in the original index
    # IF the index was also built with projected embeddings.
    # BUT, for the 'verification' step, we usually want to see how the projected query matches the projected database.
    # Since rebuilding the whole index for a quick test is slow, we will assume for this script:
    # 1. Baseline: Query (Text) -> Index (Image+Text)
    # 2. Trained: Query (Text) -> Projection -> Index (Image+Text -> Projection)

    # Actually, to properly test the trained model, we MUST project the database too.
    # So if model_path is provided, we load embeddings, project them, and build a temporary in-memory index.

    if model_path:
        print(f"Loading projection model from {model_path}...")
        projection_model = load_projection_model(Path(model_path), embedding_dim)

        print("Projecting database embeddings...")
        original_embeddings = np.load(embeddings_path)

        # Process in batches to avoid OOM
        projected_list = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(original_embeddings), batch_size):
                batch = torch.FloatTensor(original_embeddings[i:i + batch_size]).to(utils.get_device())
                proj = projection_model(batch)
                projected_list.append(proj.cpu().numpy())

        database_embeddings = np.concatenate(projected_list)
        item_ids = utils.load_json(config.ITEM_IDS_PATH)

        # Build temp index
        faiss_manager.embedding_dim = 512  # Output dim of projection
        faiss_manager.create_index(len(database_embeddings))
        faiss_manager.add_vectors(database_embeddings, item_ids)

    else:
        # Load existing index
        try:
            faiss_manager.load_index()
        except FileNotFoundError:
            print("Index not found. Please run 'python main.py --build-index' first.")
            return

    results_data = {}

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        # Extract query embedding
        text_embedding = extractor.extract_text_embeddings([query])[0]

        # Handle dimension mismatch for Combined Index (1024)
        if embedding_dim == 1024 and text_embedding.shape[0] == 512:
            # Combined is [Image, Text], so we pad Image part with zeros to match text-to-text search
            # Query = [Zeros(512), Text(512)]
            zeros = np.zeros(512, dtype=text_embedding.dtype)
            text_embedding = np.concatenate([zeros, text_embedding])

        # If model exists, project the query too
        if model_path:
            with torch.no_grad():
                query_tensor = torch.FloatTensor(text_embedding).unsqueeze(0).to(utils.get_device())
                # Note: The projection model expects input of 'embedding_dim'.
                # If our database is Combined (Image+Text = 1024) but we query with Text (512), we have a dimension mismatch.
                # For this specific test case (Text-to-Image retrieval), we should ideally use the Text part of the space.
                # However, if the model was trained on Combined embeddings, it expects 1024.
                # Padding with zeros is a naive way, but better is to ensure training data matches usage.
                # For this task, let's assume we are querying with text against the database.

                if embedding_dim == 1024 and text_embedding.shape[0] == 512:
                    # Pad with zeros to match combined dimension if needed
                    # Or better, just duplicate text or use image placeholder.
                    # Let's try zero padding for the image part (first 512 usually image, last 512 text)
                    # Wait, usually combined is [Image, Text].
                    # If we query with text, we might want to match the Text part.
                    # Let's construct a 1024 vector: [Zeros, Text]
                    combined_query = np.concatenate([np.zeros(512), text_embedding])
                    query_tensor = torch.FloatTensor(combined_query).unsqueeze(0).to(utils.get_device())

                text_embedding = projection_model(query_tensor).cpu().numpy()[0]

        # Search
        distances, indices = faiss_manager.search(text_embedding, k=top_k)

        query_results = []
        for dist, idx in zip(distances, indices):
            if idx < len(faiss_manager.item_ids):
                item_id = faiss_manager.item_ids[idx]
                item_meta = metadata.get(item_id, {})

                title = item_meta.get('title', 'N/A')
                desc = item_meta.get('description', 'N/A')

                print(f"  [{dist:.4f}] {title}")

                query_results.append({
                    'item_id': item_id,
                    'score': float(dist),
                    'title': title,
                    'description': desc
                })

        results_data[query] = query_results

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', action='append', required=True, help='Text query (can be used multiple times)')
    parser.add_argument('--model-path', type=str, help='Path to trained projection model')
    parser.add_argument('--save-results', type=str, help='File to save results JSON')
    args = parser.parse_args()

    run_queries(args.query, model_path=args.model_path, output_file=args.save_results)
