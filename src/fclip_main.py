"""
Main script for Fashion Compatibility System
Demonstrates finding 5 similar products to a selected item
"""
import argparse
from pathlib import Path

from src.fclip_embedding_extractor import EmbeddingExtractor
from src.fclip_similarity_search import SimilaritySearch
from src import fclip_config as config
from utils import fclip_utils as utils


def extract_embeddings(split_path: Path = config.TRAIN_SPLIT_PATH, use_text: bool = True):
    """Extract embeddings from dataset"""
    print("=" * 60)
    print("STEP 1: Extracting Embeddings")
    print("=" * 60)

    extractor = EmbeddingExtractor()
    metadata = utils.load_metadata()

    extractor.process_dataset(
        split_path=split_path,
        metadata=metadata,
        use_text=use_text,
        save_embeddings=True
    )

    print("\nEmbedding extraction completed!\n")


def build_faiss_index():
    """Build Faiss index from embeddings"""
    print("=" * 60)
    print("STEP 2: Building Faiss Index")
    print("=" * 60)

    search_system = SimilaritySearch(model_path=config.MODELS_DIR / "projection_model.pth")

    # Check if combined embeddings exist, otherwise use image embeddings
    # Check if image embeddings exist
    if config.IMAGE_EMBEDDINGS_PATH.exists():
        embeddings_path = config.IMAGE_EMBEDDINGS_PATH
        embedding_dim = config.EMBEDDING_DIM
    else:
        raise FileNotFoundError(
            "No embeddings found. Please run embedding extraction first."
        )

    # Update embedding dimension in FaissManager
    search_system.faiss_manager.embedding_dim = embedding_dim

    search_system.build_index_from_embeddings(
        embeddings_path=embeddings_path,
        item_ids_path=config.ITEM_IDS_PATH,
        save_index=True
    )

    print("\nFaiss index built successfully!\n")


def find_similar_products(item_id: str, top_k: int = 5):
    """Find and display similar products"""
    print("=" * 60)
    print("STEP 3: Finding Similar Products")
    print("=" * 60)

    search_system = SimilaritySearch(model_path=config.MODELS_DIR / "projection_model.pth")

    # Try to load existing index, otherwise build it
    try:
        search_system.load_index()
    except FileNotFoundError:
        print("Index not found. Building index...")
        build_faiss_index()
        search_system.load_index()

    # Display similar products
    search_system.display_similar_products(item_id, top_k=top_k)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fashion Compatibility System - Find Similar Products"
    )
    parser.add_argument(
        '--extract-embeddings',
        action='store_true',
        help='Extract embeddings from dataset'
    )
    parser.add_argument(
        '--build-index',
        action='store_true',
        help='Build Faiss index from embeddings'
    )
    parser.add_argument(
        '--find-similar',
        type=str,
        help='Find similar products for given item ID'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of similar products to retrieve (default: 5)'
    )
    parser.add_argument(
        '--use-text',
        action='store_true',
        default=True,
        help='Use text embeddings in addition to image embeddings'
    )

    args = parser.parse_args()

    # If no arguments provided, show help and run example
    if not any([args.extract_embeddings, args.build_index, args.find_similar]):
        print("Fashion Compatibility System")
        print("=" * 60)
        print("\nUsage examples:")
        print("  python main.py --extract-embeddings")
        print("  python main.py --build-index")
        print("  python main.py --find-similar <item_id>")
        print("\nRunning full pipeline with example item...")
        print("=" * 60)

        # Run full pipeline
        if not config.IMAGE_EMBEDDINGS_PATH.exists():
            print("\nExtracting embeddings first...")
            extract_embeddings(use_text=args.use_text)

        if not config.FAISS_INDEX_PATH.exists():
            print("\nBuilding index...")
            build_faiss_index()

        # Get a sample item ID
        item_ids = utils.load_json(config.ITEM_IDS_PATH)
        if item_ids:
            sample_item_id = item_ids[0]
            print(f"\nFinding similar products for sample item: {sample_item_id}")
            find_similar_products(sample_item_id, top_k=args.top_k)
        else:
            print("No items found in dataset.")
    else:
        # Run requested operations
        if args.extract_embeddings:
            extract_embeddings(use_text=args.use_text)

        if args.build_index:
            build_faiss_index()

        if args.find_similar:
            find_similar_products(args.find_similar, top_k=args.top_k)


if __name__ == "__main__":
    main()

