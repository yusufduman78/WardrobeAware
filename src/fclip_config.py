"""
Configuration file for Fashion Compatibility System
"""
import os
from pathlib import Path

# Base paths
# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from backend import config as backend_config
    DATA_DIR = backend_config.DATA_ROOT
    METADATA_PATH = backend_config.METADATA_PATH
    IMAGE_DIR = backend_config.IMAGES_DIR
except ImportError:
    print("Warning: Could not import backend.config. Using local path definition.")
    DATA_DIR = BASE_DIR / "data" / "polyvore_outfits"
    METADATA_PATH = DATA_DIR / "polyvore_item_metadata.json"
    IMAGE_DIR = DATA_DIR / "images"

OUTPUT_DIR = BASE_DIR / "experiments" / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Dataset paths
TRAIN_SPLIT_PATH = DATA_DIR / "disjoint" / "train.json"
TEST_SPLIT_PATH = DATA_DIR / "disjoint" / "test.json"
VALID_SPLIT_PATH = DATA_DIR / "disjoint" / "valid.json"
COMPATIBILITY_TRAIN_FILE = DATA_DIR / "disjoint" / "compatibility_train.txt"

# Model configuration
FASHION_CLIP_MODEL_NAME = "fashion-clip"
EMBEDDING_DIM = 512  # Fashion-CLIP produces 512-dimensional embeddings
BATCH_SIZE = 128
DEVICE = "cuda"  # Will be set to "cpu" if CUDA not available

# Faiss index configuration
FAISS_INDEX_TYPE = "L2"  # Options: "L2", "IP" (Inner Product), "Cosine"
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss_index.bin"
FAISS_ITEM_IDS_PATH = OUTPUT_DIR / "faiss_item_ids.json"

# Embedding output paths (hybrid mode)
EMBEDDINGS_OUTPUT_DIR = OUTPUT_DIR / "embeddings"
EMBEDDINGS_OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_EMBEDDINGS_PATH = EMBEDDINGS_OUTPUT_DIR / "image_embeddings.npy"
TEXT_EMBEDDINGS_PATH = EMBEDDINGS_OUTPUT_DIR / "text_embeddings.npy"

ITEM_IDS_PATH = EMBEDDINGS_OUTPUT_DIR / "item_ids.json"

# Metric learning configuration
TRIPLET_MARGIN = 0.5
HARD_TRIPLET_MINING_ENABLED = True
NUM_HARD_NEGATIVES = 5
NUM_HARD_POSITIVES = 5

# Similarity search configuration
TOP_K_SIMILAR = 5  # Number of similar products to retrieve

# Training configuration
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
SAVE_CHECKPOINT_EVERY = 5
TRAIN_LOGS_DIR = OUTPUT_DIR / "logs"
TRAIN_PLOTS_DIR = OUTPUT_DIR / "plots"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# Ensure directories exist
TRAIN_LOGS_DIR.mkdir(exist_ok=True)
TRAIN_PLOTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

