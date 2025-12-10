import os
from pathlib import Path

# --- Dosya Yolları ---
# Absolute path based on this file's location
# This file is in <project_root>/recommender/config.py

import sys

# Add project root to sys.path to allow importing backend
# BASE_DIR is project root
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from backend import config as backend_config
    DATA_ROOT = backend_config.DATA_ROOT
    METADATA_PATH = backend_config.METADATA_PATH
    IMAGES_ROOT = backend_config.IMAGES_DIR
except ImportError:
    # Fallback if backend module not found (e.g. running standalone script in recommender folder)
    print("Warning: Could not import backend.config. Using local path definition.")
    DATA_ROOT = BASE_DIR / "data" / "polyvore_outfits"
    METADATA_PATH = DATA_ROOT / "polyvore_item_metadata.json"
    IMAGES_ROOT = DATA_ROOT / "images"

# Eğitim seti olarak 'nondisjoint' (daha fazla veri) veya 'disjoint' seçebilirsin.
# Makale disjoint üzerinde test ediliyor ama eğitim için nondisjoint genelde daha zengin.
SPLIT_DIR = DATA_ROOT / "nondisjoint"
TRAIN_JSON = SPLIT_DIR / "train.json"
VALID_JSON = SPLIT_DIR / "valid.json"

# --- Hiperparametreler (Makaleden referansla) ---
BATCH_SIZE = 128        # GPU gücüne göre 256 yapabilirsin
EMBEDDING_DIM = 64      # Makaledeki boyut
LEARNING_RATE = 5e-5    # Makaledeki LR
NUM_EPOCHS = 15
MARGIN = 0.2            # Triplet Loss Margin
LAMBDA_L1 = 5e-4        # Mask sparsity regularization

# Görüntü Ayarları
IMAGE_SIZE = 224
NUM_WORKERS = 4         # Veri yükleme hızı için