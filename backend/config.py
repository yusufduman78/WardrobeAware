import os
from pathlib import Path

# Centralized Configuration
# Change this IP to your machine's LAN IP
HOST_IP = "192.168.1.6" 
PORT = "8000"
BASE_URL = f"http://{HOST_IP}:{PORT}"

# Data Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data" / "polyvore_outfits"

METADATA_PATH = DATA_ROOT / "polyvore_item_metadata.json"
IMAGES_DIR = DATA_ROOT / "images"
