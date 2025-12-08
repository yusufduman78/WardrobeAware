from fastapi import APIRouter, HTTPException
import random
from pathlib import Path
from utils import fclip_utils as utils
from src import fclip_config as config
from backend import config as backend_config

router = APIRouter(prefix="/feed", tags=["feed"])

# Cache metadata in memory
METADATA = None

def get_metadata():
    global METADATA
    if METADATA is None:
        try:
            METADATA = utils.load_metadata()
        except Exception as e:
            print(f"Error loading metadata: {e}")
            METADATA = {}
    return METADATA

@router.get("")
async def get_feed(limit: int = 10):
    metadata = get_metadata()
    if not metadata:
        raise HTTPException(status_code=500, detail="Metadata not loaded")
    
    # Get random items
    # Metadata is a dict: {item_id: {details}}
    all_ids = list(metadata.keys())
    selected_ids = random.sample(all_ids, min(limit, len(all_ids)))
    
    feed_items = []
    print(f"DEBUG: Selected {len(selected_ids)} random items for feed.")
    images_dir = backend_config.IMAGES_DIR
    missing_images = []
    
    for item_id in selected_ids:
        item = metadata[item_id]
        # Construct local image URL
        image_url = f"{backend_config.BASE_URL}/images/{item_id}.jpg"
        
        # Check if image file exists
        image_path = images_dir / f"{item_id}.jpg"
        if not image_path.exists():
            missing_images.append(str(item_id))
        
        feed_items.append({
            "id": item_id,
            "name": utils.get_item_text(item),
            "price": item.get("price", 0),
            "likes": item.get("likes", 0),
            "image_url": image_url,
            "category_id": item.get("category_id", "")
        })
    
    if missing_images:
        print(f"WARNING: {len(missing_images)} images not found. Sample IDs: {missing_images[:5]}")
    else:
        print(f"DEBUG: All {len(feed_items)} images found in directory.")
    
    print(f"DEBUG: Returning {len(feed_items)} items to frontend.")
    return feed_items
