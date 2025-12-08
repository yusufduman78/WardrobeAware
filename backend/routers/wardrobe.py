from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body
from typing import Optional, List
from pydantic import BaseModel
import shutil
import os
from pathlib import Path
from PIL import Image
import io
import uuid
import sys
import numpy as np
import torch

from backend import config
from backend.database import get_db_connection
from src.segmentation import segmenter

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Lazy import for FashionCLIP (heavy dependency)
EmbeddingExtractor = None
try:
    from src.fclip_embedding_extractor import EmbeddingExtractor
except ImportError:
    print("Warning: Could not import EmbeddingExtractor. Upload recommendations will not work.")

router = APIRouter(
    prefix="/wardrobe",
    tags=["wardrobe"]
)

# Ensure upload directory exists
UPLOAD_DIR = Path(config.IMAGES_DIR) / "user_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Store confirmed categories for uploaded items (item_id -> category)
UPLOADED_CATEGORIES_FILE = UPLOAD_DIR / "uploaded_categories.json"
UPLOADED_CATEGORIES = {}

# Store embeddings for uploaded items (item_id -> embedding list)
UPLOADED_EMBEDDINGS_FILE = UPLOAD_DIR / "uploaded_embeddings.json"
UPLOADED_EMBEDDINGS = {}

def load_uploaded_categories():
    """Load confirmed categories for uploaded items"""
    global UPLOADED_CATEGORIES
    if UPLOADED_CATEGORIES_FILE.exists():
        try:
            import json
            with open(UPLOADED_CATEGORIES_FILE, 'r') as f:
                UPLOADED_CATEGORIES = json.load(f)
            print(f"Loaded {len(UPLOADED_CATEGORIES)} confirmed categories for uploaded items")
        except Exception as e:
            print(f"Error loading uploaded categories: {e}")
            UPLOADED_CATEGORIES = {}

def save_uploaded_categories():
    """Save confirmed categories for uploaded items"""
    try:
        import json
        with open(UPLOADED_CATEGORIES_FILE, 'w') as f:
            json.dump(UPLOADED_CATEGORIES, f, indent=2)
    except Exception as e:
        print(f"Error saving uploaded categories: {e}")

def load_uploaded_embeddings():
    """Load cached embeddings for uploaded items"""
    global UPLOADED_EMBEDDINGS
    if UPLOADED_EMBEDDINGS_FILE.exists():
        try:
            import json
            import numpy as np
            with open(UPLOADED_EMBEDDINGS_FILE, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays
                UPLOADED_EMBEDDINGS = {
                    item_id: np.array(embedding) 
                    for item_id, embedding in data.items()
                }
            print(f"Loaded {len(UPLOADED_EMBEDDINGS)} cached embeddings for uploaded items")
        except Exception as e:
            print(f"Error loading uploaded embeddings: {e}")
            UPLOADED_EMBEDDINGS = {}

def save_uploaded_embeddings():
    """Save cached embeddings for uploaded items"""
    try:
        import json
        import numpy as np
        # Convert numpy arrays to lists for JSON serialization
        data = {
            item_id: embedding.tolist() 
            for item_id, embedding in UPLOADED_EMBEDDINGS.items()
        }
        with open(UPLOADED_EMBEDDINGS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving uploaded embeddings: {e}")

def get_uploaded_embedding(item_id):
    """Get cached embedding for uploaded item, or None if not cached"""
    global UPLOADED_EMBEDDINGS
    return UPLOADED_EMBEDDINGS.get(item_id)

def cache_uploaded_embedding(item_id, embedding):
    """Cache embedding for uploaded item"""
    global UPLOADED_EMBEDDINGS
    UPLOADED_EMBEDDINGS[item_id] = embedding
    save_uploaded_embeddings()

# Load on startup
load_uploaded_categories()
load_uploaded_embeddings()

# Global embedding extractor (lazy load)
_embedding_extractor = None

def get_embedding_extractor():
    global _embedding_extractor
    if _embedding_extractor is None and EmbeddingExtractor is not None:
        try:
            _embedding_extractor = EmbeddingExtractor()
            print("EmbeddingExtractor initialized for wardrobe recommendations")
        except Exception as e:
            print(f"Error initializing EmbeddingExtractor: {e}")
    return _embedding_extractor

@router.post("/upload")
async def upload_item(
    file: UploadFile = File(...),
    user_id: str = Form("test_user"),
    category: Optional[str] = Form(None)
):
    """
    Uploads an image, segments it, classifies it, and returns category predictions.
    User should confirm or change the category before using it for recommendations.
    """
    try:
        print(f"\n{'='*80}")
        print(f"[WARDROBE/UPLOAD] Starting upload for user: {user_id}")
        print(f"{'='*80}")
        
        # 1. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        print(f"[WARDROBE/UPLOAD] Image loaded: {image.size}")
        
        # 2. Remove Background (Segment)
        print("[WARDROBE/UPLOAD] Segmenting image...")
        processed_image = segmenter.segment_image(image)
        print("[WARDROBE/UPLOAD] Segmentation complete")
        
        # 3. Save Image
        item_id = str(uuid.uuid4())
        filename = f"{item_id}.png" # Save as PNG for transparency
        save_path = UPLOAD_DIR / filename
        
        processed_image.save(save_path, format="PNG")
        print(f"[WARDROBE/UPLOAD] Saved to {save_path}")
        
        # 4. Classify Image
        predicted_category = None
        category_predictions = []
        all_categories = []
        
        extractor = get_embedding_extractor()
        if extractor is not None:
            print("[WARDROBE/UPLOAD] Extracting embedding for classification...")
            # Convert RGBA to RGB for embedding extraction
            rgb_image = processed_image.convert('RGB')
            
            # Check if embedding is already cached
            cached_embedding = get_uploaded_embedding(item_id)
            if cached_embedding is not None:
                print(f"[WARDROBE/UPLOAD] Using cached embedding")
                uploaded_embedding = cached_embedding
            else:
                uploaded_embedding = extractor.extract_image_embeddings([rgb_image])[0]
                # Cache the embedding for future use
                cache_uploaded_embedding(item_id, uploaded_embedding)
                print(f"[WARDROBE/UPLOAD] Embedding extracted and cached (shape: {uploaded_embedding.shape})")
            
            # Get category predictions
            # Import here to avoid circular import
            import backend.routers.outfit as outfit_router
            
            model = outfit_router.get_recommender()
            if model is not None:
                predicted_category = outfit_router.predict_category_for_uploaded_image(
                    uploaded_embedding, model, extractor
                )
                print(f"[WARDROBE/UPLOAD] Predicted category: {predicted_category}")
                
                # Get all category predictions with probabilities
                classifier_dict, label_encoder = outfit_router.load_category_classifier()
                if classifier_dict is not None and label_encoder is not None:
                    try:
                        if classifier_dict.get('type') == 'fclip':
                            classifier_head = classifier_dict['classifier']
                            device = classifier_dict['device']
                            
                            uploaded_embedding_norm = uploaded_embedding / np.linalg.norm(uploaded_embedding)
                            embedding_tensor = torch.tensor(uploaded_embedding_norm, dtype=torch.float32).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                logits = classifier_head(embedding_tensor)
                                probabilities = torch.softmax(logits, dim=1)
                            
                            prob_values = probabilities.cpu().numpy()[0]
                            category_probs = dict(zip(label_encoder.classes_, prob_values))
                            sorted_categories = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)
                            
                            category_predictions = [
                                {"category": cat, "probability": float(prob), "percentage": float(prob * 100)}
                                for cat, prob in sorted_categories
                            ]
                            
                            all_categories = list(label_encoder.classes_)
                        else:
                            # Sklearn classifier
                            sklearn_classifier = classifier_dict['model']
                            uploaded_embedding_norm = uploaded_embedding / np.linalg.norm(uploaded_embedding)
                            embedding_array = uploaded_embedding_norm.reshape(1, -1)
                            
                            probabilities = sklearn_classifier.predict_proba(embedding_array)[0]
                            category_probs = dict(zip(label_encoder.classes_, probabilities))
                            sorted_categories = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)
                            
                            category_predictions = [
                                {"category": cat, "probability": float(prob), "percentage": float(prob * 100)}
                                for cat, prob in sorted_categories
                            ]
                            
                            all_categories = list(label_encoder.classes_)
                    except Exception as e:
                        print(f"[WARDROBE/UPLOAD] Error getting category predictions: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            print("[WARDROBE/UPLOAD] Warning: EmbeddingExtractor not available, skipping classification")
            # Get all categories from CATEGORY_GROUPS
            all_categories = list(outfit_router.CATEGORY_GROUPS.keys())
        
        # If category was provided manually, use it
        if category:
            predicted_category = category
        
        # 5. Add to Database (Interactions as 'owned')
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO interactions (user_id, item_id, action) VALUES (?, ?, ?)",
            (user_id, item_id, 'owned')
        )
        conn.commit()
        conn.close()
        
        # 6. Return Item Info with category predictions
        image_url = f"{config.BASE_URL}/images/user_uploads/{filename}"
        
        print(f"[WARDROBE/UPLOAD] Upload complete. Item ID: {item_id}")
        print(f"{'='*80}\n")
        
        return {
            "id": item_id,
            "image_url": image_url,
            "name": "My Uploaded Item",
            "predicted_category": predicted_category or "unknown",
            "category_predictions": category_predictions,  # All categories with probabilities
            "all_categories": all_categories,  # List of all available categories
            "needs_confirmation": True  # User should confirm or change category
        }
        
    except Exception as e:
        print(f"[WARDROBE/UPLOAD] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class UpdateCategoryRequest(BaseModel):
    item_id: str
    category: str  # User-confirmed category

@router.get("/get_all_categories")
async def get_all_categories():
    """
    Returns all available categories (11 classes from Polyvore dataset).
    """
    import backend.routers.outfit as outfit_router
    
    # Try to get categories from classifier first (more accurate)
    classifier_dict, label_encoder = outfit_router.load_category_classifier()
    if label_encoder is not None:
        categories = list(label_encoder.classes_)
    else:
        # Fallback to CATEGORY_GROUPS
        categories = sorted(list(outfit_router.CATEGORY_GROUPS.keys()))
    
    return {
        "categories": categories,
        "count": len(categories),
        "note": "User can select one of these categories or choose 'other' if item doesn't fit any category"
    }

@router.post("/update_category")
async def update_category(request: UpdateCategoryRequest):
    """
    Updates the confirmed category for an uploaded item.
    User should call this after reviewing the predicted category.
    """
    item_id = request.item_id
    category = request.category
    
    print(f"\n{'='*80}")
    print(f"[WARDROBE/UPDATE_CATEGORY] Updating category for item: {item_id}")
    print(f"[WARDROBE/UPDATE_CATEGORY] New category: {category}")
    print(f"{'='*80}")
    
    # Check if item exists
    uploaded_file = UPLOAD_DIR / f"{item_id}.png"
    if not uploaded_file.exists():
        raise HTTPException(status_code=404, detail=f"Uploaded item {item_id} not found")
    
    # Validate category (should be one of the available categories or "other")
    import backend.routers.outfit as outfit_router
    
    classifier_dict, label_encoder = outfit_router.load_category_classifier()
    if label_encoder is not None:
        valid_categories = list(label_encoder.classes_) + ["other"]
    else:
        valid_categories = list(outfit_router.CATEGORY_GROUPS.keys()) + ["other"]
    
    if category not in valid_categories:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid category. Must be one of: {valid_categories}"
        )
    
    # Save confirmed category
    global UPLOADED_CATEGORIES
    UPLOADED_CATEGORIES[item_id] = category
    save_uploaded_categories()
    
    print(f"[WARDROBE/UPDATE_CATEGORY] Category confirmed: {category}")
    print(f"{'='*80}\n")
    
    return {
        "item_id": item_id,
        "confirmed_category": category,
        "message": "Category updated successfully. You can now use this item for outfit recommendations."
    }

class RecommendFromUploadRequest(BaseModel):
    uploaded_item_id: str
    limit: int = 10

@router.post("/recommend_from_upload")
async def recommend_from_upload(request: RecommendFromUploadRequest):
    """
    Get recommendations based on an uploaded item.
    Uses FashionCLIP embeddings to find compatible items.
    """
    uploaded_item_id = request.uploaded_item_id
    limit = request.limit
    
    print(f"\n{'='*80}")
    print(f"[WARDROBE/RECOMMEND] Request for uploaded item: {uploaded_item_id}")
    print(f"{'='*80}")
    
    # Check if uploaded file exists
    uploaded_file = UPLOAD_DIR / f"{uploaded_item_id}.png"
    if not uploaded_file.exists():
        raise HTTPException(status_code=404, detail=f"Uploaded item {uploaded_item_id} not found")
    
    # Get embedding extractor
    extractor = get_embedding_extractor()
    if extractor is None:
        raise HTTPException(status_code=503, detail="Embedding extractor not available. FashionCLIP may not be installed.")
    
    # Get recommender model
    from backend.routers.outfit import get_recommender
    model = get_recommender()
    if model is None:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")
    
    try:
        # Load uploaded image
        uploaded_image = Image.open(uploaded_file).convert('RGB')
        print(f"[WARDROBE/RECOMMEND] Loaded uploaded image: {uploaded_file}")
        
        # Extract embedding for uploaded image
        print(f"[WARDROBE/RECOMMEND] Extracting embedding for uploaded image...")
        uploaded_embedding = extractor.extract_image_embeddings([uploaded_image])[0]
        uploaded_embedding_tensor = torch.tensor(uploaded_embedding).to(model.device)
        
        # If using projection model, project the embedding
        if model.model_type == "Projection":
            with torch.no_grad():
                uploaded_embedding_tensor = uploaded_embedding_tensor.unsqueeze(0)
                projected_embedding = model.model(uploaded_embedding_tensor)
        else:
            projected_embedding = uploaded_embedding_tensor.unsqueeze(0)
        
        print(f"[WARDROBE/RECOMMEND] Embedding extracted and projected")
        
        # Get candidate items from metadata
        from backend.routers.outfit import METADATA, CATEGORY_GROUPS
        import random
        
        all_candidate_ids = list(METADATA.keys())
        # Sample more candidates for better results
        sample_size = min(100, len(all_candidate_ids))
        candidate_ids = random.sample(all_candidate_ids, sample_size)
        
        print(f"[WARDROBE/RECOMMEND] Evaluating {len(candidate_ids)} candidates")
        
        scored_candidates = []
        
        for idx, cand_id in enumerate(candidate_ids, 1):
            try:
                # Get candidate embedding
                if model.model_type == "Projection":
                    cand_emb = model.get_embedding(cand_id)
                    if cand_emb is None:
                        continue
                    with torch.no_grad():
                        cand_projected = model.model(cand_emb)
                    
                    # Calculate distance
                    dist = torch.nn.functional.pairwise_distance(projected_embedding, cand_projected).item()
                else:
                    # ResNet model - use existing method
                    dist = model.predict_compatibility(uploaded_item_id, cand_id)
                    # But we need to handle the case where uploaded_item_id is not in metadata
                    # For now, skip ResNet model for uploaded items
                    continue
                
                # Convert to score
                score = np.exp(-dist) * 100
                
                if idx % 20 == 0:
                    print(f"  [PROGRESS] Evaluated {idx}/{len(candidate_ids)} candidates")
                
                scored_candidates.append({
                    "item_id": cand_id,
                    "score": score,
                    "distance": dist
                })
            except Exception as e:
                print(f"  [ERROR] Failed to score candidate {cand_id}: {e}")
                continue
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top N
        top_candidates = scored_candidates[:limit]
        
        print(f"\n[WARDROBE/RECOMMEND] Top {len(top_candidates)} recommendations:")
        for idx, cand in enumerate(top_candidates, 1):
            print(f"  [{idx}] {cand['item_id']} | Distance: {cand['distance']:.4f} | Score: {cand['score']:.2f}%")
        
        # Format response
        recommendations = []
        for cand in top_candidates:
            meta = METADATA.get(cand['item_id'], {})
            image_url = f"{config.BASE_URL}/images/{cand['item_id']}.jpg"
            
            recommendations.append({
                "id": cand['item_id'],
                "image_url": image_url,
                "name": meta.get('url_name', meta.get('title', 'Unknown Item')),
                "price": meta.get('price', 0),
                "match_score": int(cand['score']),
                "category": meta.get('semantic_category', 'unknown')
            })
        
        print(f"{'='*80}\n")
        
        return {"recommendations": recommendations}
        
    except Exception as e:
        print(f"[WARDROBE/RECOMMEND] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
