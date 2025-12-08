from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
import random
import numpy as np
import torch
import json
from pathlib import Path
from PIL import Image
import sys

# Backend config
from backend import config

# Recommender Imports
import sys
# Add project root to sys.path to allow importing recommender module
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from recommender.inference import TypeAwareRecommender
except ImportError:
    print("Warning: Could not import TypeAwareRecommender. Make sure you are running from project root.")
    TypeAwareRecommender = None

# Lazy import for FashionCLIP (for uploaded images)
EmbeddingExtractor = None
try:
    from src.fclip_embedding_extractor import EmbeddingExtractor
except ImportError:
    print("Warning: Could not import EmbeddingExtractor. Uploaded image recommendations may not work.")

router = APIRouter(
    prefix="/outfit",
    tags=["outfit"]
)

# ----------------------------------------------------------------------
# Model Initialization
# ----------------------------------------------------------------------
recommender = None

def get_recommender():
    global recommender
    if recommender is None and TypeAwareRecommender is not None:
        try:
            # Model path: models/model_epoch_15.pth (relative to project root)
            # We assume the backend is run from project root or we construct absolute path
            base_path = Path(__file__).parent.parent.parent
            model_path = base_path / "models" / "model_epoch_15.pth"
            
            if model_path.exists():
                recommender = TypeAwareRecommender(model_path=str(model_path))
                print(f"TypeAwareRecommender loaded successfully from {model_path}")
            else:
                # Try alternative paths
                alt_paths = [
                    base_path / "models" / "projection_model.pth",
                    base_path / "model_epoch_15.pth",
                    Path("models/model_epoch_15.pth"),
                    Path("model_epoch_15.pth")
                ]
                loaded = False
                for alt_path in alt_paths:
                    if alt_path.exists():
                        recommender = TypeAwareRecommender(model_path=str(alt_path))
                        print(f"TypeAwareRecommender loaded successfully from {alt_path}")
                        loaded = True
                        break
                
                if not loaded:
                    print(f"Warning: Model checkpoint not found. Tried:")
                    print(f"  - {model_path}")
                    for alt_path in alt_paths:
                        print(f"  - {alt_path}")
                    print("Trying to load with default path...")
                    # Try with default (inference.py will handle it)
                    recommender = TypeAwareRecommender()
        except Exception as e:
            print(f"Error loading TypeAwareRecommender: {e}")
            import traceback
            traceback.print_exc()
    return recommender

# Initialize on import (optional, or lazy load)
get_recommender()

# Global embedding extractor for uploaded images
_embedding_extractor = None

def get_embedding_extractor():
    global _embedding_extractor
    if _embedding_extractor is None and EmbeddingExtractor is not None:
        try:
            _embedding_extractor = EmbeddingExtractor()
            print("EmbeddingExtractor initialized for outfit recommendations")
        except Exception as e:
            print(f"Error initializing EmbeddingExtractor: {e}")
    return _embedding_extractor

# Upload directory for user images
UPLOAD_DIR = Path(config.IMAGES_DIR) / "user_uploads"

# Category Classifier (lazy load)
_category_classifier = None
_category_label_encoder = None

def load_category_classifier():
    """Load the trained FashionCLIP category classifier"""
    global _category_classifier, _category_label_encoder
    if _category_classifier is None:
        try:
            import pickle
            import torch
            from torch import nn
            
            # Try FashionCLIP classifier first (preferred)
            fclip_classifier_path = Path(__file__).parent.parent.parent / "models" / "fclip_category_classifier.pth"
            fclip_label_encoder_path = Path(__file__).parent.parent.parent / "models" / "fclip_category_label_encoder.pkl"
            
            if fclip_classifier_path.exists() and fclip_label_encoder_path.exists():
                # Load FashionCLIP classifier
                checkpoint = torch.load(fclip_classifier_path, map_location='cpu')
                num_categories = checkpoint['num_categories']
                
                # Load FashionCLIP model
                from src.fclip_embedding_extractor import EmbeddingExtractor
                extractor = get_embedding_extractor()
                if extractor is None:
                    raise Exception("EmbeddingExtractor not available")
                
                # Create classifier head
                classifier_head = nn.Linear(512, num_categories)
                classifier_head.load_state_dict(checkpoint['model_state_dict'])
                classifier_head.eval()
                
                # Store both extractor and classifier
                _category_classifier = {
                    'type': 'fclip',
                    'extractor': extractor,
                    'classifier': classifier_head,
                    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                }
                _category_classifier['classifier'].to(_category_classifier['device'])
                
                with open(fclip_label_encoder_path, 'rb') as f:
                    _category_label_encoder = pickle.load(f)
                
                print(f"FashionCLIP category classifier loaded from {fclip_classifier_path}")
                print(f"  Categories: {len(_category_label_encoder.classes_)}")
                print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
            else:
                # Fallback to sklearn classifier
                classifier_path = Path(__file__).parent.parent.parent / "models" / "category_classifier.pkl"
                label_encoder_path = Path(__file__).parent.parent.parent / "models" / "category_label_encoder.pkl"
                
                if classifier_path.exists():
                    with open(classifier_path, 'rb') as f:
                        _category_classifier = {'type': 'sklearn', 'model': pickle.load(f)}
                    print(f"Sklearn category classifier loaded from {classifier_path}")
                    
                    if label_encoder_path.exists():
                        with open(label_encoder_path, 'rb') as f:
                            _category_label_encoder = pickle.load(f)
                        print(f"Label encoder loaded from {label_encoder_path}")
                    else:
                        # Create label encoder from categories in metadata
                        from sklearn.preprocessing import LabelEncoder
                        all_categories = [data.get('semantic_category') for data in METADATA.values() if data.get('semantic_category')]
                        unique_categories = sorted(list(set(all_categories)))
                        _category_label_encoder = LabelEncoder()
                        _category_label_encoder.fit(unique_categories)
                        print(f"Created label encoder with {len(unique_categories)} categories")
                else:
                    print(f"Warning: Category classifier not found")
                    print("You may need to train it first using: python src/train_fclip_classifier.py")
        except Exception as e:
            print(f"Error loading category classifier: {e}")
            import traceback
            traceback.print_exc()
    return _category_classifier, _category_label_encoder

def predict_category_for_uploaded_image(uploaded_embedding, model, extractor):
    """
    Predict category for uploaded image using trained FashionCLIP classifier.
    Returns the predicted category.
    """
    print(f"[CATEGORY_PREDICTION] Starting category prediction using classifier...")
    
    classifier_dict, label_encoder = load_category_classifier()
    
    if classifier_dict is None or label_encoder is None:
        print(f"[CATEGORY_PREDICTION] Classifier not available, falling back to distance-based method...")
        # Fallback to old method
        return predict_category_by_distance(uploaded_embedding, model, extractor)
    
    try:
        if classifier_dict.get('type') == 'fclip':
            # FashionCLIP classifier
            classifier_head = classifier_dict['classifier']
            device = classifier_dict['device']
            
            # Normalize embedding
            uploaded_embedding_norm = uploaded_embedding / np.linalg.norm(uploaded_embedding)
            
            # Convert to tensor
            embedding_tensor = torch.tensor(uploaded_embedding_norm, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                logits = classifier_head(embedding_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_label = torch.argmax(logits, dim=1).item()
            
            predicted_category = label_encoder.inverse_transform([predicted_label])[0]
            prob_values = probabilities.cpu().numpy()[0]
            category_probs = dict(zip(label_encoder.classes_, prob_values))
            
            # Sort by probability
            sorted_categories = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)
            
            print(f"[CATEGORY_PREDICTION] Predicted category: {predicted_category}")
            print(f"[CATEGORY_PREDICTION] Top 3 predictions:")
            for cat, prob in sorted_categories[:3]:
                print(f"  {cat}: {prob:.4f} ({prob*100:.2f}%)")
            
            return predicted_category
            
        else:
            # Sklearn classifier
            sklearn_classifier = classifier_dict['model']
            
            # Normalize embedding
            uploaded_embedding_norm = uploaded_embedding / np.linalg.norm(uploaded_embedding)
            
            # Reshape for classifier (1 sample, n features)
            embedding_array = uploaded_embedding_norm.reshape(1, -1)
            
            # Predict category
            predicted_label = sklearn_classifier.predict(embedding_array)[0]
            predicted_category = label_encoder.inverse_transform([predicted_label])[0]
            
            # Get prediction probabilities
            probabilities = sklearn_classifier.predict_proba(embedding_array)[0]
            category_probs = dict(zip(label_encoder.classes_, probabilities))
            
            # Sort by probability
            sorted_categories = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)
            
            print(f"[CATEGORY_PREDICTION] Predicted category: {predicted_category}")
            print(f"[CATEGORY_PREDICTION] Top 3 predictions:")
            for cat, prob in sorted_categories[:3]:
                print(f"  {cat}: {prob:.4f} ({prob*100:.2f}%)")
            
            return predicted_category
        
    except Exception as e:
        print(f"[CATEGORY_PREDICTION] Error using classifier: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to distance-based method
        return predict_category_by_distance(uploaded_embedding, model, extractor)

def predict_category_by_distance(uploaded_embedding, model, extractor):
    """
    Fallback method: Predict category by comparing with sample items from each category.
    Returns the category with the highest average similarity.
    """
    print(f"[CATEGORY_PREDICTION] Using distance-based prediction (fallback)...")
    
    category_scores = {}
    samples_per_category = 5  # Sample 5 items from each category to get average
    
    for cat, cat_items in CATEGORY_GROUPS.items():
        if len(cat_items) == 0:
            continue
            
        # Sample items from this category
        sample_items = random.sample(cat_items, min(samples_per_category, len(cat_items)))
        category_distances = []
        
        for sample_id in sample_items:
            try:
                if model.model_type == "Projection":
                    # Get sample item embedding
                    sample_emb = model.get_embedding(sample_id)
                    if sample_emb is None:
                        continue
                    
                    with torch.no_grad():
                        # Project both embeddings
                        uploaded_emb_tensor = torch.tensor(uploaded_embedding).to(model.device).unsqueeze(0)
                        uploaded_projected = model.model(uploaded_emb_tensor)
                        sample_projected = model.model(sample_emb)
                        # Calculate distance
                        dist = torch.nn.functional.pairwise_distance(uploaded_projected, sample_projected).item()
                else:
                    # ResNet model - use FashionCLIP embeddings
                    if extractor is not None:
                        sample_img_path = Path(config.IMAGES_DIR) / f"{sample_id}.jpg"
                        if sample_img_path.exists():
                            sample_pil_img = Image.open(sample_img_path).convert('RGB')
                            sample_fclip_emb = extractor.extract_image_embeddings([sample_pil_img])[0]
                            
                            # Compare using cosine distance
                            uploaded_emb_norm = uploaded_embedding / np.linalg.norm(uploaded_embedding)
                            sample_emb_norm = sample_fclip_emb / np.linalg.norm(sample_fclip_emb)
                            cosine_sim = np.dot(uploaded_emb_norm, sample_emb_norm)
                            dist = 1.0 - cosine_sim
                        else:
                            continue
                    else:
                        continue
                
                category_distances.append(dist)
            except Exception as e:
                print(f"[CATEGORY_PREDICTION] Error comparing with {sample_id}: {e}")
                continue
        
        if category_distances:
            avg_distance = np.mean(category_distances)
            avg_score = np.exp(-avg_distance) * 100
            category_scores[cat] = avg_score
            print(f"[CATEGORY_PREDICTION] Category {cat}: avg_distance={avg_distance:.4f}, avg_score={avg_score:.2f}%")
    
    if category_scores:
        # Find category with highest score
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        print(f"[CATEGORY_PREDICTION] Best category: {best_category} (score: {best_score:.2f}%)")
        return best_category
    else:
        print(f"[CATEGORY_PREDICTION] Could not predict category, using 'unknown'")
        return "unknown"

# ----------------------------------------------------------------------
# Data Loading (Metadata for candidates)
# ----------------------------------------------------------------------
METADATA = {}
try:
    with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
        METADATA = json.load(f)
    print(f"Loaded {len(METADATA)} items from metadata.")
except Exception as e:
    print(f"Error loading metadata: {e}")

# Group items by category for fast retrieval
CATEGORY_GROUPS = {}
ID_TO_GROUP = {}

def index_items_by_category():
    global CATEGORY_GROUPS, ID_TO_GROUP
    count = 0
    for item_id, data in METADATA.items():
        cat = data.get('semantic_category') # Use semantic_category for ResNet model mapping
        if cat:
            if cat not in CATEGORY_GROUPS:
                CATEGORY_GROUPS[cat] = []
            CATEGORY_GROUPS[cat].append(item_id)
            ID_TO_GROUP[item_id] = cat
            count += 1
    print(f"Indexed {count} items into {len(CATEGORY_GROUPS)} categories.")

index_items_by_category()

# ----------------------------------------------------------------------
# 3 Base Category System (Tops, Bottoms, Shoes)
# ----------------------------------------------------------------------

# Base categories that MUST be included in recommendations
BASE_CATEGORIES = ["tops", "bottoms", "shoes"]

# Mapping from detailed categories to base categories
CATEGORY_TO_BASE = {
    # Tops
    "tops": "tops",
    "shirts": "tops",
    "blouses": "tops",
    "t-shirts": "tops",
    "sweaters": "tops",
    "jackets": "tops",
    "outerwear": "tops",
    
    # Bottoms
    "bottoms": "bottoms",
    "pants": "bottoms",
    "jeans": "bottoms",
    "shorts": "bottoms",
    "skirts": "bottoms",
    "trousers": "bottoms",
    
    # Shoes
    "shoes": "shoes",
    "sneakers": "shoes",
    "boots": "shoes",
    "heels": "shoes",
    "sandals": "shoes",
    
    # All body (counts as both tops and bottoms)
    "all_body": "both",  # Special case
    "dresses": "both",
    "jumpsuits": "both",
    "rompers": "both",
}

def get_base_category(category: str) -> List[str]:
    """
    Map a category to base categories (tops, bottoms, shoes).
    Returns list of base categories this item belongs to.
    """
    if not category or category == "unknown":
        return []
    
    category_lower = category.lower()
    
    # Check direct mapping
    if category_lower in CATEGORY_TO_BASE:
        base = CATEGORY_TO_BASE[category_lower]
        if base == "both":
            return ["tops", "bottoms"]
        return [base]
    
    # Check if category contains keywords
    if any(keyword in category_lower for keyword in ["top", "shirt", "blouse", "sweater", "jacket", "outer"]):
        return ["tops"]
    if any(keyword in category_lower for keyword in ["bottom", "pant", "jean", "short", "skirt", "trouser"]):
        return ["bottoms"]
    if any(keyword in category_lower for keyword in ["shoe", "sneaker", "boot", "heel", "sandal"]):
        return ["shoes"]
    if any(keyword in category_lower for keyword in ["dress", "jumpsuit", "romper", "all_body"]):
        return ["tops", "bottoms"]
    
    # Default: return empty (not a base category)
    return []

def get_items_for_base_category(base_cat: str) -> List[str]:
    """
    Get all item IDs that belong to a base category.
    """
    items = []
    for category, item_ids in CATEGORY_GROUPS.items():
        base_cats = get_base_category(category)
        if base_cat in base_cats:
            items.extend(item_ids)
    return items

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

class OutfitCompletionRequest(BaseModel):
    item_id: str

class RecommendationItem(BaseModel):
    id: str
    image_url: str
    name: str
    price: float
    match_score: int
    category: str

class OutfitCompletionResponse(BaseModel):
    recommendations: List[RecommendationItem]

@router.post("/complete")
async def complete_outfit(request: OutfitCompletionRequest):
    """
    Given an item ID, recommends compatible items from different categories
    using the ResNet-based TypeAwareRecommender.
    """
    item_id = str(request.item_id)  # Ensure string type
    
    print(f"[OUTFIT/COMPLETE] Received request for item_id: {item_id} (type: {type(item_id)})")
    print(f"[OUTFIT/COMPLETE] Metadata keys count: {len(METADATA)}")
    print(f"[OUTFIT/COMPLETE] Item in metadata: {item_id in METADATA}")
    
    # Check if this is an uploaded image (UUID format or in user_uploads)
    is_uploaded_image = False
    uploaded_image_path = None
    uploaded_embedding = None
    extractor = None  # Will be set if uploaded image
    
    if item_id not in METADATA:
        # Try with different string formats
        item_id_int = None
        try:
            item_id_int = int(item_id)
            if str(item_id_int) in METADATA:
                item_id = str(item_id_int)
                print(f"[OUTFIT/COMPLETE] Found item with int conversion: {item_id}")
            elif item_id_int in METADATA:
                item_id = item_id_int
                print(f"[OUTFIT/COMPLETE] Found item with int key: {item_id}")
        except:
            pass
        
        # Check if it's an uploaded image
        if item_id not in METADATA and (item_id_int is None or item_id_int not in METADATA):
            uploaded_image_path = UPLOAD_DIR / f"{item_id}.png"
            if uploaded_image_path.exists():
                is_uploaded_image = True
                print(f"[OUTFIT/COMPLETE] Item is an uploaded image: {uploaded_image_path}")
                
                # Get embedding from cache or extract
                import backend.routers.wardrobe as wardrobe_router
                wardrobe_router.load_uploaded_embeddings()  # Reload cache
                
                cached_embedding = wardrobe_router.get_uploaded_embedding(item_id)
                if cached_embedding is not None:
                    uploaded_embedding = cached_embedding
                    print(f"[OUTFIT/COMPLETE] Using cached embedding for uploaded image (shape: {uploaded_embedding.shape})")
                else:
                    extractor = get_embedding_extractor()
                    if extractor is None:
                        raise HTTPException(status_code=503, detail="Embedding extractor not available for uploaded images")
                    
                    uploaded_image = Image.open(uploaded_image_path).convert('RGB')
                    uploaded_embedding = extractor.extract_image_embeddings([uploaded_image])[0]
                    # Cache it for future use
                    wardrobe_router.cache_uploaded_embedding(item_id, uploaded_embedding)
                    print(f"[OUTFIT/COMPLETE] Extracted and cached embedding for uploaded image (shape: {uploaded_embedding.shape})")
            else:
                print(f"[OUTFIT/COMPLETE] ERROR: Item {item_id} not found in metadata or uploads")
                raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    model = get_recommender()
    if model is None:
        # Fallback if model fails to load
        print("Model not loaded, returning empty recommendations")
        return {"recommendations": []}
        
    # Get input item category
    if is_uploaded_image:
        # For uploaded images, first check if user has confirmed a category
        # Lazy import to avoid circular dependency
        import backend.routers.wardrobe as wardrobe_router
        wardrobe_router.load_uploaded_categories()  # Reload to get latest
        
        if item_id in wardrobe_router.UPLOADED_CATEGORIES:
            input_cat = wardrobe_router.UPLOADED_CATEGORIES[item_id]
            print(f"[OUTFIT/COMPLETE] Using user-confirmed category: {input_cat}")
        else:
            # If not confirmed, predict category
            print(f"[OUTFIT/COMPLETE] Category not confirmed, predicting category for uploaded image...")
            input_cat = predict_category_for_uploaded_image(uploaded_embedding, model, extractor)
            print(f"[OUTFIT/COMPLETE] Predicted category: {input_cat}")
            print(f"[OUTFIT/COMPLETE] WARNING: Category not confirmed by user. Consider calling /wardrobe/update_category first.")
    else:
        input_cat = ID_TO_GROUP.get(item_id)
        if not input_cat:
            # Fallback if not indexed but in metadata
            input_cat = METADATA[item_id].get('semantic_category')
        
        if not input_cat:
             # If still no category, we can't use TypeAware model effectively
             # But let's try to guess or just pick random other categories
             print(f"Warning: No category for {item_id}")
             input_cat = "unknown"

    print(f"\n{'='*80}")
    print(f"[OUTFIT/COMPLETE] Request for item: {item_id}")
    print(f"[OUTFIT/COMPLETE] Item category: {input_cat}")
    print(f"{'='*80}")

    recommendations = []
    
    # For uploaded images, project embedding once and reuse it
    uploaded_projected = None
    if is_uploaded_image and uploaded_embedding is not None and model.model_type == "Projection":
        with torch.no_grad():
            uploaded_emb_tensor = torch.tensor(uploaded_embedding).to(model.device).unsqueeze(0)
            uploaded_projected = model.model(uploaded_emb_tensor)  # Shape: (1, dim)
        print(f"[OUTFIT/COMPLETE] Uploaded embedding projected (shape: {uploaded_projected.shape})")
    
    # Get base categories for input item
    input_base_cats = get_base_category(input_cat)
    print(f"[OUTFIT/COMPLETE] Input item base categories: {input_base_cats}")
    
    # Determine which base categories we need to recommend
    # We need to recommend the 3 base categories (tops, bottoms, shoes)
    # But exclude the ones the input item already covers
    required_base_cats = []
    for base_cat in BASE_CATEGORIES:
        if base_cat not in input_base_cats:
            required_base_cats.append(base_cat)
    
    print(f"[OUTFIT/COMPLETE] Required base categories to recommend: {required_base_cats}")
    
    # Threshold for other (non-base) categories
    OTHER_CATEGORY_THRESHOLD = 0.3  # Minimum similarity score (0-1) to include other categories
    
    # Step 1: Get recommendations for required base categories (MUST HAVE)
    base_category_recommendations = {}
    for base_cat in required_base_cats:
        print(f"\n{'='*80}")
        print(f"[OUTFIT/COMPLETE] Processing REQUIRED base category: {base_cat}")
        print(f"{'='*80}")
        
        # Get all items in this base category
        base_cat_items = get_items_for_base_category(base_cat)
        if not base_cat_items:
            print(f"[OUTFIT/COMPLETE] No items found for base category: {base_cat}")
            continue
        
        # Sample candidates (more samples for better results)
        sample_size = min(20, len(base_cat_items))
        sample_candidates = random.sample(base_cat_items, sample_size)
        
        print(f"[OUTFIT/COMPLETE] Sampling {sample_size} candidates from {len(base_cat_items)} items")
        
        scored_candidates = []
        
        if is_uploaded_image and uploaded_embedding is not None and uploaded_projected is not None:
            # Uploaded image: Use pre-projected embedding (already computed above)
            # Get all candidate embeddings at once
            candidate_embeddings = []
            valid_candidate_ids = []
            for cand_id in sample_candidates:
                cand_emb = model.get_embedding(cand_id)
                if cand_emb is not None:
                    candidate_embeddings.append(cand_emb)
                    valid_candidate_ids.append(cand_id)
            
            if candidate_embeddings:
                # Batch project all candidates
                candidate_emb_tensor = torch.cat(candidate_embeddings, dim=0)  # Shape: (N, dim)
                with torch.no_grad():
                    candidate_projected = model.model(candidate_emb_tensor)  # Shape: (N, dim)
                
                # Batch calculate distances
                distances = torch.nn.functional.pairwise_distance(
                    uploaded_projected.expand(len(candidate_embeddings), -1),
                    candidate_projected
                ).cpu().numpy()
                
                # Convert to scores
                scores = np.exp(-distances) * 100
                
                # Create scored candidates
                for cand_id, dist, score in zip(valid_candidate_ids, distances, scores):
                    scored_candidates.append({
                        "item_id": cand_id,
                        "score": float(score),
                        "distance": float(dist)
                    })
            else:
                print(f"[WARNING] No embeddings found for any candidate in {base_cat}")
        elif is_uploaded_image and model.model_type != "Projection":
            # ResNet model - use predict_compatibility
            for cand_id in sample_candidates:
                try:
                    dist = model.predict_compatibility(item_id, cand_id)
                    score = model.predict_similarity_score(item_id, cand_id)
                    scored_candidates.append({
                        "item_id": cand_id,
                        "score": score,
                        "distance": dist
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to score candidate {cand_id}: {e}")
                    continue
        else:
            # Regular metadata item - use fast path
            for cand_id in sample_candidates:
                try:
                    dist = model.predict_compatibility(item_id, cand_id)
                    score = model.predict_similarity_score(item_id, cand_id)
                    scored_candidates.append({
                        "item_id": cand_id,
                        "score": score,
                        "distance": dist
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to score candidate {cand_id}: {e}")
                    continue
        
        # Sort and take best
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        if scored_candidates:
            best = scored_candidates[0]
            base_category_recommendations[base_cat] = best
            print(f"[OUTFIT/COMPLETE] Best for {base_cat}: {best['item_id']} (score: {best['score']:.2f})")
    
    # Step 2: Get recommendations for other categories (OPTIONAL, threshold-based)
    all_cats = list(CATEGORY_GROUPS.keys())
    other_cats = [c for c in all_cats if c != input_cat and get_base_category(c) == []]  # Non-base categories
    
    print(f"\n{'='*80}")
    print(f"[OUTFIT/COMPLETE] Processing OTHER categories (threshold: {OTHER_CATEGORY_THRESHOLD})")
    print(f"[OUTFIT/COMPLETE] Found {len(other_cats)} other categories")
    print(f"{'='*80}")
    
    other_recommendations = []
    for cat in other_cats:
        candidates = CATEGORY_GROUPS[cat]
        total_candidates_in_category = len(candidates)
        # Pick random candidates to score (e.g., 10 per category)
        sample_size = min(10, len(candidates))
        sample_candidates = random.sample(candidates, sample_size)
        
        print(f"\n{'='*80}")
        print(f"[OUTFIT/COMPLETE] Category: {cat}")
        print(f"[OUTFIT/COMPLETE] Total candidates in category: {total_candidates_in_category}")
        print(f"[OUTFIT/COMPLETE] Sampling {sample_size} candidates for scoring")
        print(f"[OUTFIT/COMPLETE] Anchor item: {item_id} (category: {input_cat})")
        print(f"{'='*80}")
        
        scored_candidates = []
        
        if is_uploaded_image and uploaded_embedding is not None and uploaded_projected is not None:
            # Uploaded image: Use pre-projected embedding (already computed above)
            # Get all candidate embeddings at once
            candidate_embeddings = []
            valid_candidate_ids = []
            for cand_id in sample_candidates:
                cand_emb = model.get_embedding(cand_id)
                if cand_emb is not None:
                    candidate_embeddings.append(cand_emb)
                    valid_candidate_ids.append(cand_id)
            
            if candidate_embeddings:
                # Batch project all candidates
                candidate_emb_tensor = torch.cat(candidate_embeddings, dim=0)  # Shape: (N, dim)
                with torch.no_grad():
                    candidate_projected = model.model(candidate_emb_tensor)  # Shape: (N, dim)
                
                # Batch calculate distances
                distances = torch.nn.functional.pairwise_distance(
                    uploaded_projected.expand(len(candidate_embeddings), -1),
                    candidate_projected
                ).cpu().numpy()
                
                # Convert to scores
                scores = np.exp(-distances) * 100
                
                # Create scored candidates
                for idx, (cand_id, dist, score) in enumerate(zip(valid_candidate_ids, distances, scores), 1):
                    print(f"[CANDIDATE {idx:2d}/{len(valid_candidate_ids)}] ID: {cand_id:12s} | Distance: {dist:8.4f} | Score: {score:6.2f}%")
                    scored_candidates.append({
                        "item_id": cand_id,
                        "score": float(score),
                        "distance": float(dist),
                        "category": cat
                    })
            else:
                print(f"[WARNING] No embeddings found for any candidate in {cat}")
        elif is_uploaded_image and model.model_type != "Projection":
            # ResNet model - use predict_compatibility
            for idx, cand_id in enumerate(sample_candidates, 1):
                try:
                    dist = model.predict_compatibility(item_id, cand_id)
                    score = model.predict_similarity_score(item_id, cand_id)
                    print(f"[CANDIDATE {idx:2d}/{sample_size}] ID: {cand_id:12s} | Distance: {dist:8.4f} | Score: {score:6.2f}%")
                    scored_candidates.append({
                        "item_id": cand_id,
                        "score": score,
                        "distance": dist,
                        "category": cat
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to score candidate {cand_id}: {e}")
                    continue
        else:
            # Regular metadata item - use fast path
            for idx, cand_id in enumerate(sample_candidates, 1):
                try:
                    dist = model.predict_compatibility(item_id, cand_id)
                    score = model.predict_similarity_score(item_id, cand_id)
                    print(f"[CANDIDATE {idx:2d}/{sample_size}] ID: {cand_id:12s} | Distance: {dist:8.4f} | Score: {score:6.2f}%")
                    scored_candidates.append({
                        "item_id": cand_id,
                        "score": score,
                        "distance": dist,
                        "category": cat
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to score candidate {cand_id}: {e}")
                    continue
        
        # Sort by score descending (Higher is better)
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Log all candidates sorted
        print(f"\n[SCORED CANDIDATES] Sorted by score (best to worst):")
        for idx, cand in enumerate(scored_candidates, 1):
            print(f"  [{idx:2d}] ID: {cand['item_id']:12s} | Distance: {cand['distance']:8.4f} | Score: {cand['score']:6.2f}%")
        
        # Take top 1 from this category, but only if it meets threshold
        if scored_candidates:
            best = scored_candidates[0]
            # Convert score to 0-1 range for threshold comparison
            score_normalized = best['score'] / 100.0
            
            if score_normalized >= OTHER_CATEGORY_THRESHOLD:
                print(f"\n[SELECTED] Best candidate: {best['item_id']} | Distance: {best['distance']:.4f} | Score: {best['score']:.2f}% (meets threshold)")
                print(f"{'='*80}\n")
                
                # Format for response
                meta = METADATA.get(best['item_id'], {})
                
                # Construct image URL
                image_url = f"{config.BASE_URL}/images/{best['item_id']}.jpg"
                
                other_recommendations.append({
                    "id": str(best['item_id']),
                    "image_url": str(image_url),
                    "name": str(meta.get('url_name', meta.get('title', 'Unknown Item'))),
                    "price": float(meta.get('price', 0)),
                    "match_score": int(best['score']),
                    "category": str(best['category'])
                })
            else:
                print(f"\n[SKIPPED] Best candidate score {best['score']:.2f}% below threshold {OTHER_CATEGORY_THRESHOLD*100:.0f}%")
                print(f"{'='*80}\n")
    
    # Step 3: Combine base category recommendations (MUST HAVE) with other recommendations (OPTIONAL)
    print(f"\n{'='*80}")
    print(f"[OUTFIT/COMPLETE] Combining recommendations")
    print(f"{'='*80}")
    
    # Add base category recommendations (always included)
    for base_cat, best in base_category_recommendations.items():
        meta = METADATA.get(best['item_id'], {})
        image_url = f"{config.BASE_URL}/images/{best['item_id']}.jpg"
        
        recommendations.append({
            "id": str(best['item_id']),
            "image_url": str(image_url),
            "name": str(meta.get('url_name', meta.get('title', 'Unknown Item'))),
            "price": float(meta.get('price', 0)),
            "match_score": int(best['score']),
            "category": str(meta.get('semantic_category', 'unknown')),
            "base_category": base_cat,
            "required": True  # Mark as required (from base categories)
        })
        print(f"[REQUIRED] {base_cat}: {best['item_id']} (score: {best['score']:.2f}%)")
    
    # Add other category recommendations (threshold-based)
    recommendations.extend(other_recommendations)
    for rec in other_recommendations:
        print(f"[OPTIONAL] {rec['category']}: {rec['id']} (score: {rec['match_score']}%)")
    
    print(f"\n[OUTFIT/COMPLETE] Final recommendations: {len(recommendations)} items ({len(base_category_recommendations)} required, {len(other_recommendations)} optional)")
    for idx, rec in enumerate(recommendations, 1):
        print(f"  [{idx}] {rec['id']} ({rec['category']}) - Score: {rec['match_score']}%")
    print(f"{'='*80}\n")
            
    return {"recommendations": recommendations}


from backend.database import get_db_connection

# ----------------------------------------------------------------------
# Helper: Visual Ordering & Thresholds
# ----------------------------------------------------------------------
def get_visual_order(category: str) -> int:
    """
    Returns a visual order index for the mannequin layout.
    1: Head/Hats
    2: Tops/Outerwear/Dresses
    3: Bottoms
    4: Shoes
    5: Bags/Accessories
    """
    cat = category.lower()
    if 'hat' in cat: return 1
    if 'top' in cat or 'outerwear' in cat or 'dress' in cat: return 2
    if 'bottom' in cat: return 3
    if 'shoe' in cat: return 4
    return 5

def get_target_categories(anchor_cat: str) -> List[str]:
    """
    Returns a list of target categories based on the anchor category.
    """
    anchor_cat = anchor_cat.lower()
    targets = []
    
    if 'top' in anchor_cat or 'outerwear' in anchor_cat:
        targets = ['bottoms', 'shoes', 'bags', 'hats']
    elif 'bottom' in anchor_cat:
        targets = ['tops', 'shoes', 'bags', 'hats']
    elif 'shoe' in anchor_cat:
        targets = ['tops', 'bottoms', 'bags']
    elif 'dress' in anchor_cat:
        targets = ['shoes', 'bags', 'jewellery']
    else:
        # Fallback
        targets = ['tops', 'bottoms', 'shoes']
        
    return targets

def generate_outfit_from_anchor(anchor_id: str, model) -> Optional[dict]:
    """
    Generates a complete outfit dictionary starting from an anchor item.
    """
    if anchor_id not in METADATA: return None
    
    anchor_cat = ID_TO_GROUP.get(anchor_id)
    if not anchor_cat: return None
    
    meta_anchor = METADATA.get(anchor_id, {})
    
    combo_items = []
    # Add Anchor
    combo_items.append({
        "id": anchor_id,
        "image_url": f"{config.BASE_URL}/images/{anchor_id}.jpg",
        "name": meta_anchor.get('url_name', 'Unknown Item'),
        "price": meta_anchor.get('price', 0),
        "category": anchor_cat,
        "order": get_visual_order(anchor_cat)
    })
    
    target_types = get_target_categories(anchor_cat)
    
    current_score_sum = 0
    count = 0
    
    print(f"\n{'='*80}")
    print(f"[OUTFIT/CREATE] Generating outfit from anchor: {anchor_id}")
    print(f"[OUTFIT/CREATE] Anchor category: {anchor_cat}")
    print(f"[OUTFIT/CREATE] Target types: {target_types}")
    print(f"{'='*80}")
    
    for target_type_keyword in target_types:
        # Find matching categories
        matching_cats = [c for c in CATEGORY_GROUPS.keys() if target_type_keyword in c]
        if not matching_cats: 
             matching_cats = [c for c in CATEGORY_GROUPS.keys() if target_type_keyword[:-1] in c]
        
        if not matching_cats:
            print(f"[SKIP] No matching categories found for '{target_type_keyword}'")
            continue
        
        target_cat = random.choice(matching_cats)
        candidates = CATEGORY_GROUPS[target_cat]
        total_candidates = len(candidates)
        sample_size = min(10, len(candidates))
        sample_candidates = random.sample(candidates, sample_size)
        
        print(f"\n[CATEGORY] {target_cat} (target: {target_type_keyword})")
        print(f"  Total candidates: {total_candidates}, Sampling: {sample_size}")
        
        best_cand = None
        best_score = -1
        all_scores = []
        
        for idx, cand_id in enumerate(sample_candidates, 1):
            if model:
                score = model.predict_similarity_score(anchor_id, cand_id)
                dist = model.predict_compatibility(anchor_id, cand_id)
            else:
                score = random.randint(50, 90)
                dist = 0.0
            
            all_scores.append((cand_id, score, dist))
            print(f"  [CAND {idx:2d}/{sample_size}] ID: {cand_id:12s} | Distance: {dist:8.4f} | Score: {score:6.2f}%")
                
            if score > best_score:
                best_score = score
                best_cand = cand_id
        
        # Log score distribution
        if all_scores:
            scores_only = [s[1] for s in all_scores]
            print(f"  [STATS] Score range: {min(scores_only):.2f}% - {max(scores_only):.2f}% | Avg: {sum(scores_only)/len(scores_only):.2f}%")
        
        # Threshold Logic
        # Core items (Top, Bottom, Shoe) -> Always add best
        # Accessories -> Add only if score > 70 (approx)
        is_core = target_type_keyword in ['tops', 'bottoms', 'shoes']
        threshold = 60 if not is_core else 0
        
        if best_cand:
            print(f"  [BEST] Selected: {best_cand} | Score: {best_score:.2f}% | Threshold: {threshold}% | Is Core: {is_core}")
            if is_core or best_score > 60: # Threshold
                current_score_sum += best_score
                count += 1
                print(f"  [ADDED] Item {best_cand} added to outfit (Score: {best_score:.2f}%)")
                
                meta = METADATA.get(best_cand, {})
                combo_items.append({
                    "id": best_cand,
                    "image_url": f"{config.BASE_URL}/images/{best_cand}.jpg",
                    "name": meta.get('url_name', 'Unknown Item'),
                    "price": meta.get('price', 0),
                    "category": target_cat,
                    "order": get_visual_order(target_cat)
                })
            else:
                print(f"  [REJECTED] Score {best_score:.2f}% below threshold {threshold}%")
        else:
            print(f"  [NO CANDIDATE] No valid candidate found")
    
    if count > 0:
        avg_score = current_score_sum / count
        # Sort items by visual order
        combo_items.sort(key=lambda x: x['order'])
        
        print(f"\n[OUTFIT SUMMARY]")
        print(f"  Total items: {len(combo_items)}")
        print(f"  Average match score: {avg_score:.2f}%")
        print(f"  Items: {[item['id'] for item in combo_items]}")
        print(f"{'='*80}\n")
        
        return {
            "id": f"combo_{random.randint(10000,99999)}",
            "match_score": int(avg_score),
            "items": combo_items
        }
    print(f"[OUTFIT FAILED] No items added to outfit\n")
    return None


@router.get("/combinations")
async def get_combinations(user_id: str = "test_user"):
    """
    Generates smart outfit combinations based on user's liked items.
    """
    print(f"\n{'='*80}")
    print(f"[OUTFIT/COMBINATIONS] Request for user: {user_id}")
    print(f"{'='*80}")
    
    # 1. Fetch Liked Items from DB
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT item_id FROM interactions WHERE user_id = ? AND action IN ('like', 'superlike')",
            (user_id,)
        )
        liked_ids = [row[0] for row in cursor.fetchall()]
        print(f"[OUTFIT/COMBINATIONS] Found {len(liked_ids)} liked items from DB")
    except Exception as e:
        print(f"[OUTFIT/COMBINATIONS] Error fetching likes: {e}")
        liked_ids = []
    finally:
        conn.close()

    if not liked_ids:
        # Fallback for demo
        print("[OUTFIT/COMBINATIONS] No likes found, using random items for demo.")
        all_ids = list(METADATA.keys())
        if all_ids:
            liked_ids = random.sample(all_ids, min(5, len(all_ids)))
            print(f"[OUTFIT/COMBINATIONS] Using {len(liked_ids)} random items as fallback")
        else:
            print("[OUTFIT/COMBINATIONS] No items available in metadata")
            return {"combinations": []}
        
    model = get_recommender()
    combinations = []
    
    # Filter anchors: only tops, bottoms, shoes
    valid_anchors = []
    for lid in liked_ids:
        cat = ID_TO_GROUP.get(lid, "")
        if 'top' in cat or 'bottom' in cat or 'shoe' in cat or 'dress' in cat:
            valid_anchors.append(lid)
    
    print(f"[OUTFIT/COMBINATIONS] Valid anchors (tops/bottoms/shoes/dresses): {len(valid_anchors)}")
            
    if not valid_anchors:
        valid_anchors = liked_ids # Fallback if no specific anchors found
        print(f"[OUTFIT/COMBINATIONS] No specific anchors, using all liked items: {len(valid_anchors)}")
    
    # Generate 5 combinations
    attempts = 0
    while len(combinations) < 5 and attempts < 10:
        attempts += 1
        anchor_id = random.choice(valid_anchors)
        print(f"\n[OUTFIT/COMBINATIONS] Attempt {attempts}/10 - Using anchor: {anchor_id}")
        
        combo = generate_outfit_from_anchor(anchor_id, model)
        if combo:
            combinations.append(combo)
            print(f"[OUTFIT/COMBINATIONS] Successfully generated combination {len(combinations)}/5")
        else:
            print(f"[OUTFIT/COMBINATIONS] Failed to generate combination")
            
    # Sort by score
    combinations.sort(key=lambda x: x['match_score'], reverse=True)
    
    print(f"\n[OUTFIT/COMBINATIONS] Final result: {len(combinations)} combinations")
    for idx, combo in enumerate(combinations, 1):
        print(f"  [{idx}] Combo {combo['id']} - Score: {combo['match_score']}% - Items: {len(combo['items'])}")
    print(f"{'='*80}\n")
    
    return {"combinations": combinations}

class CreateOutfitRequest(BaseModel):
    anchor_item_id: str

@router.post("/create_from_anchor")
async def create_outfit_from_anchor(request: CreateOutfitRequest):
    """
    Creates a single outfit based on a specific anchor item.
    """
    model = get_recommender()
    combo = generate_outfit_from_anchor(request.anchor_item_id, model)
    
    if combo:
        return combo
    else:
        raise HTTPException(status_code=400, detail="Could not generate outfit for this item")
