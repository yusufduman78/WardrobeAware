import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Recommender Imports
try:
    from recommender import config
    from recommender.model import TypeAwareNet
except ImportError:
    import config
    from model import TypeAwareNet

# FashionCLIP Imports
from src import fclip_config as fconfig
from src.fclip_metric_learning import EmbeddingProjection
from utils import fclip_utils as utils

class TypeAwareRecommender:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Recommender running on: {self.device}")
        
        # Config'den varsayılan yolları al
        if model_path is None:
            model_path = "model_epoch_15.pth" 
            
        self.model_path = model_path
        self.transform = self._get_transforms()
        self.model_type = "ResNet" # Default
        self.embeddings_map = None
        
        self.model, self.cat_to_id = self._load_model_and_metadata()
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        
        # Metadata'yı (semantic_category için) yükle
        with open(config.METADATA_PATH, 'r') as f:
            self.metadata = json.load(f)

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model_and_metadata(self):
        print(f"Loading model from {self.model_path}...")
        if not os.path.exists(self.model_path):
            # Try absolute path or relative to project root
            alt_paths = [
                PROJECT_ROOT / self.model_path,
                PROJECT_ROOT / "models" / self.model_path,
                Path(self.model_path),
                Path("models") / self.model_path
            ]
            
            found = False
            for alt_path in alt_paths:
                if alt_path.exists():
                    self.model_path = str(alt_path)
                    print(f"Found model at: {self.model_path}")
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(f"Model checkpoint not found. Tried: {self.model_path} and alternative paths")
            
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # Detect Model Type
        keys = list(state_dict.keys())
        if any(k.startswith('net.') for k in keys) or any(k.startswith('linear1') for k in keys) or "projection" in str(self.model_path):
            print("Detected EmbeddingProjection Model (MLP). Switching mode.")
            self.model_type = "Projection"
            
            # Load Embeddings
            print("Loading FashionCLIP embeddings for Projection model...")
            if fconfig.IMAGE_EMBEDDINGS_PATH.exists():
                embs = np.load(fconfig.IMAGE_EMBEDDINGS_PATH)
                ids = utils.load_json(fconfig.ITEM_IDS_PATH)
                self.embeddings_map = {str(k): v for k, v in zip(ids, embs)}
                print(f"Loaded {len(self.embeddings_map)} embeddings.")
            else:
                print("WARNING: Embeddings not found! Recommendations will fail.")
                self.embeddings_map = {}

            # Init Model
            model = EmbeddingProjection(input_dim=512, output_dim=512)
            try:
                model.load_state_dict(state_dict, strict=False)
            except:
                pass
            
            # Helper for category (not used for projection but good to have)
            cat_to_id = {} 

        else:
            # Standard ResNet
            print("Detected TypeAwareNet Model (ResNet).")
            cat_to_id = checkpoint.get('cat_to_id', {})
            num_categories = len(cat_to_id) if cat_to_id else 11
            embedding_dim = checkpoint.get('args', {}).get('embedding_dim', 64)
            
            model = TypeAwareNet(embedding_dim=embedding_dim, num_categories=num_categories, pretrained=False)
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        
        return model, cat_to_id

    def get_item_category_id(self, item_id):
        if item_id not in self.metadata:
            return None
        
        cat_str = self.metadata[item_id].get('semantic_category')
        if cat_str and cat_str in self.cat_to_id:
            return self.cat_to_id[cat_str]
        return None

    def _load_image_from_path(self, path):
        if not os.path.exists(path):
            img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))
        else:
            img = Image.open(path).convert('RGB')
        return self.transform(img).unsqueeze(0).to(self.device)

    def load_image(self, item_id):
        # Legacy: used for ResNet
        filename = f"{item_id}.jpg"
        path = os.path.join(config.IMAGES_ROOT, filename)
        return self._load_image_from_path(path)
        
    def get_embedding(self, item_id):
        # For Projection Model
        if self.embeddings_map and str(item_id) in self.embeddings_map:
            return torch.tensor(self.embeddings_map[str(item_id)]).to(self.device).unsqueeze(0)
        
        # Try finding cleaned ID
        clean = str(item_id).split('_')[0]
        if self.embeddings_map and clean in self.embeddings_map:
            return torch.tensor(self.embeddings_map[clean]).to(self.device).unsqueeze(0)
            
        return None

    def predict_compatibility(self, item_id_1, item_id_2, category_1=None, image_path_1=None, category_2=None, image_path_2=None):
        """
        Calculates compatibility distance. Lower is better.
        Optionally accepts category and image path overrides for items not in metadata.
        """
        if self.model_type == "Projection":
            emb1 = self.get_embedding(item_id_1)
            emb2 = self.get_embedding(item_id_2)
            
            # If embedding not found via ID, we can't easily compute it here for Projection model 
            # without the encoder. The caller should have handled this or we fail.
            if emb1 is None or emb2 is None:
                return 10.0 # High distance (unknown)
                
            with torch.no_grad():
                p1 = self.model(emb1)
                p2 = self.model(emb2)
                dist = F.pairwise_distance(p1, p2).item()
            return dist
            
        else:
            # ResNet Logic
            # Resolve Category 1
            if category_1:
                cat_id_1 = self.cat_to_id.get(category_1)
            else:
                cat_id_1 = self.get_item_category_id(item_id_1)
                
            # Resolve Category 2
            if category_2:
                cat_id_2 = self.cat_to_id.get(category_2)
            else:
                cat_id_2 = self.get_item_category_id(item_id_2)
            
            if cat_id_1 is None or cat_id_2 is None:
                return 10.0 # High distance
                
            # Resolve Image 1
            if image_path_1:
                img_1 = self._load_image_from_path(image_path_1)
            else:
                img_1 = self.load_image(item_id_1)
                
            # Resolve Image 2
            if image_path_2:
                img_2 = self._load_image_from_path(image_path_2)
            else:
                img_2 = self.load_image(item_id_2)
            
            with torch.no_grad():
                emb_1 = self.model.get_general_embedding(img_1)
                emb_2 = self.model.get_general_embedding(img_2)
                
                # Masks (if using learned masks)
                mask_12 = self.model.masks[cat_id_1, cat_id_2]
                
                # Using raw embeddings directly as they are well-distributed
                proj_1 = emb_1 # * mask_12
                proj_2 = emb_2 # * mask_12
                
                dist = F.pairwise_distance(proj_1, proj_2, p=2).item()
                return dist

    def predict_similarity_score(self, item_id_1, item_id_2, **kwargs):
        """
        Distance -> Score (0-100)
        Passes **kwargs (category_1, image_path_1, etc) to predict_compatibility
        """
        dist = self.predict_compatibility(item_id_1, item_id_2, **kwargs)
        
        # For projection model, distances might be smaller (e.g. 0.0-2.0)
        # ResNet distances were also similar.
        # Simple exponential decay map
        score = np.exp(-dist) * 100
        return score

    def predict_batch_compatibility(self, anchor_id, candidate_ids, anchor_category=None, anchor_image_path=None):
        """
        Efficiently predicts structural compatibility for one anchor against many candidates.
        Uses batch processing to minimize disk I/O and maximized GPU throughput.
        Only supports ResNet/TypeAwareNet logic currently.
        """
        if self.model_type == "Projection":
            # Fallback to loop for Projection (can't batch easily without embeddings_map)
            results = []
            for cand_id in candidate_ids:
                dist = self.predict_compatibility(anchor_id, cand_id)
                score = self.predict_similarity_score(anchor_id, cand_id)
                results.append((cand_id, dist, score))
            return results

        # --- Batch Logic for ResNet ---
        
        # 1. Prepare Anchor (Once)
        if anchor_category:
            anchor_cat_id = self.cat_to_id.get(anchor_category)
        else:
            anchor_cat_id = self.get_item_category_id(anchor_id)
            
        if anchor_cat_id is None:
            print(f"[BATCH] Warning: Anchor category not found for {anchor_id}")
            return [(cand, 10.0, 0.0) for cand in candidate_ids]

        # Load Anchor Image
        if anchor_image_path:
            img_anchor = self._load_image_from_path(anchor_image_path)
        else:
            img_anchor = self.load_image(anchor_id)

        # Get Anchor Embedding
        with torch.no_grad():
             emb_anchor = self.model.get_general_embedding(img_anchor) # Shape: (1, dim)
        
        # 2. Prepare Candidates (Batch)
        results = []
        batch_images = []
        valid_indices = []
        valid_candidates = []
        
        # Determine candidate categories map for mask lookup
        cand_cat_ids = []

        # Load images into RAM (This is still I/O bound but faster than interleaved)
        for idx, cand_id in enumerate(candidate_ids):
            cand_cat_id = self.get_item_category_id(cand_id)
            if cand_cat_id is None:
                continue
                
            # Load candidate image
            filename = f"{cand_id}.jpg"
            path = os.path.join(config.IMAGES_ROOT, filename)
            
            if not os.path.exists(path):
                continue
                
            try:
                img_pil = Image.open(path).convert('RGB')
                img_tensor = self.transform(img_pil) # (3, 224, 224)
                batch_images.append(img_tensor)
                valid_candidates.append(cand_id)
                cand_cat_ids.append(cand_cat_id)
            except Exception as e:
                print(f"[BATCH] Error loading {cand_id}: {e}")
                continue

        if not batch_images:
            return []

        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images).to(self.device) # (B, 3, 224, 224)
        
        # 3. Batch Inference
        with torch.no_grad():
            emb_candidates = self.model.get_general_embedding(batch_tensor) # (B, dim)
            
            # Since masks are pairwise (cat1, cat2), and cat2 varies per candidate, 
            # we technically need to apply different masks. 
            # However, for speed, we can skip mask or apply per-item. 
            # Let's apply per-item but vectorized if possible, or simple loop on embeddings (fast)
            
            # Simple loop for distance calculation on embeddings (FAST in RAM)
            for i, cand_id in enumerate(valid_candidates):
                cand_cat_id = cand_cat_ids[i]
                cand_emb = emb_candidates[i] # (dim)
                
                # Apply Mask if exists
                # mask = self.model.masks[anchor_cat_id, cand_cat_id]
                # dist = pairwise between (emb_anchor * mask) and (cand_emb * mask)
                
                # Simplified: Just euclidean distance between raw embeddings
                dist = F.pairwise_distance(emb_anchor, cand_emb.unsqueeze(0), p=2).item()
                score = np.exp(-dist) * 100
                
                results.append((cand_id, dist, score))
                
        return results

if __name__ == "__main__":
    # Test bloğu
    import sys
    print("Testing TypeAwareRecommender...")
    # Change this to test specific model if needed
    try:
        recommender = TypeAwareRecommender(model_path="models/projection_model.pth")
    except:
        recommender = TypeAwareRecommender() # Default

    # Validasyon setinden rastgele bir çift bulup test edelim
    with open(config.VALID_JSON, 'r') as f:
        valid_outfits = json.load(f)
        
    if valid_outfits:
        outfit = valid_outfits[0]
        items = outfit['items']
        if len(items) >= 2:
            id1 = items[0]['item_id']
            id2 = items[1]['item_id']
            print(f"Testing compatibility between {id1} and {id2}...")
            
            dist = recommender.predict_compatibility(id1, id2)
            score = recommender.predict_similarity_score(id1, id2)
            
            print(f"Distance: {dist:.4f}")
            print(f"Similarity Score: {score:.2f}")

