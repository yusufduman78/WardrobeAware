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

    def load_image(self, item_id):
        # Legacy: used for ResNet
        filename = f"{item_id}.jpg"
        path = os.path.join(config.IMAGES_ROOT, filename)
        
        if not os.path.exists(path):
            # Return black image placeholder
            img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))
        else:
            img = Image.open(path).convert('RGB')
            
        return self.transform(img).unsqueeze(0).to(self.device)
        
    def get_embedding(self, item_id):
        # For Projection Model
        if self.embeddings_map and str(item_id) in self.embeddings_map:
            return torch.tensor(self.embeddings_map[str(item_id)]).to(self.device).unsqueeze(0)
        
        # Try finding cleaned ID
        clean = str(item_id).split('_')[0]
        if self.embeddings_map and clean in self.embeddings_map:
            return torch.tensor(self.embeddings_map[clean]).to(self.device).unsqueeze(0)
            
        return None

    def predict_compatibility(self, item_id_1, item_id_2):
        """
        Calculates compatibility distance. Lower is better.
        """
        if self.model_type == "Projection":
            emb1 = self.get_embedding(item_id_1)
            emb2 = self.get_embedding(item_id_2)
            
            if emb1 is None or emb2 is None:
                return 10.0 # High distance (unknown)
                
            with torch.no_grad():
                p1 = self.model(emb1)
                p2 = self.model(emb2)
                dist = F.pairwise_distance(p1, p2).item()
            return dist
            
        else:
            # ResNet Logic
            cat_id_1 = self.get_item_category_id(item_id_1)
            cat_id_2 = self.get_item_category_id(item_id_2)
            
            if cat_id_1 is None or cat_id_2 is None:
                return 10.0 # High distance
                
            img_1 = self.load_image(item_id_1)
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

    def predict_similarity_score(self, item_id_1, item_id_2):
        """
        Distance -> Score (0-100)
        """
        dist = self.predict_compatibility(item_id_1, item_id_2)
        
        # For projection model, distances might be smaller (e.g. 0.0-2.0)
        # ResNet distances were also similar.
        # Simple exponential decay map
        score = np.exp(-dist) * 100
        return score

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

