"""
Model Comparison Script for Fashion Compatibility (Validation Set)
Compares different model architectures on the Compatibility AUC task using VALIDATION data.
Generates pairs dynamically from valid.json to ensure data availability.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from collections import defaultdict, OrderedDict
from sklearn.metrics import roc_auc_score
from typing import List

# Fix for parallel processing on Windows
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src import fclip_config as config
from src.fclip_metric_learning import EmbeddingProjection
from recommender.model import TypeAwareNet
from utils import fclip_utils as utils

def clean_id(item_id):
    return str(item_id).split('_')[0]

# ----------------------------------------------------------------------
# Wrappers
# ----------------------------------------------------------------------
class ModelWrapper:
    def score_pair(self, item_id_1: str, item_id_2: str) -> float:
        raise NotImplementedError

class FClipModelWrapper(ModelWrapper):
    def __init__(self, checkpoint_path, embeddings_map, device='cuda'):
        self.device = device
        self.embeddings_map = embeddings_map
        self.missing_count = 0
        self.is_concat = False
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # Check input dimension of first layer
        in_dim = 512
        if '0.weight' in state_dict:
            if state_dict['0.weight'].shape[1] == 1024:
                print("  Detected Concat-MLP structure (Input 1024).")
                self.is_concat = True
                in_dim = 1024
            elif state_dict['0.weight'].shape[1] == 512:
                print("  Detected Projection structure (Input 512).")
                self.is_concat = False
        
        # Reconstruct or Load
        if any(k.startswith('net.') for k in state_dict.keys()):
            # Cleaning keys for reconstruction
            clean_state = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    clean_state[k.replace('net.', '')] = v
            
            # Update check based on clean state
            if '0.weight' in clean_state and clean_state['0.weight'].shape[1] == 1024:
                 self.is_concat = True
                 in_dim = 1024
            
            self.model = self._build_sequential(clean_state, in_dim)
            self.model.load_state_dict(clean_state)
        else:
            # Standard load
            self.model = EmbeddingProjection(input_dim=512, output_dim=512)
            if self.is_concat: # Override if we detected concat but it's not 'net.' (unlikely but safe)
                 # We need to build a concat model manually if it's not standard
                 self.model = self._build_sequential(state_dict, 1024)
            
            try:
                self.model.load_state_dict(state_dict, strict=False)
            except:
                pass
            
        self.model.to(device)
        self.model.eval()

    def _build_sequential(self, state_dict, in_dim):
        # Heuristic reconstruction
        layers = []
        
        # Layer 0
        l0_out = state_dict['0.weight'].shape[0]
        layers.append(nn.Linear(in_dim, l0_out))
        
        if '1.weight' in state_dict: layers.append(nn.BatchNorm1d(l0_out))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Layer 4 (Optional)
        if '4.weight' in state_dict:
            l4_out = state_dict['4.weight'].shape[0]
            layers.append(nn.Linear(l0_out, l4_out))
            if '5.weight' in state_dict: layers.append(nn.BatchNorm1d(l4_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            last_dim = l4_out
        else:
            last_dim = l0_out
            
        # Layer 8 (Optional)
        if '8.weight' in state_dict:
             l8_out = state_dict['8.weight'].shape[0]
             layers.append(nn.Linear(last_dim, l8_out))
             
        return nn.Sequential(*layers)

    def get_embedding(self, item_id):
        if item_id in self.embeddings_map: return self.embeddings_map[item_id]
        cid = clean_id(item_id)
        if cid in self.embeddings_map: return self.embeddings_map[cid]
        return None

    def score_pair(self, item_id_1: str, item_id_2: str) -> float:
        e1 = self.get_embedding(item_id_1)
        e2 = self.get_embedding(item_id_2)
        if e1 is None or e2 is None:
            self.missing_count += 1
            return float('inf')
        
        t1 = torch.tensor(e1).to(self.device).unsqueeze(0)
        t2 = torch.tensor(e2).to(self.device).unsqueeze(0)
        
        if self.is_concat:
            # Concat model: returns score (higher is better usually, or logit)
            # We assume output is valid compatibility score.
            # Convert to 'distance' (negate) for uniform API if it's a probability/logit
            with torch.no_grad():
                inp = torch.cat([t1, t2], dim=1)
                score = self.model(inp).item()
                # If sigmoided, 1 is good. If linear logit, high is good.
                # We return -score so that "lower is better" logic works for AUC? 
                # Wait, AUC logic: true=1, score should be high.
                # Wrapper logic: score_pair returns DISTANCE.
                # So we return -score.
                return -score
        else:
            # Projection model: Euclidean distance
            with torch.no_grad():
                p1 = self.model(t1)
                p2 = self.model(t2)
                return F.pairwise_distance(p1, p2).item() # Lower is better

class TypeAwareModelWrapper(ModelWrapper):
    def __init__(self, checkpoint_path, image_dir, device='cuda', metadata=None):
        self.device = device
        self.image_dir = Path(image_dir)
        self.metadata = metadata or {}
        self.missing_count = 0
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.cat_to_id = checkpoint.get('cat_to_id', {})
        embedding_dim = checkpoint.get('args', {}).get('embedding_dim', 64) 
        num_categories = len(self.cat_to_id) if self.cat_to_id else 11
        
        self.model = TypeAwareNet(embedding_dim=embedding_dim, num_categories=num_categories, pretrained=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, item_id):
        clean = clean_id(item_id)
        path = self.image_dir / f"{clean}.jpg"
        try:
            if not path.exists(): return None
            img = Image.open(path).convert('RGB')
            return self.transform(img).to(self.device).unsqueeze(0)
        except: return None

    def get_cid(self, item_id):
        clean = clean_id(item_id)
        meta = self.metadata.get(clean, {})
        sem_cat = meta.get('semantic_category', 'all-body')
        return self.cat_to_id.get(sem_cat, 0)

    def score_pair(self, item_id_1: str, item_id_2: str) -> float:
        img1 = self.load_image(item_id_1)
        img2 = self.load_image(item_id_2)
        if img1 is None or img2 is None:
            self.missing_count += 1
            return float('inf')
            
        cid1 = torch.tensor([self.get_cid(item_id_1)]).to(self.device)
        cid2 = torch.tensor([self.get_cid(item_id_2)]).to(self.device)
        
        with torch.no_grad():
            emb1 = self.model.get_general_embedding(img1)
            emb2 = self.model.get_general_embedding(img2)
            mask = self.model.masks[cid1, cid2]
            proj1 = emb1 * mask
            proj2 = emb2 * mask
            return F.pairwise_distance(proj1, proj2).item()

# ----------------------------------------------------------------------
# Generator & Evaluator
# ----------------------------------------------------------------------

def load_fclip_embeddings():
    print("Loading FashionCLIP embeddings...")
    embs = np.load(config.IMAGE_EMBEDDINGS_PATH)
    ids = utils.load_json(config.ITEM_IDS_PATH)
    return {str(k): v for k, v in zip(ids, embs)}

def generate_validation_pairs(num_pairs=2000):
    print("Generating synthetic validation pairs from valid.json...")
    valid_path = config.DATA_DIR / "disjoint" / "valid.json"
    data = utils.load_json(valid_path)
    
    # Flatten items
    all_items = []
    outfits = []
    
    for entry in data:
        items = [x['item_id'] for x in entry['items']]
        if len(items) >= 2:
            outfits.append(items)
            all_items.extend(items)
            
    all_items = list(set(all_items))
    print(f"  Pool: {len(outfits)} outfits, {len(all_items)} unique items")
    
    pairs = [] # (id1, id2, label)
    
    # Positives
    for _ in range(num_pairs // 2):
        outfit = random.choice(outfits)
        if len(outfit) < 2: continue
        i1, i2 = random.sample(outfit, 2)
        pairs.append((i1, i2, 1))
        
    # Negatives
    for _ in range(num_pairs // 2):
        i1 = random.choice(all_items)
        i2 = random.choice(all_items)
        pairs.append((i1, i2, 0)) # Assume random is negative
        
    random.shuffle(pairs)
    return pairs

def run_evaluation(wrapper, pairs):
    y_true = []
    y_scores = []
    valid_count = 0
    
    for i1, i2, label in tqdm(pairs):
        dist = wrapper.score_pair(i1, i2)
        
        if dist == float('inf'): continue
            
        valid_count += 1
        y_true.append(label)
        # We need Score (High=Compatible). 
        # wrapper.score_pair returns Distance (Low=Compatible)
        # So we negate it (-dist).
        y_scores.append(-dist)
        
    print(f"  Valid Pairs: {valid_count}/{len(pairs)}")
    if hasattr(wrapper, 'missing_count'):
        print(f"  Missing Items: {wrapper.missing_count}")
        wrapper.missing_count = 0
        
    if len(set(y_true)) < 2: return 0.5
    return roc_auc_score(y_true, y_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', help='Paths to model checkpoints (.pth)')
    args = parser.parse_args()
    
    metadata = utils.load_metadata()
    device = utils.get_device()
    print(f"Using Device: {device}")
    
    # Generate Data
    pairs = generate_validation_pairs(num_pairs=4000)
    
    fclip_map = None
    results = defaultdict(dict)
    
    for model_path in args.models:
        print(f"\nEvaluating: {Path(model_path).name}")
        try:
            # Quick detection
            m_name = Path(model_path).name.lower()
            if any(x in m_name for x in ["rn", "projection", "compatibility", "epoch"]):
                m_type = 'EmbeddingProjection'
            else:
                m_type = 'TypeAwareNet'
                
            wrapper = None
            
            if m_type == 'EmbeddingProjection':
                if fclip_map is None: fclip_map = load_fclip_embeddings()
                wrapper = FClipModelWrapper(model_path, fclip_map, device=device)
            elif m_type == 'TypeAwareNet':
                wrapper = TypeAwareModelWrapper(model_path, config.IMAGE_DIR, device=device, metadata=metadata)
            
            if wrapper:
                auc = run_evaluation(wrapper, pairs)
                print(f"Result - Validation AUC: {auc:.4f}")
                results[Path(model_path).name]['AUC'] = auc
            
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print(f"{'Model':<35} | {'Val AUC':<10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<35} | {res.get('AUC', 0.0):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
