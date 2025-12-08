"""
Model Comparison Script for Fashion Compatibility
Compares different model architectures on the Fill-in-the-Blank (FITB) task and Compatibility AUC.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from collections import defaultdict, OrderedDict
from sklearn.metrics import roc_auc_score
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Imports
from src import fclip_config as config
from src.fclip_metric_learning import EmbeddingProjection
from recommender.model import TypeAwareNet
from utils import fclip_utils as utils

def clean_id(item_id):
    """Remove underscore suffix from Polyvore items (e.g., '12345_1' -> '12345')"""
    return str(item_id).split('_')[0]

# ----------------------------------------------------------------------
# Wrappers
# ----------------------------------------------------------------------

class ModelWrapper:
    def score_pair(self, item_id_1: str, item_id_2: str) -> float:
        raise NotImplementedError
        
    def score_outfit(self, item_ids: List[str]) -> float:
        if len(item_ids) < 2: return 0.0
        score = 0
        count = 0
        for i in range(len(item_ids)):
            for j in range(i+1, len(item_ids)):
                d = self.score_pair(item_ids[i], item_ids[j])
                if d != float('inf'):
                    score += d
                    count += 1
        return score / max(1, count)

class FClipModelWrapper(ModelWrapper):
    def __init__(self, checkpoint_path, embeddings_map, device='cuda', strict_keys=True):
        self.device = device
        self.embeddings_map = embeddings_map
        self.missing_count = 0
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        if any(k.startswith('net.') for k in state_dict.keys()):
            print("  Detected Sequential MLP structure. Reconstructing...")
            self.model = self._build_sequential_from_state(state_dict)
            clean_state = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    clean_state[k.replace('net.', '')] = v
            self.model.load_state_dict(clean_state)
        else:
            print("  Detected Standard EmbeddingProjection structure.")
            self.model = EmbeddingProjection(input_dim=512, output_dim=512)
            try:
                self.model.load_state_dict(state_dict, strict=strict_keys)
            except Exception as e:
                print(f"  Warning: Strict load failed ({e}), trying loose load.")
                self.model.load_state_dict(state_dict, strict=False)
            
        self.model.to(device)
        self.model.eval()

    def _build_sequential_from_state(self, state_dict):
        layers_map = {}
        max_idx = 0
        for k, v in state_dict.items():
            if k.startswith('net.'):
                parts = k.split('.')
                if parts[1].isdigit():
                    idx = int(parts[1])
                    param_type = parts[2]
                    max_idx = max(max_idx, idx)
                    if idx not in layers_map: layers_map[idx] = {}
                    layers_map[idx][param_type] = v
        
        seq = nn.Sequential()
        if 0 in layers_map:
            l0_out, l0_in = layers_map[0]['weight'].shape
            seq.add_module('0', nn.Linear(l0_in, l0_out))
        if 1 in layers_map:
             l1_w = layers_map[1]['weight']
             seq.add_module('1', nn.BatchNorm1d(l1_w.shape[0]))
        seq.add_module('2', nn.ReLU())
        seq.add_module('3', nn.Dropout(0.2))
        
        if 4 in layers_map:
            l4_out, l4_in = layers_map[4]['weight'].shape
            seq.add_module('4', nn.Linear(l4_in, l4_out))
        if 5 in layers_map:
             l5_w = layers_map[5]['weight']
             seq.add_module('5', nn.BatchNorm1d(l5_w.shape[0]))
        seq.add_module('6', nn.ReLU())
        seq.add_module('7', nn.Dropout(0.2))
        
        if 8 in layers_map:
            l8_out, l8_in = layers_map[8]['weight'].shape
            seq.add_module('8', nn.Linear(l8_in, l8_out))
        
        return seq

    def get_embedding(self, item_id):
        # 1. Exact match
        if item_id in self.embeddings_map:
            return self.embeddings_map[item_id]
            
        # 2. Cleaned match (stripped suffix)
        cid = clean_id(item_id)
        if cid in self.embeddings_map:
            return self.embeddings_map[cid]
            
        return None

    def score_pair(self, item_id_1: str, item_id_2: str) -> float:
        e1 = self.get_embedding(item_id_1)
        e2 = self.get_embedding(item_id_2)
        
        if e1 is None or e2 is None:
            self.missing_count += 1
            return float('inf')
            
        emb1 = torch.tensor(e1).to(self.device).unsqueeze(0)
        emb2 = torch.tensor(e2).to(self.device).unsqueeze(0)
        
        try:
            with torch.no_grad():
                proj1 = self.model(emb1)
                proj2 = self.model(emb2)
                dist = F.pairwise_distance(proj1, proj2).item()
            return dist
        except RuntimeError as e:
            if "mat1 and mat2 shapes" in str(e):
                # Critical dimension mismatch (e.g. 512 input to 256 model)
                # Instead of crashing, just return inf to invalidate this metric
                return float('inf')
            raise e

class TypeAwareModelWrapper(ModelWrapper):
    def __init__(self, checkpoint_path, image_dir, device='cuda', metadata=None):
        self.device = device
        self.image_dir = Path(image_dir)
        self.missing_count = 0
        self.metadata = metadata or {}
        self.debug_printed = 0
        
        print(f"Loading checkpoint: {checkpoint_path}")
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
            if not path.exists():
                if self.debug_printed < 5:
                    print(f"DEBUG: MISSING FAIL. ID='{item_id}' Clean='{clean}' Path='{path}' Exists={path.exists()}")
                    self.debug_printed += 1
                return None
            img = Image.open(path).convert('RGB')
            return self.transform(img).to(self.device).unsqueeze(0)
        except Exception as e:
            if self.debug_printed < 5:
                print(f"DEBUG: EXCEPTION. ID='{item_id}' Path='{path}' Error='{e}'")
                self.debug_printed += 1
            return None
        
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
            dist = F.pairwise_distance(proj1, proj2).item()
            
        return dist

# ----------------------------------------------------------------------
# Logic
# ----------------------------------------------------------------------

def load_fclip_embeddings():
    print("Loading FashionCLIP embeddings...")
    embs = np.load(config.IMAGE_EMBEDDINGS_PATH)
    ids = utils.load_json(config.ITEM_IDS_PATH)
    # Ensure IDs are strings
    return {str(k): v for k, v in zip(ids, embs)}

def detect_model_type(path):
    try:
        cp = torch.load(path, map_location='cpu')
        state = cp.get('model_state_dict', cp.get('state_dict', cp))
        if not isinstance(state, dict): return 'Unknown'
        keys = list(state.keys())
        
        if any(k.startswith('linear1') for k in keys): return 'EmbeddingProjection'
        if any(k.startswith('net.') for k in keys): return 'EmbeddingProjection'
        if any(k.startswith('backbone.') for k in keys): return 'TypeAwareNet'
    except:
        pass
    return 'Unknown'

def run_compatibility_eval(wrapper, compat_file):
    print(f"Running Compatibility AUC on {compat_file.name}...")
    
    with open(compat_file, 'r') as f:
        lines = f.readlines()
        
    y_true = []
    y_scores = []
    
    valid_count = 0
    dim_error_reported = False
    
    for line in tqdm(lines):
        parts = line.strip().split()
        label = int(parts[0])
        item_ids = parts[1:] # Keep original IDs (wrapper handles cleaning)
        
        outfit_score = wrapper.score_outfit(item_ids)
        
        if outfit_score == float('inf'):
            continue
            
        valid_count += 1
        y_true.append(label)
        y_scores.append(-outfit_score)
        
    print(f"  Valid Compatibility Pairs: {valid_count}/{len(lines)}")
    if hasattr(wrapper, 'missing_count'):
        print(f"  Missing Items (or errors): {wrapper.missing_count}")
        wrapper.missing_count = 0
        
    if len(set(y_true)) < 2:
        print("  Warning: Only one class present in valid data.")
        return 0.5
        
    auc = roc_auc_score(y_true, y_scores)
    return auc

def run_fitb_eval(wrapper, questions):
    print(f"Running FITB on {len(questions)} questions...")
    correct = 0
    total = 0
    valid_pred = 0
    
    for q in tqdm(questions):
        context = q['question']
        candidates = q['answers']
        
        scores = []
        for cand in candidates:
            s = 0
            cnt = 0
            for ctx in context:
                if ctx == "_": continue
                d = wrapper.score_pair(cand, ctx)
                if d != float('inf'):
                    s += d
                    cnt += 1
            
            avg = s / max(cnt, 1)
            if cnt == 0: avg = float('inf')
            scores.append(avg)
            
        if all(s == float('inf') for s in scores):
            pass
        else:
            valid_pred += 1
            if np.argmin(scores) == 0:
                correct += 1
        total += 1
        
    print(f"  Valid FITB Questions: {valid_pred}/{total}")
    return correct / max(total, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', help='Paths to model checkpoints (.pth)')
    args = parser.parse_args()
    
    metadata = utils.load_metadata()
    device = utils.get_device()
    print(f"Using Device: {device}")
    
    fib_path = config.DATA_DIR / "disjoint" / "fill_in_blank_test.json"
    if not fib_path.exists(): fib_path = config.DATA_DIR / "disjoint" / "fill_in_blank.json"
    fitb_data = utils.load_json(fib_path)
    
    compat_path = config.DATA_DIR / "disjoint" / "compatibility_test.txt"
    fclip_map = None
    results = defaultdict(dict)
    
    for model_path in args.models:
        print(f"\n{'='*20}\nEvaluating: {Path(model_path).name}\n{'='*20}")
        try:
            m_type = detect_model_type(model_path)
            wrapper = None
            
            if m_type == 'EmbeddingProjection':
                if fclip_map is None: fclip_map = load_fclip_embeddings()
                wrapper = FClipModelWrapper(model_path, fclip_map, device=device)
            elif m_type == 'TypeAwareNet':
                wrapper = TypeAwareModelWrapper(model_path, config.IMAGE_DIR, device=device, metadata=metadata)
            else:
                print("Skipping unknown model.")
                continue
                
            if compat_path.exists():
                auc = run_compatibility_eval(wrapper, compat_path)
                print(f"Result - Compatibility AUC: {auc:.4f}")
                results[Path(model_path).name]['AUC'] = auc
            
            acc = run_fitb_eval(wrapper, fitb_data)
            print(f"Result - FITB Accuracy: {acc:.4f}")
            results[Path(model_path).name]['FITB'] = acc
            
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print(f"{'Model':<35} | {'AUC':<10} | {'FITB':<10}")
    print("-" * 60)
    for name, res in results.items():
        auc = res.get('AUC', 0.0)
        fitb = res.get('FITB', 0.0)
        print(f"{name:<35} | {auc:.4f}     | {fitb:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
