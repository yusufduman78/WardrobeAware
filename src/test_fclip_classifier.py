"""
Test script for FashionCLIP category classifier on Polyvore dataset.
Tests the classifier on real images from each category.
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
import pickle
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import config and utils
from backend import config as backend_config
from utils import fclip_utils as utils
from src import fclip_config as fclip_config
from src.fclip_embedding_extractor import EmbeddingExtractor

try:
    from fashion_clip.fashion_clip import FashionCLIP
except ImportError:
    try:
        from fashion_clip import FashionCLIP
    except ImportError:
        raise ImportError("Could not import FashionCLIP. Please install it.")


def load_classifier():
    """Load the trained FashionCLIP classifier"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load classifier checkpoint
    classifier_path = PROJECT_ROOT / "models" / "fclip_category_classifier.pth"
    label_encoder_path = PROJECT_ROOT / "models" / "fclip_category_label_encoder.pkl"
    
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found at {classifier_path}")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")
    
    # Load checkpoint
    checkpoint = torch.load(classifier_path, map_location='cpu')
    num_categories = checkpoint['num_categories']
    
    # Load FashionCLIP model
    print("Loading FashionCLIP model...")
    fashion_clip = FashionCLIP(fclip_config.FASHION_CLIP_MODEL_NAME)
    fashion_clip.model.to(device)
    fashion_clip.model.eval()
    
    # Create classifier head
    classifier_head = nn.Linear(512, num_categories)
    classifier_head.load_state_dict(checkpoint['model_state_dict'])
    classifier_head.eval()
    classifier_head.to(device)
    
    # Load label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Classifier loaded successfully!")
    print(f"  Categories: {len(label_encoder.classes_)}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  Categories: {list(label_encoder.classes_)}")
    
    return fashion_clip, classifier_head, label_encoder, device


def predict_category(image_path, fashion_clip, classifier_head, label_encoder, device):
    """Predict category for a single image"""
    try:
        # Load and preprocess image
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image.load()
        
        # Get embedding from FashionCLIP
        inputs = fashion_clip.preprocess(images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            img_emb = fashion_clip.model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
        # Predict with classifier
        with torch.no_grad():
            logits = classifier_head(img_emb)
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(logits, dim=1).item()
        
        predicted_category = label_encoder.inverse_transform([predicted_label])[0]
        confidence = probabilities[0][predicted_label].item()
        
        # Get top 3 predictions
        prob_values = probabilities.cpu().numpy()[0]
        category_probs = dict(zip(label_encoder.classes_, prob_values))
        sorted_categories = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_categories[:3]
        
        return predicted_category, confidence, top3
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, 0.0, []


def test_classifier(num_samples_per_category=10):
    """Test classifier on Polyvore dataset"""
    print("=" * 80)
    print("Testing FashionCLIP Category Classifier")
    print("=" * 80)
    
    # Load classifier
    fashion_clip, classifier_head, label_encoder, device = load_classifier()
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = utils.load_metadata()
    print(f"Loaded {len(metadata)} items from metadata")
    
    # Load images directory
    images_dir = Path(backend_config.IMAGES_DIR)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Group items by category
    print("\nGrouping items by category...")
    category_items = defaultdict(list)
    for item_id, data in metadata.items():
        category = data.get('semantic_category')
        if category:
            image_path = images_dir / f"{item_id}.jpg"
            if image_path.exists():
                category_items[category].append(item_id)
    
    print(f"Found {len(category_items)} categories with images")
    for cat, items in category_items.items():
        print(f"  {cat}: {len(items)} items")
    
    # Test on samples from each category
    print(f"\nTesting on {num_samples_per_category} samples per category...")
    print("=" * 80)
    
    all_predictions = []
    category_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'details': []})
    
    for category in sorted(category_items.keys()):
        items = category_items[category]
        if len(items) < num_samples_per_category:
            sample_items = items
        else:
            sample_items = np.random.choice(items, num_samples_per_category, replace=False)
        
        print(f"\nTesting category: {category} ({len(sample_items)} samples)")
        print("-" * 80)
        
        for item_id in tqdm(sample_items, desc=f"  {category}", leave=False):
            image_path = images_dir / f"{item_id}.jpg"
            
            predicted, confidence, top3 = predict_category(
                image_path, fashion_clip, classifier_head, label_encoder, device
            )
            
            if predicted is not None:
                is_correct = (predicted == category)
                category_results[category]['total'] += 1
                if is_correct:
                    category_results[category]['correct'] += 1
                
                all_predictions.append({
                    'item_id': item_id,
                    'true_category': category,
                    'predicted_category': predicted,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
                category_results[category]['details'].append({
                    'item_id': item_id,
                    'predicted': predicted,
                    'confidence': confidence,
                    'correct': is_correct,
                    'top3': top3
                })
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    total_correct = 0
    total_samples = 0
    
    for category in sorted(category_results.keys()):
        results = category_results[category]
        if results['total'] > 0:
            accuracy = (results['correct'] / results['total']) * 100
            total_correct += results['correct']
            total_samples += results['total']
            
            print(f"\n{category}:")
            print(f"  Accuracy: {accuracy:.2f}% ({results['correct']}/{results['total']})")
            
            # Show some examples
            print(f"  Sample predictions:")
            for detail in results['details'][:3]:  # Show first 3
                status = "✓" if detail['correct'] else "✗"
                print(f"    {status} Predicted: {detail['predicted']} "
                      f"(confidence: {detail['confidence']:.2%})")
                if not detail['correct']:
                    print(f"      Top 3: {', '.join([f'{cat}({prob:.2%})' for cat, prob in detail['top3']])}")
    
    # Overall accuracy
    if total_samples > 0:
        overall_accuracy = (total_correct / total_samples) * 100
        print("\n" + "=" * 80)
        print(f"OVERALL ACCURACY: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
        print("=" * 80)
    
    # Confusion matrix (simplified)
    print("\nConfusion Matrix (Top 5 errors):")
    print("-" * 80)
    errors = [p for p in all_predictions if not p['correct']]
    error_counts = defaultdict(int)
    for err in errors:
        key = f"{err['true_category']} -> {err['predicted_category']}"
        error_counts[key] += 1
    
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    for (error_pair, count) in sorted_errors[:5]:
        print(f"  {error_pair}: {count} times")
    
    print("\nTest completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test FashionCLIP category classifier")
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per category to test (default: 10)"
    )
    args = parser.parse_args()
    
    try:
        test_classifier(num_samples_per_category=args.samples)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
