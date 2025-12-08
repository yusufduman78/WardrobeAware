"""
Fine-tune FashionCLIP's last layer as a category classifier for Polyvore dataset.
Only trains the classifier head, freezes the backbone.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import config and utils
from src import fclip_config as config
from utils import fclip_utils as utils

try:
    from fashion_clip.fashion_clip import FashionCLIP
except ImportError:
    try:
        from fashion_clip import FashionCLIP
    except ImportError:
        raise ImportError("Could not import FashionCLIP. Please install it.")


class PolyvoreCategoryDataset(Dataset):
    """Dataset for category classification - Lazy loading to prevent memory leaks"""
    
    def __init__(self, metadata, image_dir, transform=None):
        self.metadata = metadata
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Filter items with semantic_category - only store paths, not images
        self.items = []
        for item_id, data in metadata.items():
            semantic_category = data.get('semantic_category')
            if semantic_category:
                image_path = self.image_dir / f"{item_id}.jpg"
                if image_path.exists():
                    self.items.append({
                        'item_id': item_id,
                        'category': semantic_category,
                        'image_path': str(image_path)  # Store as string for lazy loading
                    })
        
        print(f"Loaded {len(self.items)} items with categories")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = Path(item['image_path'])
        
        try:
            # Lazy Load: Open image only when needed, convert to RGB immediately
            # Use context manager to ensure file handle is closed immediately
            with Image.open(image_path) as img:
                # Convert to RGB and load into memory (forces decode, closes file)
                image = img.convert("RGB")
                # Ensure image is fully loaded before closing file handle
                image.load()
            return image, item['category'], item['item_id']
        except Exception as e:
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224))
            return image, item['category'], item['item_id']


class FashionCLIPClassifier(nn.Module):
    """FashionCLIP with a classifier head"""
    
    def __init__(self, fashion_clip_model, num_categories, device):
        super().__init__()
        self.fashion_clip = fashion_clip_model
        self.device = device
        
        # Freeze FashionCLIP backbone
        # fashion_clip.model is the CLIPModel, freeze its parameters
        for param in self.fashion_clip.model.parameters():
            param.requires_grad = False
        
        # Add classifier head
        # FashionCLIP produces 512-dim embeddings
        self.classifier = nn.Linear(512, num_categories)
        self.classifier.to(device)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, images):
        # images is a list of PIL Images (from DataLoader)
        # Get embeddings from FashionCLIP (frozen) - Direct model inference to avoid tqdm bars
        with torch.no_grad():
            # Preprocess images directly (bypasses FashionCLIP.encode_images which shows tqdm)
            inputs = self.fashion_clip.preprocess(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings directly from model
            img_embs = self.fashion_clip.model.get_image_features(**inputs)
            # Normalize embeddings
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
            
            # Free memory immediately
            del inputs
        
        # Classify (only this part is trainable)
        logits = self.classifier(img_embs)
        return logits


def train_classifier():
    """Train FashionCLIP classifier"""
    device_str = utils.get_device()
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Load metadata
    print("Loading metadata...")
    metadata = utils.load_metadata()
    print(f"Loaded {len(metadata)} items from metadata")
    
    # Get all unique categories
    all_categories = [data.get('semantic_category') for data in metadata.values() 
                     if data.get('semantic_category')]
    unique_categories = sorted(list(set(all_categories)))
    num_categories = len(unique_categories)
    print(f"Found {num_categories} unique categories: {unique_categories}")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_categories)
    
    # Save label encoder
    label_encoder_path = config.MODELS_DIR / "fclip_category_label_encoder.pkl"
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {label_encoder_path}")
    
    # Load FashionCLIP
    print("Loading FashionCLIP model...")
    fashion_clip = FashionCLIP(config.FASHION_CLIP_MODEL_NAME)
    fashion_clip.model.to(device)
    fashion_clip.model.eval()
    print("FashionCLIP model loaded and set to eval mode")
    
    # Create classifier model
    model = FashionCLIPClassifier(fashion_clip, num_categories, device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create dataset
    print("Creating dataset...")
    # No transform needed - FashionCLIP expects PIL Images
    dataset = PolyvoreCategoryDataset(metadata, config.IMAGE_DIR, transform=None)
    
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Custom collate function to handle PIL Images and filter None values
    def custom_collate_fn(batch):
        # Filter out None values (failed image loads)
        batch = [x for x in batch if x is not None]
        if not batch:
            return None
        # Unzip the batch
        images, categories, item_ids = zip(*batch)
        # Return as lists (FashionCLIP expects lists of PIL Images)
        return list(images), list(categories), list(item_ids)
    
    # Create data loaders with proper batch processing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=custom_collate_fn,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=custom_collate_fn,
        pin_memory=(device.type == 'cuda')
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    num_epochs = 5  # Quick training
    best_val_acc = 0.0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in train_pbar:
            if batch is None:
                continue
                
            images, categories, item_ids = batch
            # images is a list of PIL Images from DataLoader
            categories_encoded = label_encoder.transform(categories)
            labels = torch.tensor(categories_encoded, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Free memory immediately
            del images, categories, item_ids, logits, labels, predicted
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for batch in val_pbar:
                if batch is None:
                    continue
                    
                images, categories, item_ids = batch
                # images is a list of PIL Images from DataLoader
                categories_encoded = label_encoder.transform(categories)
                labels = torch.tensor(categories_encoded, dtype=torch.long).to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Free memory immediately
                del images, categories, item_ids, logits, labels, predicted
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = config.MODELS_DIR / "fclip_category_classifier.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_categories': num_categories,
            }, model_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%) to {model_path}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation:")
    print("="*80)
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.MODELS_DIR / 'fclip_category_classifier.pth'}")
    print(f"Label encoder saved to: {label_encoder_path}")


if __name__ == "__main__":
    train_classifier()
