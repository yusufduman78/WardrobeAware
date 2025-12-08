"""
Fashion-CLIP Embedding Extraction Module
Optimized with PyTorch Best Practices for Memory Efficiency

Key features:
- Parallel image loading with DataLoader (num_workers)
- Lazy loading: images loaded only when needed
- Single clean progress bar
- All internal FashionCLIP/HuggingFace progress bars disabled
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from PIL import Image

try:
    from fashion_clip.fashion_clip import FashionCLIP
except ImportError:
    try:
        from fashion_clip import FashionCLIP
    except ImportError:
        raise ImportError("Could not import FashionCLIP. Please check installation.")
from src import fclip_config as config
from utils import fclip_utils as utils


class FashionDataset(Dataset):
    """
    PyTorch Dataset for Lazy Loading of Images.
    Only stores file paths in __init__, loads images in __getitem__.
    """
    def __init__(self, items: List[Dict]):
        """
        Args:
            items: List of dictionaries containing 'image_path', 'item_id', 'text_prompt'
        """
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Convert string path back to Path if needed
        image_path = Path(item['image_path']) if isinstance(item['image_path'], str) else item['image_path']

        try:
            # Lazy Load: Open image only when needed, convert to RGB immediately
            # Use context manager to ensure file handle is closed immediately
            with Image.open(image_path) as img:
                # Convert to RGB and load into memory (forces decode, closes file)
                image = img.convert("RGB")
                # Ensure image is fully loaded before closing file handle
                image.load()
            return image, item['item_id'], item['text_prompt']
        except Exception as e:
            # Return None to be filtered out in collate_fn
            return None


def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL Images and filter out None values.
    Returns lists as FashionCLIP expects PIL Images, not tensors.
    """
    # Filter out None values (failed image loads)
    batch = [x for x in batch if x is not None]
    
    if not batch:
        return None
    
    # Unzip the batch
    images, item_ids, text_prompts = zip(*batch)
    
    # Return as lists (FashionCLIP expects lists of PIL Images/Strings)
    return list(images), list(item_ids), list(text_prompts)


class EmbeddingExtractor:
    """Extract embeddings from fashion items using Fashion-CLIP"""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the embedding extractor

        Args:
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.device = device or utils.get_device()
        print(f"Initializing Fashion-CLIP on {self.device}...")
        self.model = FashionCLIP(config.FASHION_CLIP_MODEL_NAME)
        self.model.model.to(self.device)
        self.model.model.eval()
        print("Fashion-CLIP model loaded successfully.")

    def process_dataset(
        self,
        split_path: Path,
        metadata: Dict,
        use_text: bool = True,
        save_embeddings: bool = True,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process entire dataset and extract embeddings using DataLoader.
        PyTorch best practices: lazy loading, proper DataLoader configuration.

        Args:
            split_path: Path to train/test/valid JSON file
            metadata: Item metadata dictionary
            use_text: Whether to extract text embeddings
            save_embeddings: Whether to save embeddings to disk
            batch_size: Batch size for DataLoader (default: 32)
            num_workers: Number of worker processes for DataLoader (default: 4)

        Returns:
            Tuple of (embeddings, item_ids, image_embeddings, text_embeddings)
        """
        print(f"Processing dataset from {split_path}...")

        # Load outfit data
        outfit_data = utils.load_json(split_path)

        # 1. Collect unique items (Metadata only - Lightweight, no image loading)
        unique_items = []
        item_ids_set = set()

        print("Scanning dataset for unique items...")
        for outfit_entry in outfit_data:
            for item_data in outfit_entry.get('items', []):
                item_id = item_data['item_id']

                if item_id in item_ids_set:
                    continue
                item_ids_set.add(item_id)

                item_meta = metadata.get(item_id, {})
                if item_meta:
                    image_path = config.IMAGE_DIR / f"{item_id}.jpg"
                    text_prompt = utils.get_item_text(item_meta)

                    # Only add if file exists (fast check, no image loading)
                    if image_path.exists():
                        unique_items.append({
                            'item_id': item_id,
                            'image_path': str(image_path),  # Store as string for serialization
                            'text_prompt': text_prompt
                        })

        print(f"Found {len(unique_items)} unique items with valid images.")

        # 2. Create Dataset and DataLoader with optimal settings
        dataset = FashionDataset(unique_items)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True if self.device == 'cuda' else False,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=False,  # Windows compatibility
            drop_last=False
        )

        all_image_embeddings = []
        all_text_embeddings = []
        all_valid_ids = []

        print(f"Starting extraction with batch_size={batch_size}, num_workers={num_workers}...")

        # 3. Extract embeddings with direct model inference (bypass FashionCLIP's encode methods)
        # This avoids FashionCLIP's internal tqdm bars and HuggingFace datasets progress bars
        with torch.no_grad():
            # Single clean progress bar on DataLoader
            pbar = tqdm(dataloader, desc="Extracting", unit="batch", leave=True)
            
            for batch in pbar:
                if batch is None:
                    continue

                batch_images, batch_ids, batch_texts = batch

                # Extract Image Embeddings - Direct model inference (no FashionCLIP.encode_images)
                # Preprocess images
                inputs = self.model.preprocess(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                img_embs = self.model.model.get_image_features(**inputs).detach().cpu().numpy()
                img_embs = utils.normalize_embeddings(img_embs)
                all_image_embeddings.append(img_embs)

                # Extract Text Embeddings (if requested) - Direct model inference
                if use_text:
                    text_inputs = self.model.preprocess(
                        text=batch_texts, 
                        return_tensors="pt",
                        max_length=77,
                        padding="max_length",
                        truncation=True
                    )
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    txt_embs = self.model.model.get_text_features(**text_inputs).detach().cpu().numpy()
                    txt_embs = utils.normalize_embeddings(txt_embs)
                    all_text_embeddings.append(txt_embs)

                all_valid_ids.extend(batch_ids)

                # Free memory immediately
                del batch_images, batch_ids, batch_texts, batch, inputs
                if use_text:
                    del text_inputs
                    
            pbar.close()

        # 4. Concatenate results
        if not all_image_embeddings:
            print("No embeddings extracted!")
            return np.array([]), [], None, None

        final_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
        final_text_embeddings = np.concatenate(all_text_embeddings, axis=0) if use_text else None

        print(f"Extraction complete. Shape: {final_image_embeddings.shape}")

        # 5. Save embeddings
        if save_embeddings:
            config.EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            np.save(config.IMAGE_EMBEDDINGS_PATH, final_image_embeddings)
            utils.save_json(all_valid_ids, config.ITEM_IDS_PATH)

            if final_text_embeddings is not None:
                np.save(config.TEXT_EMBEDDINGS_PATH, final_text_embeddings)

            print(f"Embeddings saved to {config.EMBEDDINGS_OUTPUT_DIR}")

        return final_image_embeddings, all_valid_ids, final_image_embeddings, final_text_embeddings

    # Keep these for backward compatibility
    def extract_image_embeddings(self, images: List[Image.Image], batch_size: int = config.BATCH_SIZE) -> np.ndarray:
        return utils.normalize_embeddings(self.model.encode_images(images, batch_size=batch_size))

    def extract_text_embeddings(self, texts: List[str], batch_size: int = config.BATCH_SIZE) -> np.ndarray:
        return utils.normalize_embeddings(self.model.encode_text(texts, batch_size=batch_size))
