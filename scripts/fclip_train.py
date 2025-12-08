"""
Training script for Metric Learning with Online Semi-Hard Triplet Mining
"""
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json

from src import fclip_config as config
from utils import fclip_utils as utils
from src.fclip_metric_learning import EmbeddingProjection, TripletLoss, OnlineTripletMiner


class FashionDataset(Dataset):
    """Dataset for loading pre-computed embeddings and outfit labels"""

    def __init__(self, embeddings_path: Path, item_ids_path: Path, outfit_data_path: Path):
        self.embeddings = np.load(embeddings_path)
        self.item_ids = utils.load_json(item_ids_path)
        self.id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}

        # Load outfit data to generate labels
        # Items in the same outfit get the same label (outfit_id)
        # Note: An item can belong to multiple outfits. For simplicity in this batch-based approach,
        # we'll assign a unique label per outfit and duplicate items if needed,
        # OR we just sample outfits.
        # Better approach for Online Mining:
        # Construct batches such that they contain P * K items (P outfits, K items per outfit).

        self.samples = []
        outfits = utils.load_json(outfit_data_path)

        print("Preparing dataset...")
        for outfit_idx, outfit in enumerate(outfits):
            items = outfit.get('items', [])
            valid_items = []
            for item in items:
                item_id = item.get('item_id')
                if item_id in self.id_to_idx:
                    valid_items.append(self.id_to_idx[item_id])

            # Only include outfits with at least 2 items (to form positive pairs)
            if len(valid_items) >= 2:
                for idx in valid_items:
                    self.samples.append({
                        'embedding_idx': idx,
                        'label': outfit_idx
                    })

        print(f"Dataset prepared with {len(self.samples)} samples from {len(outfits)} outfits.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        embedding = self.embeddings[sample['embedding_idx']]
        label = sample['label']
        return torch.FloatTensor(embedding), torch.LongTensor([label])


def train(epochs: int = 2, batch_size: int = 64, learning_rate: float = 1e-4):
    """Training loop"""
    device = utils.get_device()
    print(f"Training on {device}")

    # Determine embedding path and dimension
    # Determine embedding path and dimension
    embeddings_path = config.IMAGE_EMBEDDINGS_PATH
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")

    # Load a sample to check dim
    sample = np.load(embeddings_path, mmap_mode='r')
    input_dim = sample.shape[1]

    print(f"Input embedding dimension: {input_dim}")

    # Initialize dataset and dataloader
    dataset = FashionDataset(
        embeddings_path=embeddings_path,
        item_ids_path=config.ITEM_IDS_PATH,
        outfit_data_path=config.TRAIN_SPLIT_PATH
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model, loss, and miner
    model = EmbeddingProjection(input_dim=input_dim, output_dim=512).to(device)
    criterion = TripletLoss(margin=0.5)
    miner = OnlineTripletMiner(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        valid_triplets = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_embeddings, batch_labels in pbar:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.squeeze().to(device)

            # Forward pass
            projected = model(batch_embeddings)

            # Mine triplets
            anchors, positives, negatives = miner.mine(projected, batch_labels)

            if anchors is None:
                continue

            # Compute loss
            loss = criterion(anchors, positives, negatives)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            valid_triplets += anchors.shape[0]

            pbar.set_postfix({'loss': loss.item(), 'triplets': anchors.shape[0]})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}, Total Triplets Mined: {valid_triplets}")

    # Save model
    config.MODELS_DIR.mkdir(exist_ok=True)
    save_path = config.MODELS_DIR / "projection_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
