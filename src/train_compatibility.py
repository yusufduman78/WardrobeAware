"""
Training Script for Fashion Compatibility Model
Uses Triplet Loss with Hard Negative Mining to learn compatibility.
"""
import argparse
import logging
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add source directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src import fclip_config as config
from src.fclip_metric_learning import EmbeddingProjection, TripletLoss, HardTripletMiner
from src.fclip_dataset import PolyvoreEmbeddingDataset, TripletDataset
from utils import fclip_utils as utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.TRAIN_LOGS_DIR / "training.log")
    ]
)
logger = logging.getLogger(__name__)

def train(args):
    """Main training loop"""
    device = utils.get_device()
    logger.info(f"Using device: {device}")

    # 1. Load Data
    logger.info("Loading embeddings and metadata...")
    # Using existing config files for training data
    outfit_data = utils.load_json(config.TRAIN_SPLIT_PATH)
    item_metadata = utils.load_metadata()
    
    # Load base dataset (raw embeddings)
    base_dataset = PolyvoreEmbeddingDataset()
    
    # 2. Initialize Model
    model = EmbeddingProjection(input_dim=config.EMBEDDING_DIM, output_dim=args.output_dim)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = TripletLoss(margin=args.margin)
    miner = HardTripletMiner(
        margin=args.margin,
        num_hard_negatives=args.num_hard_negatives,
        num_hard_positives=args.num_hard_positives,
        device=device
    )

    # Metrics history
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [] # Placeholder if we implement validation
    }
    
    logger.info("Starting training...")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        # Mine triplets (Offline mining strategy for now, as implemented in miner)
        # We mine new triplets every epoch based on current embeddings state?
        # Ideally, we should update embeddings if the backbone was training, but here
        # we are training the projection head. The input embeddings are FIXED efficiently.
        # BUT, the hard mining depends on the distance in the PROJECTED space?
        # Strictly speaking, hard mining should use current model state. 
        # The current miner implementation computes distances on INPUT embeddings if passed directly.
        # Let's project embeddings first to find hard triplets in the LEARNED space.
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Projecting embeddings for mining...")
        
        # Project all embeddings for mining
        with torch.no_grad():
            model.eval()
            # Process in batches to avoid OOM
            projected_embs_list = []
            loader = DataLoader(base_dataset, batch_size=args.batch_size * 4, shuffle=False)
            
            for batch in loader:
                embs = batch['embedding'].to(device)
                proj = model(embs)
                projected_embs_list.append(proj.cpu().numpy())
            
            current_embeddings = np.concatenate(projected_embs_list, axis=0)
        
        # Mine hard triplets using projected embeddings
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Mining hard triplets...")
        triplets = miner.mine_hard_triplets(
            embeddings=current_embeddings,
            item_ids=base_dataset.item_ids,
            outfit_data=outfit_data,
            metadata=item_metadata
        )
        
        if not triplets:
            logger.warning("No triplets mined! Stopping training.")
            break
            
        triplet_dataset = TripletDataset(triplets, base_dataset.embeddings) # Use original embeddings as input
        triplet_loader = DataLoader(triplet_dataset, batch_size=args.batch_size, shuffle=True)
        
        model.train()
        pbar = tqdm(triplet_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        steps = 0
        
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            
            loss = criterion(emb_a, emb_p, emb_n)
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy (d(a,p) < d(a,n))
            with torch.no_grad():
                d_p = torch.norm(emb_a - emb_p, p=2, dim=1)
                d_n = torch.norm(emb_a - emb_n, p=2, dim=1)
                acc = (d_p < d_n).float().mean().item()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_acc += acc
            steps += 1
            
            pbar.set_postfix({'loss': f"{loss_val:.4f}", 'acc': f"{acc:.4f}"})
        
        avg_loss = epoch_loss / steps if steps > 0 else 0
        avg_acc = epoch_acc / steps if steps > 0 else 0
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f} - Avg Acc: {avg_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = config.CHECKPOINT_DIR / f"compatibility_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # 4. Save Final Model
    final_model_path = config.MODELS_DIR / "projection_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # 5. Plot Results
    plot_training_curves(history)

def plot_training_curves(history):
    """Plot and save training loss and accuracy curves"""
    # Plot Loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    # plt.plot(history['valid_loss'], label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Triplet Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Triplet Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = config.TRAIN_PLOTS_DIR / "training_curves.png"
    plt.savefig(plot_path)
    logger.info(f"Saved training curves to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Fashion Compatibility Model")
    
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--margin', type=float, default=config.TRIPLET_MARGIN, help='Triplet loss margin')
    parser.add_argument('--output_dim', type=int, default=512, help='Output embedding dimension')
    parser.add_argument('--num_hard_negatives', type=int, default=config.NUM_HARD_NEGATIVES, help='Number of hard negatives to mine')
    parser.add_argument('--num_hard_positives', type=int, default=config.NUM_HARD_POSITIVES, help='Number of hard positives to mine')
    
    args = parser.parse_args()
    
    try:
        train(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
