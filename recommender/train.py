import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
import torch.nn.functional as F

import config
from dataset import PolyvoreTripletDataset
from model import TypeAwareNet
from utils import TypeAwareTripletLoss, GeneralTripletLoss, save_checkpoint

def calculate_accuracy(proj_anchor, proj_pos, proj_neg):
    """
    Triplet Accuracy: Percentage of triplets where dist(a, p) < dist(a, n)
    """
    dist_pos = F.pairwise_distance(proj_anchor, proj_pos, p=2)
    dist_neg = F.pairwise_distance(proj_anchor, proj_neg, p=2)
    correct = (dist_pos < dist_neg).float()
    return correct.mean().item()

def train(args):
    # 1. Cihaz Ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Veri Hazırlığı
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PolyvoreTripletDataset(split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    # 3. Model Kurulumu
    num_categories = len(train_dataset.cat_to_id)
    model = TypeAwareNet(embedding_dim=config.EMBEDDING_DIM, num_categories=num_categories).to(device)

    # 4. Optimizer ve Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_comp = TypeAwareTripletLoss(margin=args.margin)
    criterion_sim = GeneralTripletLoss(margin=args.margin)

    # 5. Eğitim Döngüsü
    print(f"Eğitim başlıyor... Prefix: {args.prefix}, L1 Lambda: {args.lambda_l1}")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (a_img, p_img, n_img, a_type, p_type, n_type) in enumerate(loop):
            a_img, p_img, n_img = a_img.to(device), p_img.to(device), n_img.to(device)
            a_type, p_type, n_type = a_type.to(device), p_type.to(device), n_type.to(device)

            optimizer.zero_grad()

            # Forward Pass
            # Unpack all 7 return values
            proj_ap, proj_pos, proj_an, proj_neg, emb_a, emb_p, emb_n = model(a_img, p_img, n_img, a_type, p_type, n_type)

            # 1. Compatibility Loss (L_comp)
            loss_comp = criterion_comp(proj_ap, proj_pos, proj_an, proj_neg)
            
            # 2. Similarity Loss (L_sim) - Weight 0.5
            loss_sim = criterion_sim(emb_a, emb_p, emb_n)
            
            # 3. L1 Regularization - Weight 5e-4 (or args.lambda_l1)
            l1_loss = torch.norm(model.masks, p=1)
            
            # Total Loss
            # L_vse is currently omitted as we don't have text embeddings yet
            lambda_sim = 0.5
            total_loss = loss_comp + (lambda_sim * loss_sim) + (args.lambda_l1 * l1_loss)

            # Backward ve Step
            total_loss.backward()
            optimizer.step()

            # Metrics
            dist_pos = F.pairwise_distance(proj_ap, proj_pos, p=2)
            dist_neg = F.pairwise_distance(proj_an, proj_neg, p=2)
            acc = (dist_pos < dist_neg).float().mean().item()

            running_loss += total_loss.item()
            running_acc += acc
            
            loop.set_postfix(loss=total_loss.item(), acc=acc, l_sim=loss_sim.item(), l_l1=l1_loss.item())

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        print(f"Epoch {epoch+1} Bitti. Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        # Save Checkpoint
        filename = f"{args.prefix}_epoch_{epoch+1}.pth"
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'cat_to_id': train_dataset.cat_to_id,
            'args': vars(args)
        }, filename=filename)

    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Type-Aware Compatibility Model")
    parser.add_argument("--prefix", type=str, default="model", help="Prefix for saved model files")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--lambda_l1", type=float, default=config.LAMBDA_L1, help="L1 regularization weight")
    parser.add_argument("--margin", type=float, default=config.MARGIN, help="Triplet loss margin")
    
    args = parser.parse_args()
    train(args)