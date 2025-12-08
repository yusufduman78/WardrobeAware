import torch
import torch.nn as nn
import torch.nn.functional as F

class TypeAwareTripletLoss(nn.Module):
    """
    Makaledeki Denklem (3): Conditional Triplet Loss.
    Anchor (a), Positive (p) ve Negative (n) arasındaki mesafeyi optimize eder.
    """
    def __init__(self, margin=0.2):
        super(TypeAwareTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_proj, pos_proj, anchor_n_proj, neg_proj):
        # Anchor ve Positive arasındaki mesafe (Uyumlu çift)
        dist_pos = F.pairwise_distance(anchor_proj, pos_proj, p=2)
        
        # Anchor ve Negative arasındaki mesafe (Uyumsuz çift)
        # Not: Negatif örnek için anchor'ın maskesi 'neg_type'a göre tekrar hesaplanır.
        dist_neg = F.pairwise_distance(anchor_n_proj, neg_proj, p=2)
        
        # Loss = max(0, pos_dist - neg_dist + margin)
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return loss.mean()

class GeneralTripletLoss(nn.Module):
    """
    Maskelenmemiş Genel Embedding'ler için standart Triplet Loss. (L_sim)
    """
    def __init__(self, margin=0.2):
        super(GeneralTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_emb, pos_emb, neg_emb):
        # Anchor ve Positive arasındaki mesafe
        dist_pos = F.pairwise_distance(anchor_emb, pos_emb, p=2)
        
        # Anchor ve Negative arasındaki mesafe
        dist_neg = F.pairwise_distance(anchor_emb, neg_emb, p=2)
        
        # Loss = max(0, pos_dist - neg_dist + margin)
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return loss.mean()

def save_checkpoint(state, filename="checkpoint.pth"):
    print(f"=> Checkpoint kaydediliyor: {filename}")
    torch.save(state, filename)