import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights # Yeni import
import torch.nn.functional as F

class TypeAwareNet(nn.Module):
    def __init__(self, embedding_dim=64, num_categories=10, pretrained=True):
        super(TypeAwareNet, self).__init__()
        
        print(f"Model oluşturuluyor... Embedding Dim: {embedding_dim}, Categories: {num_categories}")
        
        # Warning'i düzeltmek için güncel yükleme yöntemi:
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        resnet = models.resnet18(weights=weights)
        
        # Son katmanı (fc) çıkarıp kendi embedding katmanımızı ekliyoruz
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
        
        # 2. Conditional Masks
        self.masks = nn.Parameter(torch.rand(num_categories, num_categories, embedding_dim))
        
        # Maskeleri initialize et
        nn.init.xavier_uniform_(self.masks)

    def get_general_embedding(self, x):
        """Görseli genel uzaya gömer"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, anchor_img, pos_img, neg_img, anchor_type, pos_type, neg_type):
        # 1. Genel Embeddingler
        emb_anchor = self.get_general_embedding(anchor_img)
        emb_pos = self.get_general_embedding(pos_img)
        emb_neg = self.get_general_embedding(neg_img)
        
        # 2. Maskeleri Seç
        mask_ap = self.masks[anchor_type, pos_type] 
        mask_an = self.masks[anchor_type, neg_type]

        # 3. Embeddingleri Maskele
        proj_anchor_p = emb_anchor * mask_ap
        proj_pos = emb_pos * mask_ap
        
        proj_anchor_n = emb_anchor * mask_an
        proj_neg = emb_neg * mask_an
        
        return (proj_anchor_p, proj_pos, proj_anchor_n, proj_neg, 
                emb_anchor, emb_pos, emb_neg)