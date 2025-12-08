"""
Metric Learning Module with Hard Triplet Mining
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from tqdm import tqdm

from src import fclip_config as config


class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""

    def __init__(self, margin: float = config.TRIPLET_MARGIN):
        """
        Initialize triplet loss

        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss

        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)

        Returns:
            Triplet loss value
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)

        loss = torch.mean(
            torch.clamp(
                distance_positive - distance_negative + self.margin,
                min=0.0
            )
        )

        return loss


class OnlineTripletMiner:
    """Online Semi-Hard Triplet Miner"""

    def __init__(self, margin: float = config.TRIPLET_MARGIN):
        self.margin = margin

    def mine(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine semi-hard triplets from batch

        Args:
            embeddings: Batch of embeddings (batch_size, dim)
            labels: Batch of labels (batch_size,) - Items with same label are positives

        Returns:
            Tuple of (anchors, positives, negatives)
        """
        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(embeddings, embeddings)

        # Create mask for same-label pairs (positives)
        # labels: (B,) -> (B, 1) == (1, B) -> (B, B)
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Mask out self-distances (diagonal)
        mask_self = torch.eye(label_equal.shape[0], dtype=torch.bool, device=embeddings.device)
        label_equal = label_equal & ~mask_self

        triplets_a = []
        triplets_p = []
        triplets_n = []

        for i in range(embeddings.shape[0]):
            # Find positives for anchor i
            pos_indices = torch.where(label_equal[i])[0]
            if len(pos_indices) == 0:
                continue

            # For each positive, find semi-hard negatives
            for p_idx in pos_indices:
                d_ap = dist_matrix[i, p_idx]

                # Negatives are items with different labels
                neg_indices = torch.where(~label_equal[i] & ~mask_self[i])[0]

                if len(neg_indices) == 0:
                    continue

                d_an = dist_matrix[i, neg_indices]

                # Semi-hard condition: d_ap < d_an < d_ap + margin
                semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + self.margin)

                # If no semi-hard, fallback to hardest negative (d_an > d_ap but minimal)
                # or just hard negatives (d_an < d_ap) if we want to push them apart
                # Standard semi-hard mining usually prioritizes the semi-hard region.

                valid_negatives = neg_indices[semi_hard_mask]

                if len(valid_negatives) > 0:
                    # Pick a random semi-hard negative
                    n_idx = valid_negatives[torch.randint(len(valid_negatives), (1,)).item()]

                    triplets_a.append(embeddings[i])
                    triplets_p.append(embeddings[p_idx])
                    triplets_n.append(embeddings[n_idx])

        if not triplets_a:
            return None, None, None

        return torch.stack(triplets_a), torch.stack(triplets_p), torch.stack(triplets_n)


class HardTripletMiner:
    """Hard triplet mining for metric learning (GPU Optimized)"""

    def __init__(
            self,
            margin: float = config.TRIPLET_MARGIN,
            num_hard_negatives: int = config.NUM_HARD_NEGATIVES,
            num_hard_positives: int = config.NUM_HARD_POSITIVES,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize hard triplet miner

        Args:
            margin: Margin for hard negative mining
            num_hard_negatives: Number of hard negatives to mine per anchor
            num_hard_positives: Number of hard positives to mine per anchor
            device: Device to use for computation
        """
        self.margin = margin
        self.num_hard_negatives = num_hard_negatives
        self.num_hard_positives = num_hard_positives
        self.device = device

    def mine_hard_triplets(
            self,
            embeddings: np.ndarray,
            item_ids: List[str],
            outfit_data: List[Dict],
            metadata: Dict,
            batch_size: int = 1000  # Chunk size for mining to fit in VRAM
    ) -> List[Tuple[int, int, int]]:
        """
        Mine hard triplets from the dataset using efficient chunked matrix operations on GPU.
        Avoids OOM by processing anchors in batches.

        Args:
            embeddings: Item embeddings (num_items, embedding_dim)
            item_ids: List of item IDs
            outfit_data: List of outfit dictionaries
            metadata: Item metadata dictionary
            batch_size: Number of anchors to process at once on GPU

        Returns:
            List of (anchor_idx, positive_idx, negative_idx) triplets
        """
        print(f"Mining hard triplets on {self.device} with chunk size {batch_size}...")
        
        num_items = len(item_ids)
        embeddings_tensor = torch.from_numpy(embeddings).to(self.device).float()
        
        # 1. Build Compatibility Map (CPU) - sparse representation
        # item_idx -> list of compatible item_indices
        print("Building compatibility map...")
        id_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
        compatibility_map = defaultdict(list)
        
        for outfit in outfit_data:
            indices = []
            for item in outfit.get('items', []):
                item_id = item.get('item_id')
                if item_id and item_id in id_to_idx:
                    indices.append(id_to_idx[item_id])
            
            # Add all pairs
            for idx1 in indices:
                for idx2 in indices:
                    if idx1 != idx2:
                        compatibility_map[idx1].append(idx2)
        
        # Optimizing map: remove duplicates
        for k in compatibility_map:
            compatibility_map[k] = list(set(compatibility_map[k]))
            
        triplets = []
        print("Selecting hard triplets (chunked)...")
        
        # 2. Process in Chunks
        # We iterate over anchors in batches
        for start_idx in tqdm(range(0, num_items, batch_size), desc="Mining chunks"):
            end_idx = min(start_idx + batch_size, num_items)
            current_batch_size = end_idx - start_idx
            
            # Get anchor embeddings for this batch
            anchor_batch = embeddings_tensor[start_idx:end_idx] # (B, Dim)
            
            # Compute distances: Anchors (Batch) vs All Items (N)
            # Result shape: (B, N)
            # Memory: 1000 * 72000 * 4 bytes ≈ 288 MB (Safe)
            dist_batch = torch.cdist(anchor_batch, embeddings_tensor, p=2)
            
            # Create boolean mask for compatibility (B, N)
            # Initialize with False
            pos_mask = torch.zeros((current_batch_size, num_items), dtype=torch.bool, device=self.device)
            
            # Fill mask
            # We can't vectorizing filling from list of lists easily without padding, 
            # but standard iteration for 1000 items is fast enough.
            # Or construct generic sparse tensor? Simple loop is likely fine for 1000 items.
            for i in range(current_batch_size):
                global_idx = start_idx + i
                compat_indices = compatibility_map.get(global_idx, [])
                if compat_indices:
                    pos_mask[i, compat_indices] = True
            
            # Prepare distance matrices for mining
            # Negatives: Mask is False
            neg_dist = dist_batch.clone()
            neg_dist[pos_mask] = float('inf')
            
            # Mask self-distance (diagonal in global matrix, but here it's different)
            # For each expected anchor row i, the self index is start_idx + i
            # Vectorized self-masking
            self_indices = torch.arange(start_idx, end_idx, device=self.device)
            batch_indices = torch.arange(current_batch_size, device=self.device)
            neg_dist[batch_indices, self_indices] = float('inf')
            
            # Find Hard Negatives (closest non-compatible items)
            # Top-K smallest
            # We ask for K items, but some rows might have all Infs if N is small (unlikely here)
            _, neg_indices = torch.topk(neg_dist, k=self.num_hard_negatives, dim=1, largest=False)
            
            # Move indices to CPU for triplet construction
            neg_indices_cpu = neg_indices.cpu().numpy()
            
            # Collect triplets
            # For each anchor in batch
            for i in range(current_batch_size):
                global_idx = start_idx + i
                
                # Get positives
                p_indices = compatibility_map.get(global_idx, [])
                if not p_indices:
                    continue
                
                # Get hard negatives for this anchor
                current_neg_indices = neg_indices_cpu[i]
                
                # Filter valid negatives (distance not Inf)
                # Check directly from dist tensor or just assume topk returned valid if N large
                # If N is large, we always find K negatives.
                
                # Subsample positives if too many
                if len(p_indices) > self.num_hard_positives:
                    selected_p = np.random.choice(p_indices, self.num_hard_positives, replace=False)
                else:
                    selected_p = p_indices
                
                for p_idx in selected_p:
                    for n_idx in current_neg_indices:
                        if neg_dist[i, n_idx] == float('inf'):
                            continue # Skip invalid
                        triplets.append((global_idx, int(p_idx), int(n_idx)))
            
            # Clean up memory
            del dist_batch, pos_mask, neg_dist, anchor_batch
            
        print(f"Mined {len(triplets)} hard triplets")
        torch.cuda.empty_cache()
        return triplets

    def get_triplet_batch(
            self,
            triplets: List[Tuple[int, int, int]],
            embeddings: np.ndarray,
            batch_size: int = config.BATCH_SIZE
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a batch of triplets for training
        """
        # Sample random triplets
        indices = np.random.choice(len(triplets), size=min(batch_size, len(triplets)), replace=False)
        selected_triplets = [triplets[i] for i in indices]

        anchor_indices = [t[0] for t in selected_triplets]
        positive_indices = [t[1] for t in selected_triplets]
        negative_indices = [t[2] for t in selected_triplets]

        anchor_batch = torch.FloatTensor(embeddings[anchor_indices])
        positive_batch = torch.FloatTensor(embeddings[positive_indices])
        negative_batch = torch.FloatTensor(embeddings[negative_indices])

        return anchor_batch, positive_batch, negative_batch


class EmbeddingProjection(nn.Module):
    """Neural network to project embeddings to a learned metric space"""

    def __init__(self, input_dim: int = config.EMBEDDING_DIM, output_dim: int = 512):
        """
        Initialize embedding projection network

        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
        """
        super(EmbeddingProjection, self).__init__()

        self.linear1 = nn.Linear(input_dim, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with L2 normalization"""
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        # L2 normalize output embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

