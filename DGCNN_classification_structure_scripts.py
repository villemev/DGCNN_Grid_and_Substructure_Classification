import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.optimize import linear_sum_assignment
import hashlib


# Environment setup

# DGCNN utilities and fallback implementations
def _median_nn_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    diff = points[:, None, :] - points[None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(axis=1))
    med = float(np.median(nn))
    return med if med > 0 else 1.0

def normalize_scene(points: np.ndarray) -> np.ndarray:
    cen = points.mean(axis=0, keepdims=True)
    p = points - cen
    scale = _median_nn_distance(p)
    return p / scale

# Load multi-structure data from JSON file
def load_multistructure_points_from_json(json_path):
    """Load points from multi-structure JSON with substructure_id labels"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for struct_idx, struct in enumerate(data.get('structures', [])):
        if 'num_substructures' not in struct or struct['num_substructures'] < 1:
            continue
            
        for pt in struct['points']:
            substructure_id = pt.get('substructure_id', 0)
            rows.append({
                'structure': struct_idx,
                'substructure_id': substructure_id,
                'u': pt['u'],
                'v': pt['v'],  
                'x_mm': pt['x_mm'],
                'y_mm': pt['y_mm']
            })
    return pd.DataFrame(rows)

## Data loading and preprocessing

# Group points by structure for substructure classification
def group_by_structure_substructure(df):
    """Group points by structure and prepare for substructure classification"""
    grouped = []
    for struct_id, group in df.groupby('structure'):
        X = group[['x_norm', 'y_norm']].values.astype(np.float32)
        y_sub = group['substructure_id'].values
        grouped.append((torch.tensor(X), torch.tensor(y_sub, dtype=torch.long), struct_id))
    return grouped



class StructureSetDatasetSubstructure(Dataset):
    def __init__(self, grouped):
        self.grouped = grouped
    def __len__(self):
        return len(self.grouped)
    def __getitem__(self, idx):
        X, y_sub, struct_id = self.grouped[idx]
        return X, y_sub, struct_id

def collate_fn_substructure(batch):
    """Collate function for substructure classification"""
    Xs, y_subs, struct_ids = zip(*batch)
    lens = [x.shape[0] for x in Xs]
    Xs_padded = pad_sequence(Xs, batch_first=True)
    y_subs_padded = pad_sequence(y_subs, batch_first=True, padding_value=-1)
    mask = torch.arange(Xs_padded.shape[1])[None, :] < torch.tensor(lens)[:, None]
    return Xs_padded, y_subs_padded, mask, struct_ids

# Shared MLP building block for encoder modules
class MLP(nn.Module):
    def __init__(self, channels, use_groupnorm: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i+1]))
            if i < len(channels) - 2:
                if use_groupnorm:
                    # GroupNorm for stability with small batches
                    num_groups = min(32, channels[i+1] // 4) if channels[i+1] >= 4 else 1
                    layers.append(nn.GroupNorm(num_groups, channels[i+1]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Helper: Masked kNN computation with adaptive k and proper padding
def masked_knn_idx_safe(feat, mask, k):
    """
    Compute masked kNN indices with adaptive k handling.
    
    Args:
        feat: [B, N, D] feature tensor
        mask: [B, N] boolean mask (True = valid point)
        k: int, desired number of neighbors
    
    Returns:
        idx: [B, N, k] neighbor indices (padded with zeros if needed)
    """
    B, N, D = feat.shape
    device = feat.device
    
    # Compute pairwise distances
    x2 = (feat ** 2).sum(dim=-1, keepdim=True)           # [B, N, 1]
    y2 = x2.transpose(1, 2)                              # [B, 1, N]
    xy = feat @ feat.transpose(1, 2)                     # [B, N, N]
    dist2 = x2 + y2 - 2 * xy                             # [B, N, N]
    dist2 = torch.clamp(dist2, min=0.0)

    # Mask invalid points: set distance to +inf where mask=False
    big = torch.full_like(dist2, float('inf'))
    row_mask = mask.unsqueeze(2)                         # [B, N, 1]
    col_mask = mask.unsqueeze(1)                         # [B, 1, N]
    valid_matrix = row_mask & col_mask                   # [B, N, N]
    dist2 = torch.where(valid_matrix, dist2, big)

    # Avoid self as nearest neighbor (diagonal = +inf)
    eye = torch.eye(N, device=device).bool().unsqueeze(0)
    dist2 = torch.where(eye, big, dist2)
    
    # Adaptive k: reduce k if fewer valid points available
    valid_counts = mask.sum(dim=1)                       # [B]
    k_eff = min(k, max(1, int(valid_counts.min().item()) - 1))
    
    # Top-k nearest neighbors
    idx = torch.topk(-dist2, k=k_eff, dim=-1).indices    # [B, N, k_eff]
    
    # Pad to fixed k if k_eff < k (for consistent tensor shapes)
    if k_eff < k:
        pad = torch.zeros(B, N, k - k_eff, dtype=torch.long, device=device)
        idx = torch.cat([idx, pad], dim=-1)              # [B, N, k]
    
    return idx

def local_frame_directional_idx_knn_topm(
    x: torch.Tensor,
    mask: torch.Tensor,
    k_base: int = 12,
    m_per_dir: int = 2,   # 2 => 8 edges total
) -> torch.Tensor:
    """
    Direction-aware neighbors using ONLY kNN candidates.
    Picks top-m in each direction (+t, -t, +n, -n).
    Returns [B,N,4*m_per_dir].
    """
    B, N, _ = x.shape
    device = x.device

    idx_k = masked_knn_idx_safe(x, mask, k_base)                 # [B,N,k]
    b_idx = torch.arange(B, device=device)[:, None, None]

    neigh = x[b_idx, idx_k]                                      # [B,N,k,2]
    ctr   = x.unsqueeze(2)                                       # [B,N,1,2]
    d     = neigh - ctr                                          # [B,N,k,2]

    dx = d[..., 0]; dy = d[..., 1]
    C11 = (dx * dx).sum(dim=2)
    C22 = (dy * dy).sum(dim=2)
    C12 = (dx * dy).sum(dim=2)

    theta = 0.5 * torch.atan2(2 * C12, (C11 - C22))
    t = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)   # [B,N,2]
    n = torch.stack([-t[..., 1], t[..., 0]], dim=-1)                # [B,N,2]

    a = (d * t.unsqueeze(2)).sum(dim=-1)   # [B,N,k] along tangent
    b = (d * n.unsqueeze(2)).sum(dim=-1)   # [B,N,k] across normal

    # top-m in each direction among k candidates
    # forward: largest a
    fwd_local = torch.topk(a, k=m_per_dir, dim=2).indices
    # back: most negative a => largest (-a)
    back_local = torch.topk(-a, k=m_per_dir, dim=2).indices
    # left: largest b
    left_local = torch.topk(b, k=m_per_dir, dim=2).indices
    # right: most negative b => largest (-b)
    right_local = torch.topk(-b, k=m_per_dir, dim=2).indices

    # map local indices -> global point indices
    fwd   = idx_k.gather(2, fwd_local)
    back  = idx_k.gather(2, back_local)
    left  = idx_k.gather(2, left_local)
    right = idx_k.gather(2, right_local)

    idx_dir = torch.cat([fwd, back, left, right], dim=2)          # [B,N,4*m]
    return idx_dir


class DirectionalEdgeConvBlock(nn.Module):
    """Directional EdgeConv over k_dir = 4*m_per_dir neighbors selected from kNN candidates."""

    def __init__(self, in_channels, out_channels, k_base=12, m_per_dir=2, use_groupnorm=True):
        super().__init__()
        self.k_base = k_base
        self.m_per_dir = m_per_dir
        self.k_dir = 4 * m_per_dir
        self.mlp = MLP([in_channels * 2, out_channels, out_channels], use_groupnorm=use_groupnorm)

    def forward(self, x, mask, geom_xy):
        B, N, _ = x.shape
        device = x.device

        idx = local_frame_directional_idx_knn_topm(
            geom_xy, mask, k_base=self.k_base, m_per_dir=self.m_per_dir
        )

        batch_idx = torch.arange(B, device=device)[:, None, None]
        x_neighbors = x[batch_idx, idx, :]
        x_center = x.unsqueeze(2).expand(-1, -1, self.k_dir, -1)

        edge_feat = torch.cat([x_neighbors - x_center, x_center], dim=-1)
        edge_flat = edge_feat.reshape(B * N * self.k_dir, -1)
        out_flat = self.mlp(edge_flat)
        out = out_flat.reshape(B, N, self.k_dir, -1).max(dim=2).values

        return out * mask.unsqueeze(-1).float()


class EdgeConvBlock(nn.Module):
    """EdgeConv block that recomputes kNN graph from current features."""

    def __init__(self, in_channels, out_channels, k=8, use_groupnorm=True):
        super().__init__()
        self.k = k
        self.mlp = MLP([in_channels * 2, out_channels, out_channels], use_groupnorm=use_groupnorm)

    def forward(self, x, mask):
        B, N, _ = x.shape
        idx = masked_knn_idx_safe(x, mask, self.k)
        batch_idx = torch.arange(B, device=x.device)[:, None, None]
        x_neighbors = x[batch_idx, idx, :]
        x_center = x.unsqueeze(2).expand(-1, -1, self.k, -1)
        edge_feat = torch.cat([x_neighbors - x_center, x_center], dim=-1)
        edge_flat = edge_feat.reshape(B * N * self.k, -1)
        out_flat = self.mlp(edge_flat)
        out = out_flat.reshape(B, N, self.k, -1).max(dim=2).values
        out = out * mask.unsqueeze(-1).float()
        return out

class DGCNNSubstructure(nn.Module):
    def __init__(self, n_substructures: int = 3, k: int = 12, hidden: int = 64):
        super().__init__()
        self.k = k
        self.n_substructures = n_substructures

        # Stage 1: boundary discovery
        self.ec1 = EdgeConvBlock(2, hidden, k=k)
        self.ec2 = EdgeConvBlock(hidden, hidden*2, k=k)

        # Stage 2: directional refinement (ONLY ec3)
        self.ec3 = DirectionalEdgeConvBlock(hidden*2, hidden*4, k_base=k, m_per_dir=2)

        # Stage 3: keep flexibility (ec4 standard)
        self.ec4 = EdgeConvBlock(hidden*4, hidden*4, k=k)


        self.global_fc = nn.Sequential(
            nn.Linear(hidden*4, hidden*4),
            nn.ReLU(inplace=True)
        )

        fused_dim = hidden + hidden*2 + hidden*4 + hidden*4 + hidden*4
        self.substructure_head = MLP([fused_dim, 512, 256, 128, n_substructures], use_groupnorm=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask):
        geom_xy = x

        h1 = self.ec1(x, mask)
        h2 = self.ec2(h1, mask)
        h3 = self.ec3(h2, mask, geom_xy)
        h4 = self.ec4(h3, mask)

        h4_masked = h4.clone()
        h4_masked[~mask] = float('-inf')
        g, _ = h4_masked.max(dim=1, keepdim=True)
        g = self.global_fc(g.squeeze(1)).unsqueeze(1).expand(-1, x.size(1), -1)

        feat = torch.cat([h1, h2, h3, h4, g], dim=-1)
        feat = self.dropout(feat)

        B, N, C = feat.shape
        sub_logits = self.substructure_head(feat.reshape(B * N, C)).reshape(B, N, self.n_substructures)
        return sub_logits


## DGCNN for substructure classification with hungarian alignment

# Hungarian Alignment Helper Functions for Substructure Classification
# 
# Since ground-truth substructure IDs are local per structure (any permutation is valid),
# we must align predicted class channels to GT IDs per structure before computing loss.
# This ensures the model learns groupings, not arbitrary numeric labels.

def _build_negative_iou_cost(pred_labels_np: np.ndarray,
                             gt_labels_np: np.ndarray,
                             K: int,
                             G: int) -> np.ndarray:
    """
    Build cost[k,g] = -IoU between predicted cluster k and GT cluster g.
    
    Args:
        pred_labels_np: [Nv] argmax predictions in 0..K-1
        gt_labels_np:   [Nv] local GT ids in 0..G-1
        K: number of predicted classes (model output channels)
        G: number of GT groups in this structure
    
    Returns:
        cost: [K, G] negative IoU cost matrix
    """
    cost = np.zeros((K, G), dtype=np.float32)
    for k in range(K):
        pk = (pred_labels_np == k)
        for g in range(G):
            yg = (gt_labels_np == g)
            inter = np.sum(pk & yg)
            union = np.sum(pk | yg)
            iou = (inter / union) if union > 0 else 0.0
            cost[k, g] = -iou
    return cost

def hungarian_remap_indices_for_structure(logits_b: torch.Tensor,
                                          y_b: torch.Tensor,
                                          mask_b: torch.Tensor) -> torch.Tensor:
    """
    Compute a permutation of class channels that best matches GT groups (Hungarian algorithm).
    
    Args:
        logits_b: [N, K] logits for one structure
        y_b: [N] GT labels for one structure
        mask_b: [N] validity mask for one structure
    
    Returns:
        remap: [K] tensor where remap[k_pred] = k_gt (0..K-1)
               Unmatched predicted classes map to themselves.
    """
    valid = (mask_b & (y_b >= 0))
    K = logits_b.shape[-1]
    remap = torch.arange(K, device=logits_b.device)
    
    # If no valid points, return identity mapping
    if valid.sum().item() == 0:
        return remap

    x = logits_b[valid]                 # [Nv, K]
    y = y_b[valid]                      # [Nv]
    pred = x.argmax(dim=-1)             # [Nv]
    G = int(y.max().item() + 1)         # number of GT groups in this structure

    # Build -IoU cost matrix and solve assignment
    cost = _build_negative_iou_cost(pred.detach().cpu().numpy(),
                                    y.detach().cpu().numpy(),
                                    K, G)
    ri, ci = linear_sum_assignment(cost)  # ri: pred idx, ci: gt idx

    # Fill mapping for matched pairs
    for k_pred, k_gt in zip(ri, ci):
        remap[k_pred] = int(k_gt)
    
    return remap

def align_logits_with_hungarian(logits_b: torch.Tensor,
                                y_b: torch.Tensor,
                                mask_b: torch.Tensor) -> tuple:
    """
    Permute class channels of logits for ONE structure using Hungarian mapping.
    
    Args:
        logits_b: [N, K] logits for one structure
        y_b: [N] GT labels for one structure
        mask_b: [N] validity mask for one structure
    
    Returns:
        x_aligned: [Nv, K] aligned logits for valid points
        y_valid:   [Nv]    GT labels for valid points
    """
    valid = (y_b != -1) & mask_b
    if valid.sum().item() == 0:
        return logits_b.new_zeros((0, logits_b.shape[-1])), y_b.new_zeros((0,), dtype=torch.long)
    
    remap = hungarian_remap_indices_for_structure(logits_b, y_b, mask_b)  # [K]
    x_valid = logits_b[valid]                    # [Nv, K]
    x_aligned = x_valid[:, remap]                # [Nv, K] permuted channels
    y_valid = y_b[valid]                         # [Nv]
    
    return x_aligned, y_valid

## Training

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducible training runs"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Data versioning: Check if training data has changed

def get_data_hash(df, n_substructures):
    """Generate a hash of the training data to detect changes (includes class count)"""
    # Create a string representation sensitive to data AND class structure changes
    data_string = f"{len(df)}_{df['structure'].nunique()}_{n_substructures}_{df['substructure_id'].value_counts().to_dict()}"
    data_string += f"_{df[['x_mm', 'y_mm', 'substructure_id']].values.tobytes().hex()[:100]}"  # Sample of actual data
    return hashlib.md5(data_string.encode()).hexdigest()