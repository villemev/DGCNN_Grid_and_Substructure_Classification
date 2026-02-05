import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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

# Load data from JSON file
def load_points_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    rows = []
    for struct_idx, struct in enumerate(data.get('structures', [])):
        # Get structure type (e.g., "fan" for curved, otherwise linear grids)
        struct_type = struct.get('structure_type', 'grid')
        for pt in struct['points']:
            u_val = pt.get('u', -1)
            v_val = pt.get('v', -1)
            rows.append({
                'structure': struct_idx,
                'u': u_val,
                'v': v_val,
                'x_mm': pt['x_mm'],
                'y_mm': pt['y_mm'],
                'structure_type': struct_type
            })
    return pd.DataFrame(rows)

# Group points by structure for separate u/v classification
def group_by_structure_separate(df):
    grouped = []
    for struct_id, group in df.groupby('structure'):
        X = group[['x_norm', 'y_norm']].values.astype(np.float32)  # Use normalized columns
        y_u = group['u_shifted'].values  # Use shifted labels (includes class 0 for outliers)
        y_v = group['v_shifted'].values  # Use shifted labels (includes class 0 for outliers)
        struct_type = group['structure_type'].iloc[0] if 'structure_type' in group.columns else 'grid'
        grouped.append((torch.tensor(X), torch.tensor(y_u), torch.tensor(y_v), struct_id, struct_type))
    return grouped

class StructureSetDatasetSeparate(Dataset):
    def __init__(self, grouped):
        self.grouped = grouped
    def __len__(self):
        return len(self.grouped)
    def __getitem__(self, idx):
        X, y_u, y_v, struct_id, struct_type = self.grouped[idx]
        return X, y_u, y_v, struct_id, struct_type

def collate_fn_separate(batch):
    Xs, y_us, y_vs, struct_ids, struct_types = zip(*batch)
    
    # Shuffle points within each structure for permutation invariance
    shuffled_data = []
    for X, y_u, y_v in zip(Xs, y_us, y_vs):
        n = X.shape[0]
        perm = torch.randperm(n)
        shuffled_data.append((X[perm], y_u[perm], y_v[perm]))
    
    Xs, y_us, y_vs = zip(*shuffled_data)
    
    lens = [x.shape[0] for x in Xs]
    Xs_padded = pad_sequence(Xs, batch_first=True)  # [B, max_n_points, 2]
    y_us_padded = pad_sequence(y_us, batch_first=True, padding_value=-1)  # [B, max_n_points]
    y_vs_padded = pad_sequence(y_vs, batch_first=True, padding_value=-1)  # [B, max_n_points]
    mask = torch.arange(Xs_padded.shape[1])[None, :] < torch.tensor(lens)[:, None]
    return Xs_padded, y_us_padded, y_vs_padded, mask, struct_ids, struct_types

# Shared building blocks for the DGCNN encoder

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



# Helper: Masked kNN computation with adaptive k per structure
def masked_knn_idx_adaptive(feat, mask, k_base=8, k_max=20):
    """
    Compute masked kNN indices with PER-STRUCTURE adaptive k.
    k scales with structure size: small grids use fewer neighbors, large grids use more.
    
    Args:
        feat: [B, N, D] feature tensor
        mask: [B, N] boolean mask (True = valid point)
        k_base: int, base number of neighbors (default: 8)
        k_max: int, maximum number of neighbors (default: 20)
    
    Returns:
        idx: [B, N, k_max] neighbor indices (padded with zeros if needed)
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
    
    # Adaptive k per structure: scale k with structure size
    valid_counts = mask.sum(dim=1)                       # [B] - points per structure
    
    # For each structure, compute adaptive k
    # Small grids (8-50 points): k = 8-12
    # Medium grids (50-200 points): k = 12-16
    # Large grids (200-600 points): k = 16-20
    k_per_struct = torch.clamp(
        (valid_counts.float() / 50.0).sqrt() * k_base,  # Scale by sqrt(size/50)
        min=k_base, 
        max=k_max
    ).long()  # [B]
    
    # Use maximum k for consistent tensor shapes
    k_global = min(k_max, max(1, int(valid_counts.min().item()) - 1))
    
    # Top-k nearest neighbors with global k
    idx = torch.topk(-dist2, k=k_global, dim=-1).indices    # [B, N, k_global]
    
    # Mask out neighbors beyond each structure's adaptive k
    # For each structure, set neighbors [k_adaptive:k_global] to 0 (first valid point)
    for b in range(B):
        k_struct = min(k_per_struct[b].item(), k_global)
        if k_struct < k_global:
            # Set excess neighbors to first valid point (index 0) - they'll be zeroed by mask
            first_valid = torch.where(mask[b])[0][0]
            idx[b, :, k_struct:] = first_valid
    
    # Pad to k_max if needed for consistent shapes across batches
    if k_global < k_max:
        pad = torch.zeros(B, N, k_max - k_global, dtype=torch.long, device=device)
        idx = torch.cat([idx, pad], dim=-1)              # [B, N, k_max]
    
    return idx

# Lightweight Multi-Head Attention for Global Context
class LightweightAttention(nn.Module):
    """Efficient multi-head attention to capture global relationships between points."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask):
        """
        Args:
            x: [B, N, dim] features
            mask: [B, N] valid point mask
        Returns:
            out: [B, N, dim] attended features
        """
        B, N, C = x.shape
        
        # Generate Q, K, V in one shot
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores: [B, heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Mask out invalid points (set attention to -inf)
        attn_mask = mask[:, None, None, :] & mask[:, None, :, None]  # [B, 1, N, N]
        attn = attn.masked_fill(~attn_mask, float('-inf'))
        
        # Softmax over last dimension (attending TO other points)
        attn = F.softmax(attn, dim=-1)
        
        # Handle NaN from all-masked rows (set to 0)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        
        # Weighted sum: [B, heads, N, head_dim] -> [B, N, dim]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out(out)
        
        # Zero out invalid points
        out = out * mask.unsqueeze(-1).float()
        
        return out

# Dynamic EdgeConv block: recomputes kNN using current features with adaptive k
class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_base=8, k_max=20, use_groupnorm=True):
        super().__init__()
        self.k_base = k_base
        self.k_max = k_max
        
        # MLP with GroupNorm for stability
        self.mlp = MLP([in_channels * 2, out_channels, out_channels], use_groupnorm=use_groupnorm)
    
    def forward(self, x, mask):
        """
        Args:
            x: [B, N, D] features
            mask: [B, N] boolean mask
        Returns:
            out: [B, N, out_channels] with masked points zeroed
        """
        # Compute kNN using current features
        idx = masked_knn_idx_adaptive(x, mask, self.k_base, self.k_max)  # [B, N, k]
        
        B, N, D = x.shape
        k = idx.shape[-1]
        
        # Gather neighbor features
        x_neighbors = torch.gather(
            x.unsqueeze(2).expand(-1, -1, k, -1),
            1,
            idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # [B, N, k, D]
        
        # Edge features: [point, neighbor - point]
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, k, D]
        edge_feat = torch.cat([x_i, x_neighbors - x_i], dim=-1)  # [B, N, k, 2*D]
        
        # Apply MLP to edge features and max pool
        edge_feat = edge_feat.reshape(B * N * k, -1)  # [B*N*k, 2*D]
        edge_feat = self.mlp(edge_feat)  # [B*N*k, out_channels]
        edge_feat = edge_feat.reshape(B, N, k, -1)  # [B, N, k, out_channels]
        out, _ = edge_feat.max(dim=2)  # [B, N, out_channels]
        
        # Zero out invalid points
        out = out * mask.unsqueeze(-1).float()
        
        return out


# Main DGCNN model with separate U/V classification heads
class DGCNNSeparateUV(nn.Module):
    def __init__(self, n_u, n_v, k_base=8, k_max=20, hidden=64):
        super().__init__()
        self.n_u = n_u
        self.n_v = n_v
        self.hidden = hidden
        
        # 4 EdgeConv layers with increasing channels
        self.ec1 = EdgeConvBlock(2, hidden, k_base, k_max)          # 2 -> 64
        self.ec2 = EdgeConvBlock(hidden, hidden*2, k_base, k_max)   # 64 -> 128
        self.ec3 = EdgeConvBlock(hidden*2, hidden*4, k_base, k_max) # 128 -> 256
        self.ec4 = EdgeConvBlock(hidden*4, hidden*4, k_base, k_max) # 256 -> 256
        
        # Multi-head attention for global context (helps with globally shifted grids)
        self.attention = LightweightAttention(hidden*4, num_heads=4)
        
        # Global pooling after attention
        self.global_fc = nn.Sequential(
            nn.Linear(hidden*4, hidden*4),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale feature fusion: 64 + 128 + 256 + 256 + 256 = 960
        fused_dim = hidden + hidden*2 + hidden*4 + hidden*4 + hidden*4
        
        # Separate classification heads with GroupNorm and stronger regularization
        self.u_head = MLP([fused_dim, 256, 128, n_u], use_groupnorm=True)
        self.v_head = MLP([fused_dim, 256, 128, n_v], use_groupnorm=True)
        
        self.dropout = nn.Dropout(0.3)  # Increased from 0.2 for better generalization

    def forward(self, x, mask):
        """
        Args:
            x: [B, N, 2] input coordinates
            mask: [B, N] boolean mask
        
        Returns:
            u_logits: [B, N, n_u]
            v_logits: [B, N, n_v]
        """
        B, N, _ = x.shape
        
        # Progressive EdgeConv with adaptive k per structure
        h1 = self.ec1(x, mask)                              # [B, N, 64]
        h2 = self.ec2(h1, mask)                             # [B, N, 128]
        h3 = self.ec3(h2, mask)                             # [B, N, 256]
        h4 = self.ec4(h3, mask)                             # [B, N, 256]
        
        # Multi-head attention: each point attends to all others
        h4_attended = self.attention(h4, mask)              # [B, N, 256]
        
        # Global context from attention-weighted features
        h4_masked = h4_attended.clone()
        h4_masked[~mask] = 0.0
        g = h4_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()  # Mean pooling [B, 256]
        g = self.global_fc(g)                               # [B, 256]
        g = g.unsqueeze(1).expand(-1, N, -1)                # [B, N, 256]
        
        # Multi-scale feature fusion
        feat = torch.cat([h1, h2, h3, h4, g], dim=-1)       # [B, N, 960]
        feat = self.dropout(feat)
        
        # Reshape for MLP processing: [B, N, C] -> [B*N, C]
        B, N, C = feat.shape
        feat_flat = feat.reshape(B * N, C)                  # [B*N, 960]
        
        # Separate u and v predictions
        u_logits_flat = self.u_head(feat_flat)              # [B*N, n_u]
        v_logits_flat = self.v_head(feat_flat)              # [B*N, n_v]
        
        # Reshape back to [B, N, n_classes]
        u_logits = u_logits_flat.reshape(B, N, self.n_u)    # [B, N, n_u]
        v_logits = v_logits_flat.reshape(B, N, self.n_v)    # [B, N, n_v]
        
        return u_logits, v_logits


class UniquenessLoss(nn.Module):
    """
    Loss component that penalizes duplicate (u,v) predictions within a structure.
    
    ONLY penalizes duplicate GRID POINTS (u>0, v>0).
    Allows multiple points with:
    - Same u-axis alignment (u>0, v=0)
    - Same v-axis alignment (u=0, v>0)
    - Same outlier status (u=0, v=0)
    """
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, u_logits, v_logits, mask):
        """
        Args:
            u_logits: [B, N, n_u] - logits for u predictions
            v_logits: [B, N, n_v] - logits for v predictions  
            mask: [B, N] - boolean mask for valid points
        
        Returns:
            loss: scalar - penalty for duplicate GRID POINT predictions
        """
        B, N, n_u = u_logits.shape
        n_v = v_logits.shape[-1]
        
        # Get predictions
        u_pred = torch.argmax(u_logits, dim=-1)  # [B, N]
        v_pred = torch.argmax(v_logits, dim=-1)  # [B, N]
        
        total_penalty = 0.0
        
        for b in range(B):
            # Get valid points in this structure
            valid_mask = mask[b]
            n_valid = valid_mask.sum().item()
            
            if n_valid <= 1:
                continue
            
            # Get predictions for valid points
            u_valid = u_pred[b, valid_mask]  # [n_valid]
            v_valid = v_pred[b, valid_mask]  # [n_valid]
            
            # Identify grid points (both u>0 and v>0)
            # Remember: 0 represents -1 (outlier/axis) after shifting
            grid_point_mask = (u_valid > 0) & (v_valid > 0)  # [n_valid]
            
            n_grid = grid_point_mask.sum().item()
            if n_grid <= 1:
                continue  # No duplicates possible with 0 or 1 grid points
            
            # Get grid point predictions only
            u_grid = u_valid[grid_point_mask]  # [n_grid]
            v_grid = v_valid[grid_point_mask]  # [n_grid]
            
            # Create matrix of pairwise (u,v) equality for grid points only
            u_match = u_grid.unsqueeze(0) == u_grid.unsqueeze(1)  # [n_grid, n_grid]
            v_match = v_grid.unsqueeze(0) == v_grid.unsqueeze(1)  # [n_grid, n_grid]
            both_match = u_match & v_match  # [n_grid, n_grid]
            
            # Remove diagonal (point matches itself)
            both_match = both_match & ~torch.eye(n_grid, dtype=torch.bool, device=u_logits.device)
            
            # Count duplicates (each duplicate pair counted twice, so divide by 2)
            n_duplicates = both_match.sum().float() / 2.0
            
            # Penalty proportional to number of duplicates
            total_penalty += n_duplicates
        
        # Normalize by batch size
        avg_penalty = total_penalty / B
        
        return self.weight * avg_penalty


