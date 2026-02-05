# Environment check: Pure PyTorch DGCNN (no external dependencies)
import os
import random
import sys

import hashlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from DGCNN_classification_UV_scripts import (
    DGCNNSeparateUV,
    StructureSetDatasetSeparate,
    UniquenessLoss,
    collate_fn_separate,
    group_by_structure_separate,
    load_points_from_json,
    normalize_scene,
)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"NumPy: {np.__version__}")
print("Using pure PyTorch DGCNN implementation (no PyG or torch_cluster dependencies).")
print(f"Random seed locked to {SEED} for reproducibility")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory set to {os.getcwd()}")

df = load_points_from_json('JSON/UV.json')

# Model save directory
models_dir = 'trained_models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model_path = os.path.join(models_dir, 'UV.pth')
n_epochs = 10  # Set number of epochs for training

# Shift labels so -1 becomes 0, and grid points become 1 to n
# This allows the model to predict outliers (class 0)
df['u_shifted'] = df['u'] + 1  # -1 -> 0, 0 -> 1, 1 -> 2, etc.
df['v_shifted'] = df['v'] + 1  # -1 -> 0, 0 -> 1, 1 -> 2, etc.

n_u = df['u_shifted'].max() + 1  # Now includes class 0 for outliers
n_v = df['v_shifted'].max() + 1  # Now includes class 0 for outliers

print("Label shift applied to include class 0 for outliers")
print(f"Original u range: {df['u'].min()} to {df['u'].max()} | Shifted: {df['u_shifted'].min()} to {df['u_shifted'].max()}")
print(f"Original v range: {df['v'].min()} to {df['v'].max()} | Shifted: {df['v_shifted'].min()} to {df['v_shifted'].max()}")
print("Class 0 represents outliers; remaining classes map to grid points")

# --- Structure-based split ---
unique_structures = df['structure'].unique()
train_structs, test_structs = train_test_split(unique_structures, test_size=0.2, random_state=42)
train_df = df[df['structure'].isin(train_structs)].reset_index(drop=True)
test_df = df[df['structure'].isin(test_structs)].reset_index(drop=True)

# Normalize using scene-based normalization for DGCNN
print(f"Loaded {len(df)} points from {len(unique_structures)} structures")
print(f"Grid dimensions: {n_u} x {n_v} = {n_u * n_v} classes")
print(f"Train structures: {len(train_structs)}, test structures: {len(test_structs)}")

# Apply scene-based normalization
for struct_df in [train_df, test_df]:
    struct_df['x_norm'] = 0.0
    struct_df['y_norm'] = 0.0
    for struct_id in struct_df['structure'].unique():
        mask = struct_df['structure'] == struct_id
        pts = struct_df.loc[mask, ['x_mm', 'y_mm']].values
        pts_norm = normalize_scene(pts)
        struct_df.loc[mask, 'x_norm'] = pts_norm[:, 0]
        struct_df.loc[mask, 'y_norm'] = pts_norm[:, 1]

#Structure-wise input with padding

train_structs_grouped = group_by_structure_separate(train_df)
test_structs_grouped = group_by_structure_separate(test_df)

train_set = StructureSetDatasetSeparate(train_structs_grouped)
test_set = StructureSetDatasetSeparate(test_structs_grouped)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn_separate)
test_loader = DataLoader(test_set, batch_size=8, collate_fn=collate_fn_separate)

##Training architecture with multi head attention

# Training loop for DGCNN with separate u/v classification and uniqueness loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNNSeparateUV(n_u=n_u, n_v=n_v, k_base=8, k_max=20, hidden=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
uniqueness_criterion = UniquenessLoss(weight=0.1)

print("Using DGCNN with adaptive kNN, attention, and uniqueness loss")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"u classes: {n_u}, v classes: {n_v}")
print("Architecture: 4 EdgeConv layers, 4-head attention, multi-scale fusion")


def get_data_hash(df, n_u, n_v):
    data_string = f"{len(df)}_{df['structure'].nunique()}_{n_u}_{n_v}"
    data_string += f"_{df['u'].value_counts().to_dict()}_{df['v'].value_counts().to_dict()}"
    data_string += f"_{df[['x_mm', 'y_mm', 'u', 'v']].values.tobytes().hex()[:100]}"
    return hashlib.md5(data_string.encode()).hexdigest()

current_data_hash = get_data_hash(train_df, n_u, n_v)
print(f"Current training data hash: {current_data_hash}")

train_losses = []
val_losses = []
val_u_accuracies = []
val_v_accuracies = []
val_combined_accuracies = []

if os.path.exists(model_path):
    print(f"Found saved model at '{model_path}'")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_data_hash = checkpoint.get('data_hash', 'unknown')
    saved_n_u = checkpoint.get('n_u', 'unknown')
    saved_n_v = checkpoint.get('n_v', 'unknown')

    data_changed = saved_data_hash != current_data_hash
    classes_changed = (saved_n_u != n_u) or (saved_n_v != n_v)

    if data_changed:
        print("Training data changed since the last checkpoint; retraining from scratch.")
    if classes_changed:
        print(
            f"Class configuration changed (saved u/v={saved_n_u}/{saved_n_v}, current={n_u}/{n_v}); "
            "retraining with updated heads."
        )

    if not data_changed and not classes_changed:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_u_accuracies = checkpoint.get('val_u_accuracies', [])
        val_v_accuracies = checkpoint.get('val_v_accuracies', [])
        val_combined_accuracies = checkpoint.get('val_combined_accuracies', [])

        best_acc = max(val_combined_accuracies) if val_combined_accuracies else float('nan')
        final_acc = val_combined_accuracies[-1] if val_combined_accuracies else float('nan')
        print(
            f"Checkpoint matches current configuration; reusing weights. "
            f"Best combined accuracy {best_acc:.4f}, last accuracy {final_acc:.4f}."
        )
        n_epochs = 0
    else:
        print("Existing checkpoint is incompatible; starting a new training run.")
else:
    print(f"No saved model found at '{model_path}'. A new training run will be started.")

# Training loop
for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0
    train_points = 0
    
    for X_batch, y_u_batch, y_v_batch, mask, _, _ in train_loader:
        X_batch = X_batch.to(device)
        y_u_batch = y_u_batch.to(device)
        y_v_batch = y_v_batch.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        u_logits, v_logits = model(X_batch, mask)
        
        u_logits_flat = u_logits.view(-1, n_u)
        v_logits_flat = v_logits.view(-1, n_v)
        y_u_flat = y_u_batch.view(-1)
        y_v_flat = y_v_batch.view(-1)
        
        loss_u = criterion(u_logits_flat, y_u_flat)
        loss_v = criterion(v_logits_flat, y_v_flat)
        uniqueness_loss = uniqueness_criterion(u_logits, v_logits, mask)
        loss = loss_u + loss_v + uniqueness_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        valid_points = (y_u_flat != -1).sum().item()
        total_train_loss += loss.item() * valid_points
        train_points += valid_points
    
    avg_train_loss = total_train_loss / max(1, train_points)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    correct_u = correct_v = correct_combined = total = 0
    
    with torch.no_grad():
        for X_batch, y_u_batch, y_v_batch, mask, _, _ in test_loader:
            X_batch = X_batch.to(device)
            y_u_batch = y_u_batch.to(device)
            y_v_batch = y_v_batch.to(device)
            mask = mask.to(device)
            
            u_logits, v_logits = model(X_batch, mask)
            u_logits_flat = u_logits.view(-1, n_u)
            v_logits_flat = v_logits.view(-1, n_v)
            y_u_flat = y_u_batch.view(-1)
            y_v_flat = y_v_batch.view(-1)
            
            loss_u = criterion(u_logits_flat, y_u_flat)
            loss_v = criterion(v_logits_flat, y_v_flat)
            uniqueness_loss = uniqueness_criterion(u_logits, v_logits, mask)
            loss = loss_u + loss_v + uniqueness_loss
            
            valid_points = (y_u_flat != -1).sum().item()
            total_val_loss += loss.item() * valid_points
            
            pred_u = torch.argmax(u_logits_flat, dim=1)
            pred_v = torch.argmax(v_logits_flat, dim=1)
            valid_mask = (y_u_flat != -1)
            
            correct_u += ((pred_u == y_u_flat) & valid_mask).sum().item()
            correct_v += ((pred_v == y_v_flat) & valid_mask).sum().item()
            correct_combined += ((pred_u == y_u_flat) & (pred_v == y_v_flat) & valid_mask).sum().item()
            total += valid_mask.sum().item()
    
    avg_val_loss = total_val_loss / max(1, total)
    val_losses.append(avg_val_loss)
    
    val_u_acc = correct_u / max(1, total)
    val_v_acc = correct_v / max(1, total)
    val_combined_acc = correct_combined / max(1, total)
    
    val_u_accuracies.append(val_u_acc)
    val_v_accuracies.append(val_v_acc)
    val_combined_accuracies.append(val_combined_acc)
    
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"[DGCNN] Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    print(f"  U Acc: {val_u_acc:.4f} - V Acc: {val_v_acc:.4f} - Combined Acc: {val_combined_acc:.4f} - LR: {current_lr:.6f}")

if n_epochs > 0:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_u_accuracies': val_u_accuracies,
        'val_v_accuracies': val_v_accuracies,
        'val_combined_accuracies': val_combined_accuracies,
        'n_u': n_u,
        'n_v': n_v,
        'data_hash': current_data_hash,
        'model_config': {
            'k_base': 8,
            'k_max': 20,
            'hidden': 64,
            'lr': 5e-4,
            'weight_decay': 1e-4,
            'architecture': 'adaptive_knn_attention_uniqueness'
        }
    }, model_path)

    best_acc = max(val_combined_accuracies) if val_combined_accuracies else float('nan')
    final_acc = val_combined_accuracies[-1] if val_combined_accuracies else float('nan')
    print(f"Model saved to '{model_path}' with data hash {current_data_hash[:8]}...")
    print(f"Best combined accuracy: {best_acc:.4f}")
    print(f"Final combined accuracy: {final_acc:.4f}")
    print(f"Model supports {n_u} u-classes and {n_v} v-classes")

print("Training routine completed.")
