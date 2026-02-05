import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from DGCNN_classification_structure_scripts import (
    DGCNNSubstructure,
    StructureSetDatasetSubstructure,
    align_logits_with_hungarian,
    collate_fn_substructure,
    get_data_hash,
    group_by_structure_substructure,
    hungarian_remap_indices_for_structure,
    load_multistructure_points_from_json,
    normalize_scene,
    set_seed,
)


print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"NumPy: {np.__version__}")
print("Using pure PyTorch DGCNN implementation (no PyG or torch_cluster dependencies).")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory set to {os.getcwd()}")

# Load the multi-structure JSON file
df = load_multistructure_points_from_json('JSON/Substructure.json')

# Create models directory and specify model path
models_dir = 'trained_models'
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'Substructure.pth')
n_epochs = 10  # Set number of epochs for training


# Quick summary
n_substructures = df['substructure_id'].nunique()
print(f"Loaded {len(df)} labeled points from {df['structure'].nunique()} structures")
print(f"Substructure classes: {sorted(df['substructure_id'].unique())}")

# Structure-based split (no data leakage)
unique_structures = df['structure'].unique()
train_structs, test_structs = train_test_split(unique_structures, test_size=0.2, random_state=42)
train_df = df[df['structure'].isin(train_structs)].reset_index(drop=True)
test_df = df[df['structure'].isin(test_structs)].reset_index(drop=True)

print(f"Train split: {len(train_structs)} structures ({len(train_df)} points)")
print(f"Test split: {len(test_structs)} structures ({len(test_df)} points)")

# Apply scene-based normalization per structure
for struct_df in [train_df, test_df]:
    struct_df['x_norm'] = 0.0
    struct_df['y_norm'] = 0.0
    for struct_id in struct_df['structure'].unique():
        mask = struct_df['structure'] == struct_id
        pts = struct_df.loc[mask, ['x_mm', 'y_mm']].values
        pts_norm = normalize_scene(pts)
        struct_df.loc[mask, 'x_norm'] = pts_norm[:, 0]
        struct_df.loc[mask, 'y_norm'] = pts_norm[:, 1]

print("Coordinates normalized per structure")

# Data loading and preprocessing

train_structs_grouped = group_by_structure_substructure(train_df)
test_structs_grouped = group_by_structure_substructure(test_df)

train_set = StructureSetDatasetSubstructure(train_structs_grouped)
test_set = StructureSetDatasetSubstructure(test_structs_grouped)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn_substructure)
test_loader = DataLoader(test_set, batch_size=8, collate_fn=collate_fn_substructure)

print(f"Datasets ready: {len(train_set)} train / {len(test_set)} test structures (batch_size=8)")

# Initialize environment for training
print("Hungarian alignment utilities loaded for permutation-invariant supervision")

set_seed(42)
print("Random seed set to 42 for reproducible results")

# Ensure we have the correct number of substructure classes
n_substructures_actual = df['substructure_id'].max() + 1
unique_substructures_actual = sorted(df['substructure_id'].unique())

print(f"Detected {n_substructures_actual} substructure classes: {unique_substructures_actual}")

# Instantiate model and optimizer
model = DGCNNSubstructure(n_substructures=n_substructures_actual, k=12, hidden=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Using k={model.k} nearest neighbors with dynamic EdgeConv blocks")
print(f"Supervising {n_substructures_actual} classes: {unique_substructures_actual}")


current_data_hash = get_data_hash(train_df, n_substructures_actual)
print(f"Current training data hash: {current_data_hash}")

train_losses = []
val_losses = []
val_accuracies = []

if os.path.exists(model_path):
    print(f"Found saved model at '{model_path}'")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_data_hash = checkpoint.get('data_hash', 'unknown')
    saved_n_substructures = checkpoint.get('n_substructures', 'unknown')

    data_changed = saved_data_hash != current_data_hash
    classes_changed = saved_n_substructures != n_substructures_actual

    if data_changed:
        print("Training data changed since the last checkpoint; retraining from scratch.")
    if classes_changed:
        print(
            f"Number of classes changed (saved={saved_n_substructures}, current={n_substructures_actual}); "
            "retraining with updated architecture."
        )

    if not data_changed and not classes_changed:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])

        best_acc = max(val_accuracies) if val_accuracies else float('nan')
        final_acc = val_accuracies[-1] if val_accuracies else float('nan')
        print(
            f"Checkpoint matches current configuration; reusing weights. "
            f"Best accuracy {best_acc:.4f}, last accuracy {final_acc:.4f}."
        )
        n_epochs = 0
    else:
        print("Existing checkpoint is incompatible; starting a new training run.")
else:
    print(f"No saved model found at '{model_path}'. A new training run will be started.")

# Training loop with Hungarian-aligned Cross Entropy
for epoch in range(n_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (X, y_sub, mask, struct_ids) in enumerate(train_loader):
        X, y_sub, mask = X.to(device), y_sub.to(device), mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        sub_logits = model(X, mask)  # [B, N, K]
        
        # Hungarian-aligned CE, summed over structures in the batch
        loss = 0.0
        B = X.size(0)
        for b in range(B):
            x_aligned_b, y_valid_b = align_logits_with_hungarian(sub_logits[b], y_sub[b], mask[b])
            if y_valid_b.numel() == 0:
                continue
            loss = loss + F.cross_entropy(x_aligned_b, y_valid_b)
        
        # If no valid points in the whole batch, skip
        if isinstance(loss, float) and loss == 0.0:
            continue
        
        # Backward + step (normalize loss by batch size)
        _loss_val = loss / B
        _loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate loss for reporting (use normalized value)
        train_loss += float(_loss_val.item())
    
    # Validation phase with Hungarian-matched accuracy
    model.eval()
    val_loss_ce = 0.0
    val_batches = 0
    val_accs = []
    
    with torch.no_grad():
        for X, y_sub, mask, struct_ids in test_loader:
            X, y_sub, mask = X.to(device), y_sub.to(device), mask.to(device)
            
            sub_logits = model(X, mask)  # [B,N,K]
            
            # CE on aligned logits for scalar loss
            B = X.size(0)
            for b in range(B):
                x_aligned_b, y_valid_b = align_logits_with_hungarian(sub_logits[b], y_sub[b], mask[b])
                if y_valid_b.numel() == 0:
                    continue
                val_loss_ce += float(F.cross_entropy(x_aligned_b, y_valid_b).item())
                val_batches += 1
            
            # Hungarian-matched accuracy per structure
            pred = sub_logits.argmax(dim=-1)   # [B,N]
            for b in range(B):
                valid_b = (y_sub[b] != -1) & mask[b]
                if not valid_b.any():
                    continue
                remap_b = hungarian_remap_indices_for_structure(sub_logits[b], y_sub[b], mask[b])
                pred_aligned_b = pred[b].clone()
                for k_pred in range(remap_b.numel()):
                    k_gt = int(remap_b[k_pred].item())
                    pred_aligned_b[pred[b] == k_pred] = k_gt
                acc_b = (pred_aligned_b[valid_b] == y_sub[b][valid_b]).float().mean().item()
                val_accs.append(acc_b)
    
    # Calculate metrics
    train_loss /= len(train_loader)
    val_loss = (val_loss_ce / val_batches) if val_batches > 0 else 0.0
    val_acc = float(np.mean(val_accs)) if val_accs else 0.0
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Print progress (removed train accuracy for ~50% faster training)
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d}/{n_epochs}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

if n_epochs > 0:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'n_substructures': n_substructures_actual,
        'unique_substructures': unique_substructures_actual,
        'data_hash': current_data_hash,
        'model_config': {
            'k': model.k,
            'hidden': 64,
            'n_substructures': n_substructures_actual,
            'architecture': 'dynamic_knn_v2'
        }
    }, model_path)

    best_acc = max(val_accuracies) if val_accuracies else float('nan')
    final_acc = val_accuracies[-1] if val_accuracies else float('nan')
    print(f"Model saved to '{model_path}' with data hash {current_data_hash[:8]}...")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Final validation accuracy: {final_acc:.4f}")
    print(f"Model supports {n_substructures_actual} substructure classes: {unique_substructures_actual}")

print("Training routine completed.")