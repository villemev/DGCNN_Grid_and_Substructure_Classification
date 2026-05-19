import os
import random
import sys
import time
from pathlib import Path

import hashlib
import numpy as np
import pandas as pd
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

# ===========================
# User-editable parameters
# ===========================
SEED = 42
JSON_PATH = 'JSON/UV.json'
N_EPOCHS = 20

MODELS_OUTPUT_DIR = 'trained_models'
MODEL_FILENAME = 'UV.pth'

RESULTS_OUTPUT_DIR = 'results'
RESULTS_CSV_FILENAME = 'Model_comparison.csv'

TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE = 8
REPRODUCIBLE = True

# Set random seeds for reproducibility
if REPRODUCIBLE:
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
else:
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for multi-GPU
torch.backends.cudnn.deterministic = REPRODUCIBLE
torch.backends.cudnn.benchmark = not REPRODUCIBLE
torch.use_deterministic_algorithms(REPRODUCIBLE, warn_only=True)

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"NumPy: {np.__version__}")
print("Using pure PyTorch DGCNN implementation (no PyG or torch_cluster dependencies).")
print(f"Random seed locked to {SEED} for reproducibility")
print(f"Reproducible mode: {REPRODUCIBLE}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory set to {os.getcwd()}")

json_path = Path(JSON_PATH)
df = load_points_from_json(json_path)

# Model save directory
models_dir = Path(MODELS_OUTPUT_DIR)
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / MODEL_FILENAME
n_epochs = N_EPOCHS

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
train_structs, test_structs = train_test_split(
    unique_structures,
    test_size=TRAIN_TEST_SPLIT,
    random_state=SEED,
)
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

train_structs_grouped = group_by_structure_separate(train_df)
test_structs_grouped = group_by_structure_separate(test_df)

train_set = StructureSetDatasetSeparate(train_structs_grouped)
test_set = StructureSetDatasetSeparate(test_structs_grouped)
if REPRODUCIBLE:
    _g = torch.Generator()
    _g.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_separate, generator=_g)
else:
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_separate)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn_separate)

start_time = time.time()

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
    print(
        f"Found saved model at '{model_path}', but retraining is forced; "
        "starting a new training run from scratch."
    )
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
    }, str(model_path))

    best_acc = max(val_combined_accuracies) if val_combined_accuracies else float('nan')
    final_acc = val_combined_accuracies[-1] if val_combined_accuracies else float('nan')
    print(f"Model saved to '{model_path}' with data hash {current_data_hash[:8]}...")
    print(f"Best combined accuracy: {best_acc:.4f}")
    print(f"Final combined accuracy: {final_acc:.4f}")
    print(f"Model supports {n_u} u-classes and {n_v} v-classes")

print("Training routine completed.")

elapsed_time = time.time() - start_time

max_u = int(df['u'].max())
max_v = int(df['v'].max())
max_grid_size = max(max_u + 1, max_v + 1)

final_u_acc = val_u_accuracies[-1] if val_u_accuracies else None
final_v_acc = val_v_accuracies[-1] if val_v_accuracies else None
final_combined_acc = val_combined_accuracies[-1] if val_combined_accuracies else None

result_row = pd.DataFrame([
    {
        'file_name': json_path.name,
        'json_path': str(json_path),
        'seed': SEED,
        'max_u': max_u,
        'max_v': max_v,
        'max_grid_size': max_grid_size,
        'n_points': len(df),
        'n_structures': df['structure'].nunique(),
        'epochs': n_epochs,
        'u_accuracy': final_u_acc,
        'v_accuracy': final_v_acc,
        'combined_accuracy': final_combined_acc,
        'elapsed_seconds': elapsed_time,
        'model_path': str(model_path),
    }
])

results_dir = Path(RESULTS_OUTPUT_DIR)
results_dir.mkdir(parents=True, exist_ok=True)

results_csv = results_dir / RESULTS_CSV_FILENAME

write_header = not results_csv.exists()
result_row.to_csv(results_csv, mode='a', header=write_header, index=False)

print("Training routine completed.")
print(f"Result written to: {results_csv}")
print(result_row.to_string(index=False))
