import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from DGCNN_classification_structure_scripts import *  # Import all utility functions

# ===========================
# User-editable parameters
# ===========================
SEED = 42
REPRODUCIBLE = True
FORCE_RETRAIN = True

JSON_PATH = 'JSON/Substructure.json'
N_EPOCHS = 20
TRAIN_TEST_SPLIT = 0.2

BATCH_SIZE = 8
K_NEIGHBORS = 12
HIDDEN_DIM = 64
LR = 1e-3           
WEIGHT_DECAY = 1e-5  

MODELS_OUTPUT_DIR = 'trained_models'
MODEL_FILENAME = 'Substructure.pth'
RESULTS_OUTPUT_DIR = 'results'
RESULTS_CSV_FILENAME = 'Model_comparison_structure.csv'

print("Python:", sys.version)
try:
    print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("SciPy:", "linear_sum_assignment available")
except Exception as e:
    print("Import error:", e)

print("Using pure PyTorch DGCNN implementation (no PyG or torch_cluster dependencies).")

if REPRODUCIBLE:
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
else:
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = REPRODUCIBLE
torch.backends.cudnn.benchmark = not REPRODUCIBLE
torch.use_deterministic_algorithms(REPRODUCIBLE, warn_only=True)

print(f"Random seed locked to {SEED}")
print(f"Reproducible mode: {REPRODUCIBLE}")
print(f"Force retrain: {FORCE_RETRAIN}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

df = load_multistructure_points_from_json(JSON_PATH)

models_dir = MODELS_OUTPUT_DIR
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, MODEL_FILENAME)
n_epochs = N_EPOCHS

n_substructures = df['substructure_id'].nunique()
print(f"Loaded {len(df)} points from {df['structure'].nunique()} structures")
print(f"   Substructure classes: {sorted(df['substructure_id'].unique())}")

# Structure-based split (no data leakage)
unique_structures = df['structure'].unique()
split_random_state = SEED if REPRODUCIBLE else None
train_structs, test_structs = train_test_split(unique_structures, test_size=TRAIN_TEST_SPLIT, random_state=split_random_state)
train_df = df[df['structure'].isin(train_structs)].reset_index(drop=True)
test_df = df[df['structure'].isin(test_structs)].reset_index(drop=True)

print(f"   Train: {len(train_structs)} structures ({len(train_df)} points)")
print(f"   Test: {len(test_structs)} structures ({len(test_df)} points)")

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

print("Data normalized and ready for training")

train_structs_grouped = group_by_structure_substructure(train_df)
test_structs_grouped = group_by_structure_substructure(test_df)

train_set = StructureSetDatasetSubstructure(train_structs_grouped)
test_set = StructureSetDatasetSubstructure(test_structs_grouped)
if REPRODUCIBLE:
    _g = torch.Generator()
    _g.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_substructure, generator=_g)
else:
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_substructure)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn_substructure)

print(f"Datasets ready: {len(train_set)} train, {len(test_set)} test structures (batch_size={BATCH_SIZE})")
print("Hungarian alignment enabled for permutation-invariant substructure training")

start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure we have the correct number of substructure classes
n_substructures_actual = df['substructure_id'].max() + 1
unique_substructures_actual = sorted(df['substructure_id'].unique())

print(f"Data analysis: found {n_substructures_actual} substructure classes: {unique_substructures_actual}")

model = DGCNNSubstructure(n_substructures=n_substructures_actual, k=K_NEIGHBORS, hidden=HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

print(f"Using DGCNN for {n_substructures_actual}-class substructure classification")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Substructure classes: {n_substructures_actual} {unique_substructures_actual}")
print("Architecture: 4 EdgeConv layers (2->64->128->256->256) with GroupNorm and multi-scale fusion")
print(f"Hyperparameters: k={K_NEIGHBORS}, hidden={HIDDEN_DIM}, lr={LR}, weight_decay={WEIGHT_DECAY}")


current_data_hash = get_data_hash(train_df, n_substructures_actual)
print(f"Current training data hash: {current_data_hash}")

if os.path.exists(model_path):
    print(f"\nFound saved model at '{model_path}'")
    
    # Load checkpoint to check data version AND class compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_data_hash = checkpoint.get('data_hash', 'unknown')
    saved_n_substructures = checkpoint.get('n_substructures', 'unknown')
    
    print(f"Saved model data hash: {saved_data_hash}")
    print(f"Current data hash: {current_data_hash}")
    print(f"Saved model classes: {saved_n_substructures}")
    print(f"Current data classes: {n_substructures_actual}")
    
    # Check both data changes AND class compatibility
    data_changed = saved_data_hash != current_data_hash
    classes_changed = saved_n_substructures != n_substructures_actual
    
    if data_changed:
        print("WARNING: Training data has changed since model was last trained")
        print("Data changes detected; model should be retrained")
    elif classes_changed:
        print("WARNING: Number of substructure classes has changed")
        print(f"Model architecture mismatch: saved={saved_n_substructures}, current={n_substructures_actual}")
        print("Model needs retraining for new class structure")
    else:
        print("Training data and class structure unchanged; model is up to date")
    
    needs_retraining = data_changed or classes_changed
    
    # Auto-decide based on changes and user flag
    if FORCE_RETRAIN:
        user_choice = 'n'
        print("Auto-selected: retraining because FORCE_RETRAIN=True")
    elif needs_retraining:
        user_choice = 'n'  # Retrain by default if anything changed
        reason = "data changes" if data_changed else "class structure changes"
        print(f"Auto-selected: retraining due to {reason} (set FORCE_RETRAIN=True to always retrain)")
    else:
        user_choice = 'y'  # Use saved model if everything unchanged
        print("Auto-selected: using saved model (set FORCE_RETRAIN=True to retrain anyway)")
    
    if user_choice == 'y' and not classes_changed:  # Can only load if classes match
        # Load the saved model
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_accuracies = checkpoint['val_accuracies']
        
        print("Model loaded successfully")
        print(f"Best accuracy: {max(val_accuracies):.4f}")
        print(f"Final accuracy: {val_accuracies[-1]:.4f}")
        print("Skipping training and using loaded model for predictions")
        
        # Skip training
        n_epochs = 0
    else:
        if classes_changed:
            print(f"Starting fresh training with new {n_substructures_actual}-class architecture")
        else:
            print("Starting fresh training")
        train_losses = []
        val_losses = []
        val_accuracies = []
        
else:
    print(f"No saved model found; starting fresh {n_substructures_actual}-class training")
    train_losses = []
    val_losses = []
    val_accuracies = []

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
    
    print(f"\nModel saved to '{model_path}' with data hash: {current_data_hash[:8]}...")
    print(f"Best validation accuracy: {max(val_accuracies):.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Model supports {n_substructures_actual} substructure classes: {unique_substructures_actual}")

elapsed_time = time.time() - start_time

final_acc = val_accuracies[-1] if val_accuracies else None
best_acc = max(val_accuracies) if val_accuracies else None

result_row = pd.DataFrame([
    {
        'file_name': os.path.basename(JSON_PATH),
        'json_path': JSON_PATH,
        'seed': SEED,
        'reproducible': REPRODUCIBLE,
        'force_retrain': FORCE_RETRAIN,
        'n_substructures': n_substructures_actual,
        'n_points': len(df),
        'n_structures': df['structure'].nunique(),
        'epochs': n_epochs,
        'k_neighbors': K_NEIGHBORS,
        'hidden_dim': HIDDEN_DIM,
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'best_accuracy': best_acc,
        'final_accuracy': final_acc,
        'elapsed_seconds': elapsed_time,
        'model_path': model_path,
    }
])

results_dir = RESULTS_OUTPUT_DIR
os.makedirs(results_dir, exist_ok=True)
results_csv = os.path.join(results_dir, RESULTS_CSV_FILENAME)

write_header = not os.path.exists(results_csv)
result_row.to_csv(results_csv, mode='a', header=write_header, index=False)

print(f"Result written to: {results_csv}")
print(result_row.to_string(index=False))
    
print("Training completed")