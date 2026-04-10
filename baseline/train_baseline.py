import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader 
from sklearn.metrics import f1_score, matthews_corrcoef
import random
import os
import concurrent.futures

# DATASET & MODEL CONFIGURATION
model_name = "big_model"
dataset_name = "PDBbind-1409"
path_data = model_name + "_data/"
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(repo_root, "data") + os.sep
print(f"Model: {model_name}")
print(f"Dataset: {dataset_name}")


# GLOBAL VARIABLES
EPOCHS = 100
BATCH = 96
lr = 5e-4
THRESHOLD = 0.7
print(f"Threshold: {THRESHOLD}")

# EARLY STOPPING PARAMETERS
PATIENCE = 5       
MIN_DELTA = 1e-4 
best_val_loss = float('inf')
best_state = None
epochs_no_improve = 0

# SEED FOR REPRODUCIBILITY
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# DEVICE CONFIGURATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FUNCTIONS
def load_aligned_data(csv_path, pt_path, label_col='p_interface', id_col='uniprot_id'):
    df = pd.read_csv(csv_path)
    emb_dict = torch.load(pt_path, map_location='cpu')
    
    id_set = set(emb_dict.keys())
    df_filtered = df[df[id_col].astype(str).isin(id_set)]
    
    X_list = [F.normalize(emb_dict[str(u_id)], p=2, dim=1) for u_id in df_filtered[id_col]]
    y_list = [np.fromstring(str(labels), dtype=np.int32, sep=',') for labels in df_filtered[label_col]]
    
    skipped = len(df) - len(df_filtered)
    if skipped > 0:
        print(f"Warning: Skipped {skipped} IDs from {csv_path} not found in {pt_path}")

    return X_list, y_list

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, X_list, y_list):
        self.X_list = X_list
        self.y_list = y_list
        self.mapping = []
        for prot_idx, labels in enumerate(y_list):
            for res_idx in range(len(labels)):
                self.mapping.append((prot_idx, res_idx))
    
    def __len__(self):
        return len(self.mapping)
    
    def __getitem__(self, idx):
        prot_idx, res_idx = self.mapping[idx]
        x = self.X_list[prot_idx][res_idx].view(-1)  
        y = torch.tensor(self.y_list[prot_idx][res_idx], dtype=torch.float32)
        return x, y

def evaluate_loader(model, loader, criterion, device, threshold=THRESHOLD):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb.float()).squeeze(-1)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(yb.detach().cpu())
    avg_loss = total_loss / len(loader.dataset)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = (torch.sigmoid(logits) > threshold).int()
    f1  = f1_score(labels.numpy(), preds.numpy(), pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(labels.numpy(), preds.numpy())
    return avg_loss, f1, mcc

# DATA LOADING
def load_data_parallel(csv_path, pt_path):
    return load_aligned_data(csv_path, pt_path)

data_configs = [
    (data_path + "zk448_test.csv",            path_data + "zk448_test.pt"),
    (data_path + dataset_name + "_train.csv", path_data + dataset_name + "_train.pt"),
    (data_path + dataset_name + "_val.csv",   path_data + dataset_name + "_val.pt"),
]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(lambda x: load_data_parallel(x[0], x[1]), data_configs))

X_test, y_test = results[0]
X_train, y_train = results[1]
X_val, y_val = results[2]

train_dataset = ProteinDataset(X_train, y_train)
val_dataset   = ProteinDataset(X_val, y_val)
test_dataset  = ProteinDataset(X_test, y_test)

test_protein_full_lengths = [len(arr) for arr in y_test]

# MODEL & OPTIMIZER
hidden_dim = train_dataset[0][0].shape[0] 
model = nn.Linear(hidden_dim, 1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

# DATALOADERS
use_pin_memory = (device.type == 'cuda')
num_workers = min(16, os.cpu_count() or 8)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,  num_workers=num_workers, pin_memory=use_pin_memory, persistent_workers=True, prefetch_factor=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory, persistent_workers=True, prefetch_factor=4) 
test_loader  = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory, persistent_workers=True, prefetch_factor=4)


# CLASS WEIGHTING
all_train_y = np.concatenate(y_train)
pos = (all_train_y == 1).sum()
neg = (all_train_y == 0).sum()

eps = 1e-8
pos_weight = torch.tensor(neg / (pos + eps), dtype=torch.float32).clamp(min=1e-6, max=1e6)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))


# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb.float()).squeeze(-1)  
        loss = criterion(logits, yb)            
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    avg_train_loss = train_loss_sum / len(train_loader.dataset)
    _, f1_test, mcc_test = evaluate_loader(model, test_loader, criterion, device, threshold=THRESHOLD)
    val_loss, f1_val, mcc_val = evaluate_loader(model, val_loader, criterion, device, threshold=THRESHOLD)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={avg_train_loss:.4f} "
        f"val_loss={val_loss:.4f} "
        f"F1={f1_test:.4f} "
        f"MCC={mcc_test:.4f} "
        f"F1_val={f1_val:.4f} "
        f"MCC_val={mcc_val:.4f}"
    )

    # EARLY STOPPING 
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping")
            break

# LOAD BEST MODEL
if best_state is not None:
    model.load_state_dict(best_state)

# SAVE MODEL
PATH = f"{model_name}_{dataset_name}_{THRESHOLD}.pt"
torch.save({
    'epoch': EPOCHS, 
    'model_state_dict': model.state_dict(), 
    'optimizer_state_dict': optimizer.state_dict(), 
    'loss': avg_train_loss, 
}, PATH)
print(f"Model saved successfully: {os.path.abspath(PATH)}")

