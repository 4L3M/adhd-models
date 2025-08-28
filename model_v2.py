import os
from pathlib import Path
import numpy as np
import mne
import pandas as pd
from collections import Counter

import kagglehub

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft, butter, filtfilt

# ----------------------
# Config
# ----------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SF = 128
WINDOW_SEC = 2
WINDOW_SIZE = WINDOW_SEC * SF
STEP_SEC = 1
STEP_SIZE = STEP_SEC * SF

USE_SPECTROGRAM = True
N_PER_SEG = 128
N_OVERLAP = 64

EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 5

BANDPASS = (1, 45)
NOTCH_50HZ = True

# ----------------------
# Filters & Spectrogram
# ----------------------
def bandpass_filter(x, fs, low, high, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def notch_filter_50hz(x, fs, width=1.0, order=4):
    low = (50 - width)/(fs/2)
    high = (50 + width)/(fs/2)
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, x)

def make_spectrogram(window, fs=SF):
    ch_spects = []
    for ch in window:
        f, t, Zxx = stft(ch, fs=fs, nperseg=min(N_PER_SEG, len(ch)), noverlap=min(N_OVERLAP, len(ch)//2))
        mag = np.abs(Zxx) + 1e-8
        log_mag = np.log(mag)
        ch_spects.append(log_mag)
    return np.stack(ch_spects, axis=0).astype(np.float32)

def normalize_window(x):
    for c in range(x.shape[0]):
        mu = x[c].mean()
        std = x[c].std() + 1e-6
        x[c] = (x[c] - mu) / std
    return x

# ----------------------
# Dataset & Models
# ----------------------
class EEGDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = normalize_window(s["x"].astype(np.float32))
        y = torch.tensor(s["y"], dtype=torch.long)
        return torch.from_numpy(x), y

class CNN2D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.net(x))

class CNN1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, x):
        return self.head(self.net(x))

# ----------------------
# Training & Evaluation
# ----------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def predict_logits(model, loader):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.numpy())
    return np.concatenate(all_logits), np.concatenate(all_y)

def evaluate(model, loader):
    logits, y = predict_logits(model, loader)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
    y_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = float('nan')
    return acc, auc

def make_class_weights(y):
    cnt = Counter(y)
    total = sum(cnt.values())
    weights = {cls: total/(len(cnt)*cnt[cls]) for cls in cnt}
    return torch.tensor([weights[0], weights[1]], dtype=torch.float32).to(device)

# ----------------------
# Main
# ----------------------
def main():
    print("Downloading/locating dataset via kagglehub…")
    path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")
    data_path = Path(path) / "adhdata.csv"
    df = pd.read_csv(data_path)

    channels = [c for c in df.columns if c not in ["Class", "ID"]]
    df["y"] = LabelEncoder().fit_transform(df["Class"])

    # Subject-wise data
    subject_data = {}
    for sid in df["ID"].unique():
        sub_df = df[df["ID"] == sid]
        Xsub = sub_df[channels].values.astype(np.float32)
        ysub = int(sub_df["y"].iloc[0])
        # Optional filtering
        if BANDPASS is not None:
            Xf = []
            for ch in range(Xsub.shape[1]):
                x = Xsub[:, ch]
                try:
                    x = bandpass_filter(x, SF, *BANDPASS)
                    if NOTCH_50HZ:
                        x = notch_filter_50hz(x, SF)
                except Exception as e:
                    print(f"Filter error for subject {sid}, channel {ch}: {e}")
                Xf.append(x.astype(np.float32))
            Xsub = np.stack(Xf, axis=1)
        subject_data[sid] = (Xsub, ysub)

    # Sliding windows
    samples = []
    for sid, (Xsub, ysub) in subject_data.items():
        n = Xsub.shape[0]
        for start in range(0, max(0, n - WINDOW_SIZE + 1), STEP_SIZE):
            end = start + WINDOW_SIZE
            if end > n:
                break
            win = Xsub[start:end].T
            feat = make_spectrogram(win) if USE_SPECTROGRAM else win.astype(np.float32)
            samples.append({"sid": sid, "x": feat, "y": ysub})

    print(f"Total windows: {len(samples)}")

    groups = np.array([s["sid"] for s in samples])
    Y = np.array([s["y"] for s in samples])

    # ----------------------
    # GroupKFold training
    # ----------------------
    gkf = GroupKFold(n_splits=5)
    accs, aucs = [], []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(np.arange(len(samples)), Y, groups), 1):
        print(f"\n===== Fold {fold} =====")
        train_samples = [samples[i] for i in tr_idx]
        val_samples = [samples[i] for i in va_idx]

        train_ds = EEGDataset(train_samples)
        val_ds = EEGDataset(val_samples)

        dummy_x, _ = train_ds[0]
        in_ch = dummy_x.shape[0]
        model = (CNN2D(in_ch) if USE_SPECTROGRAM else CNN1D(in_ch)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(weight=make_class_weights([s["y"] for s in train_samples]))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        best_auc, best_state, patience = -np.inf, None, PATIENCE

        for epoch in range(1, EPOCHS+1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_acc, val_auc = evaluate(model, val_loader)
            print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} | val_acc={val_acc:.4f} | val_auc={val_auc:.4f}")
            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                patience = PATIENCE
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping!")
                    break

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        va_acc, va_auc = evaluate(model, val_loader)
        print(f"Best Fold {fold} -> ACC={va_acc:.4f}, AUC={va_auc:.4f}")
        accs.append(va_acc)
        aucs.append(va_auc)

    print(f"\n==== CV Summary ====\nACC: {np.mean(accs):.3f} ± {np.std(accs):.3f}\nAUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

if __name__ == "__main__":
    main()
