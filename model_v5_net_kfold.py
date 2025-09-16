# Model z pliku model_v4.py
# sieć neuronowa, wynik ok 80%
# dodane theta/beta/tbr

from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Dense, Concatenate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
import mne
import kagglehub

# -------- CONFIG --------

path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")
print("Path to dataset:", path)
data_path = Path(path) / "adhdata.csv"
CSV_PATH = data_path

FS = 128
EPOCH_SEC = 2
EPOCH_SAMPLES = EPOCH_SEC * FS
EPOCH_STEP = EPOCH_SAMPLES // 2
MODEL_SAVE_DIR = "EEGNet_kfold_models2"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# -------- Load CSV and build epochs (surowe sygnały) --------
df = pd.read_csv(CSV_PATH)
exclude = ["Class","ID"]
channels = [c for c in df.columns if c not in exclude]
label_map = {"Control":0, "ADHD":1}

# -------- Build epochs --------
X_epochs, y_epochs, groups = [], [], []

for subject_id, sub_df in tqdm(df.groupby("ID")):
    data = sub_df[channels].values.T  # (channels, n_samples)
    n_samples = data.shape[1]
    if n_samples < EPOCH_SAMPLES:
        continue
    label = label_map[sub_df["Class"].mode()[0]]
    start = 0
    while start + EPOCH_SAMPLES <= n_samples:
        epoch = data[:, start:start+EPOCH_SAMPLES]
        X_epochs.append(epoch)
        y_epochs.append(label)
        groups.append(subject_id)
        start += EPOCH_STEP

X_epochs = np.array(X_epochs)  # (n_epochs, channels, samples)
y_epochs = np.array(y_epochs)
groups = np.array(groups)

# standardize per channel
for ch in range(X_epochs.shape[1]):
    X_epochs[:, ch, :] = (X_epochs[:, ch, :] - X_epochs[:, ch, :].mean()) / (X_epochs[:, ch, :].std() + 1e-12)

# add channel dimension for EEGNet: (n_samples, n_channels, n_time, 1)
X_epochs = X_epochs[..., np.newaxis]

# -------- Compute theta, beta, TBR features using MNE --------
features = []
for epoch in tqdm(X_epochs.squeeze(-1), desc="Computing TBR features"):
    # create MNE RawArray for this epoch
    info = mne.create_info(ch_names=[str(c) for c in range(epoch.shape[0])], sfreq=FS, ch_types='eeg')
    raw_epoch = mne.io.RawArray(epoch, info)

    # compute variance as proxy for band power
    theta = raw_epoch.copy().filter(4, 7).get_data().var(axis=-1).mean()
    beta = raw_epoch.copy().filter(13, 30).get_data().var(axis=-1).mean()
    tbr = theta / (beta + 1e-12)
    features.append([theta, beta, tbr])

features = np.array(features)  # (n_epochs, 3)

# -------- EEGNet model --------
def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
    input1 = layers.Input(shape=(Chans, Samples, 1), name="input_layer")
    # Block 1
    block1 = layers.Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.DepthwiseConv2D((Chans,1), depth_multiplier=D, use_bias=False, padding='valid')(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('elu')(block1)
    block1 = layers.AveragePooling2D((1,4))(block1)
    block1 = layers.Dropout(dropoutRate)(block1)
    # Block 2
    block2 = layers.SeparableConv2D(F2, (1,16), padding='same', use_bias=False)(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.AveragePooling2D((1,8))(block2)
    block2 = layers.Dropout(dropoutRate)(block2)
    flatten = layers.Flatten()(block2)
    dense = layers.Dense(nb_classes, activation='softmax')(flatten)
    model = models.Model(inputs=input1, outputs=dense)
    return model

# -------- K-FOLD training --------
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_results = []

RESULTS_FILE = "KFold_EEGNet_TBR_results.txt"
with open(RESULTS_FILE, "w") as f:
    f.write("=== Wyniki K-Fold EEGNet+TBR ===\n\n")

print(f"Rozpoczynam K-Fold (k={n_splits})...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X_epochs, y_epochs)):
    X_train, X_test = X_epochs[train_idx], X_epochs[test_idx]
    X_train_feats, X_test_feats = features[train_idx], features[test_idx]
    y_train, y_test = y_epochs[train_idx], y_epochs[test_idx]

    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 2)

    # EEGNet
    base_model = EEGNet(nb_classes=2, Chans=X_epochs.shape[1], Samples=X_epochs.shape[2])
    eeg_features = base_model.layers[-2].output
    feat_input = Input(shape=(3,), name="Feat_input")
    feat_dense = Dense(8, activation="relu")(feat_input)
    merged = Concatenate()([eeg_features, feat_dense])
    out = Dense(2, activation="softmax")(merged)
    model = models.Model(inputs=[base_model.input, feat_input], outputs=out)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train
    print(f"\n[Fold {fold + 1}/{n_splits}] Trening...")
    history = model.fit(
        {"input_layer": X_train, "Feat_input": X_train_feats},
        y_train_cat,
        validation_data=({"input_layer": X_test, "Feat_input": X_test_feats}, y_test_cat),
        epochs=30, batch_size=32, verbose=1
    )

    # Save model
    model.save(os.path.join(MODEL_SAVE_DIR, f"EEGNet_fold_{fold+1}.h5"))

    # Evaluate
    probs = model.predict({"input_layer": X_test, "Feat_input": X_test_feats}, verbose=0)
    preds = np.argmax(probs, axis=1)

    auc = roc_auc_score(y_test, probs[:, 1])
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    fold_results = {
        "fold": fold+1,
        "AUC": auc,
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }
    all_results.append(fold_results)

    print(f"[Fold {fold + 1}/{n_splits}] "
          f"AUC={auc:.3f}, Acc={acc:.3f}, BalAcc={bal_acc:.3f}, "
          f"Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

    # ---- zapisz wynik cząstkowy ----
    with open(RESULTS_FILE, "a") as f:
        f.write(f"Fold {fold + 1}/{n_splits}\n")
        for k, v in fold_results.items():
            if k != "fold":
                f.write(f"{k}={v:.4f}\n")
        f.write("Historia treningu (ostatnie epoki):\n")
        for e in range(len(history.history["loss"])):
            f.write(f"  Epoka {e + 1:02d} - "
                    f"loss={history.history['loss'][e]:.4f}, "
                    f"acc={history.history['accuracy'][e]:.4f}, "
                    f"val_loss={history.history['val_loss'][e]:.4f}, "
                    f"val_acc={history.history['val_accuracy'][e]:.4f}\n")
        f.write("\n")

# ---- zapisz wynik końcowy ----
mean_results = {k: np.mean([r[k] for r in all_results]) for k in all_results[0] if k != "fold"}
std_results = {k: np.std([r[k] for r in all_results]) for k in all_results[0] if k != "fold"}

print("\n=== PODSUMOWANIE ===")
for k in mean_results:
    print(f"{k}: {mean_results[k]:.3f} ± {std_results[k]:.3f}")

with open(RESULTS_FILE, "a") as f:
    f.write("=== PODSUMOWANIE ===\n")
    for k in mean_results:
        f.write(f"{k}={mean_results[k]:.4f} ± {std_results[k]:.4f}\n")