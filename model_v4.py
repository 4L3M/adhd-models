from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from tqdm import tqdm
import os

# -------- CONFIG --------

CSV_PATH = "your_dataset.csv"      # <- zmień na swoją ścieżkę
path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")
print("Path to dataset:", path)
data_path = Path(path) / "adhdata.csv"

FS = 128
EPOCH_SEC = 2
EPOCH_SAMPLES = EPOCH_SEC * FS
EPOCH_STEP = EPOCH_SAMPLES // 2
MODEL_SAVE_DIR = "EEGNet_LOSO_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# -------- Load CSV and build epochs (surowe sygnały) --------
df = pd.read_csv(CSV_PATH)
exclude = ["Class","ID"]
channels = [c for c in df.columns if c not in exclude]
label_map = {"Control":0, "ADHD":1}

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

# -------- EEGNet model --------
def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
    input1 = layers.Input(shape=(Chans, Samples, 1))
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

# -------- LOSO training --------
unique_subjects = np.unique(groups)
all_aucs, all_bal_accs = [], []

for subj in tqdm(unique_subjects):
    train_idx = groups != subj
    test_idx = groups == subj

    X_train, X_test = X_epochs[train_idx], X_epochs[test_idx]
    y_train, y_test = y_epochs[train_idx], y_epochs[test_idx]

    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 2)

    model = EEGNet(nb_classes=2, Chans=X_epochs.shape[1], Samples=X_epochs.shape[2])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit model
    history = model.fit(X_train, y_train_cat, epochs=30, batch_size=32, verbose=0,
                        validation_data=(X_test, y_test_cat))

    # save model
    model_path = os.path.join(MODEL_SAVE_DIR, f"EEGNet_subj_{subj}.h5")
    model.save(model_path)

    # predict
    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)

    # metrics
    auc = roc_auc_score(y_test, probs[:,1])
    bal_acc = (preds == y_test).mean()

    all_aucs.append(auc)
    all_bal_accs.append(bal_acc)

print(f"LOSO EEGNet results: AUC={np.mean(all_aucs):.3f} ± {np.std(all_aucs):.3f}, "
      f"Balanced Acc={np.mean(all_bal_accs):.3f}")
