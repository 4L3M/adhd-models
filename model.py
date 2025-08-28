import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import kagglehub
import os
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier

import joblib

from scipy.signal import welch

# === 1. Wczytanie danych ===

path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")
print("Path to dataset:", path)

data_path = Path(path) / "adhdata.csv"


df = pd.read_csv(data_path)  # tu zmień na swoją ścieżkę

all_columns = df.columns.tolist()
channels = [col for col in all_columns if col not in ['Class', 'ID']]

X_raw = df[channels].values
y = df['Class'].values
groups = df['ID'].values

# Label encoding ADHD/Control
le = LabelEncoder()
y = le.fit_transform(y)

print("Kanały EEG:", channels)
print("Rozmiar danych:", X_raw.shape)

# === 2. Funkcja do ekstrakcji cech EEG ===
def extract_bandpower(signal, sf, bands):
    signal = np.asarray(signal)   # upewnij się, że to numpy array
    freqs, psd = welch(signal, fs=sf, nperseg=min(128, len(signal)))  # ważne: nperseg <= długość sygnału
    band_powers = {}
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band_name] = np.trapezoid(psd[idx_band], freqs[idx_band])
    return band_powers


def extract_features(df, channels, sf=128, window_size=256):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }

    features = []
    labels = []
    groups = []

    # Iteruj po oknach
    for start in range(0, len(df) - window_size, window_size):
        end = start + window_size
        window = df.iloc[start:end]

        row_feats = []
        for ch in channels:
            signal = window[ch].values

            # Moc w pasmach częstotliwościowych
            band_powers = extract_bandpower(signal, sf=sf, bands=bands)
            row_feats.extend(band_powers.values())

            # Stosunek theta/beta
            theta = band_powers["theta"]
            beta = band_powers["beta"]
            ratio = theta / (beta + 1e-6)
            row_feats.append(ratio)

        features.append(row_feats)
        labels.append(window["Class"].iloc[0])
        groups.append(window["ID"].iloc[0])

    return np.array(features), np.array(labels), np.array(groups)


print("Ekstrakcja cech EEG...")
X, y, groups = extract_features(df, channels)
print("Gotowe X shape:", X.shape)

# === 3. Modele do testów ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
#    "LogisticRegression": LogisticRegression(max_iter=1000, solver='saga', penalty='l1'),
    "SVM": SVC(kernel='rbf', probability=True, C=1, gamma='scale', random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=15, random_state=42)
}

# === 4. Walidacja krzyżowa ===
gkf = GroupKFold(n_splits=5)
results = {}

for model_name, model in models.items():
    print(f"\n=== Test modelu: {model_name} ===")
    accs, aucs = [], []

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f"  Fold {fold}: Accuracy={acc:.3f}, ROC-AUC={auc:.3f}")

        accs.append(acc)
        aucs.append(auc)

    results[model_name] = {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs)
    }

# === 5. Podsumowanie ===
print("\n=== Wyniki modeli ===")
for model_name, res in results.items():
    print(f"{model_name}: ACC={res['acc_mean']:.3f} ± {res['acc_std']:.3f}, "
          f"AUC={res['auc_mean']:.3f} ± {res['auc_std']:.3f}")

best_model_name = max(results, key=lambda k: results[k]['auc_mean'])
print(f"\n>>> Najlepszy model: {best_model_name}")

# === 6. Finalny model ===
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', models[best_model_name])
])

final_pipeline.fit(X, y)
joblib.dump(final_pipeline, f"adhd_eeg_best_model_{best_model_name}.joblib")
print(f"Model zapisany jako adhd_eeg_best_model_{best_model_name}.joblib")
