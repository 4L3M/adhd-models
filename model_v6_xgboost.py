# eeg_adhhd_feature_pipeline_xgb_loso.py
# Klasyfikator XGBoost (XGBClassifier) z early stopping
# Walidacja: Leave-One-Subject-Out (LOSO)
# Wyniki zapisują się do pliku TXT + CSV

"""
EEG ADHD Feature Pipeline z XGBoost + LOSO cross-validation

Ten skrypt implementuje pipeline do klasyfikacji ADHD vs Control na podstawie sygnałów EEG.
Wykorzystuje bogaty zestaw cech z każdej epoki (okno 2s, 50% overlap):

1. Cechy czasowe:
   - średnia, wariancja, odchylenie, skośność, kurtoza
   - RMS, peak-to-peak, zero-crossing rate
   - parametry Hjortha (aktywność, mobilność, złożoność)

2. Cechy częstotliwościowe:
   - moc pasmowa (Welch) w pasmach: delta, theta, alpha, beta, gamma
   - relatywna moc pasmowa
   - całkowita moc, TBR (theta/beta ratio)
   - entropia spektralna

3. Cechy FFT:
   - max, średnia, odchylenie, energia
   - energia w pasmach (absolutna i relatywna)

4. Agregaty globalne:
   - średnia i std mocy pasmowej po kanałach
   - globalne TBR i entropia spektralna

Ulepszenia zastosowane w pipeline:
- Walidacja LOSO (Leave-One-Subject-Out) – pełna generalizacja między pacjentami
- XGBoost z early stopping
- Standaryzacja cech osobno w każdym foldzie
- Zapisywanie wyników do TXT + CSV + modeli per pacjent

"""

from pathlib import Path
from tqdm import tqdm
import kagglehub
import numpy as np
from numpy.fft import rfft, rfftfreq
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.utils import shuffle
import xgboost as xgb
from xgboost import XGBClassifier, callback
import joblib
import os

MODEL_DIR = "xgb_models"  # nazwa folderu
os.makedirs(MODEL_DIR, exist_ok=True)


# -------- CONFIG --------
FS = 128
EPOCH_SEC = 4
EPOCH_SAMPLES = int(EPOCH_SEC * FS)
EPOCH_STEP = EPOCH_SAMPLES // 2
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
RANDOM_STATE = 42

# pobranie datasetu z kagglehub
path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")
print("Path to dataset:", path)
data_path = Path(path) / "adhdata.csv"
CSV_PATH = data_path

# -------- Helper functions --------
def bandpower_welch(data, fs, band, nperseg=256):
    f, Pxx = welch(data, fs=fs, nperseg=nperseg)
    low, high = band
    idx_band = np.logical_and(f >= low, f <= high)
    return np.trapezoid(Pxx[idx_band], f[idx_band])

def spectral_entropy_from_psd(psd, freqs):
    psd_norm = psd / np.sum(psd)
    psd_norm = np.where(psd_norm == 0, 1e-12, psd_norm)
    return -np.sum(psd_norm * np.log2(psd_norm))

def hjorth_parameters(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx) if dx.size > 0 else 0.0
    var_ddx = np.var(ddx) if ddx.size > 0 else 0.0
    activity = var_x
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
    complexity = (np.sqrt(var_ddx/var_dx) / mobility) if (var_dx > 0 and mobility > 0) else 0.0
    return activity, mobility, complexity

def zero_crossing_rate(x):
    return ((x[:-1] * x[1:]) < 0).sum() / len(x)

def fft_features_epoch(epoch, channels, fs=FS):
    n_ch, n_s = epoch.shape
    feat = {}
    for ci, ch in enumerate(channels):
        x = epoch[ci, :]
        fft_vals = np.abs(rfft(x))
        freqs = rfftfreq(n_s, 1/fs)

        feat[f"{ch}_fft_max"] = np.max(fft_vals)
        feat[f"{ch}_fft_mean"] = np.mean(fft_vals)
        feat[f"{ch}_fft_std"] = np.std(fft_vals)
        feat[f"{ch}_fft_energy"] = np.sum(fft_vals**2)

        total_energy = np.sum(fft_vals**2)
        for band_name, (low, high) in BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_energy = np.sum(fft_vals[idx]**2)
            feat[f"{ch}_fft_energy_{band_name}"] = band_energy
            feat[f"{ch}_fft_rel_energy_{band_name}"] = band_energy / (total_energy + 1e-12)
    return feat

def extract_features_epoch(epoch, channels):
    n_ch, n_s = epoch.shape
    feat = {}
    for ci, ch in enumerate(channels):
        x = epoch[ci, :]

        feat[f"{ch}_mean"] = np.mean(x)
        feat[f"{ch}_std"] = np.std(x)
        feat[f"{ch}_var"] = np.var(x)
        feat[f"{ch}_skew"] = skew(x)
        feat[f"{ch}_kurt"] = kurtosis(x)
        feat[f"{ch}_rms"] = np.sqrt(np.mean(x**2))
        feat[f"{ch}_ptp"] = np.ptp(x)
        feat[f"{ch}_zcr"] = zero_crossing_rate(x)

        a,m,c = hjorth_parameters(x)
        feat[f"{ch}_hjorth_activity"] = a
        feat[f"{ch}_hjorth_mobility"] = m
        feat[f"{ch}_hjorth_complexity"] = c

        f, Pxx = welch(x, fs=FS, nperseg=min(256, n_s))
        feat[f"{ch}_spec_entropy"] = spectral_entropy_from_psd(Pxx, f)
        total_power = 0.0
        band_pows = {}
        for band_name, band_range in BANDS.items():
            bp = bandpower_welch(x, FS, band_range, nperseg=min(256,n_s))
            band_pows[band_name] = bp
            feat[f"{ch}_bp_{band_name}"] = bp
            total_power += bp
        for band_name in BANDS.keys():
            denom = total_power if total_power > 0 else 1e-12
            feat[f"{ch}_relbp_{band_name}"] = band_pows[band_name] / denom
        feat[f"{ch}_total_power"] = total_power
        feat[f"{ch}_tbr"] = band_pows["theta"] / (band_pows["beta"] + 1e-12)

    for band_name in BANDS.keys():
        vals = [feat[f"{ch}_bp_{band_name}"] for ch in channels]
        feat[f"global_mean_bp_{band_name}"] = np.mean(vals)
        feat[f"global_std_bp_{band_name}"] = np.std(vals)
    tbr_vals = [feat[f"{ch}_tbr"] for ch in channels]
    feat["global_mean_tbr"] = np.mean(tbr_vals)
    feat["global_mean_spec_entropy"] = np.mean([feat[f"{ch}_spec_entropy"] for ch in channels])

    fft_feat = fft_features_epoch(epoch, channels)
    feat.update(fft_feat)

    return feat

# -------- Build epochs from CSV --------
def build_epochs_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    exclude = ["Class","ID"]
    channels = [c for c in df.columns if c not in exclude]
    print(f"Wykryto {len(channels)} kanałów EEG: {channels}")

    label_map = {"Control":0, "ADHD":1}
    y_epochs, X_features, groups = [], [], []

    subjects = list(df["ID"].unique())
    for subject_id in tqdm(subjects, desc="Przetwarzanie pacjentów"):
        sub_df = df[df["ID"] == subject_id]
        data = sub_df[channels].values.T
        n_samples = data.shape[1]
        if n_samples < EPOCH_SAMPLES:
            continue
        label = sub_df["Class"].mode()[0]
        y_val = label_map.get(label, None)
        if y_val is None: continue

        starts = range(0, n_samples - EPOCH_SAMPLES + 1, EPOCH_STEP)
        for start in starts:
            epoch = data[:, start:start+EPOCH_SAMPLES]
            feats = extract_features_epoch(epoch, channels)
            X_features.append(feats)
            y_epochs.append(y_val)
            groups.append(subject_id)

    X_df = pd.DataFrame(X_features)
    y = np.array(y_epochs)
    groups = np.array(groups)
    return X_df, y, groups

# -------- Main --------
if __name__ == "__main__":
    X_df, y, groups = build_epochs_from_csv(CSV_PATH)
    print(f"\nZbudowano {len(X_df)} epok od {len(np.unique(groups))} pacjentów.\n")

    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_df, y, groups = shuffle(X_df, y, groups, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    joblib.dump(scaler, "scaler_xgb_loso.joblib")
    pd.Series(X_df.columns).to_csv("feature_names_loso.csv", index=False)

    aucs, bal_accs, mccs = [], [], []
    feature_importances = np.zeros(X.shape[1])

    # -------- LOSO --------
    unique_subjects = np.unique(groups)

    with open("xgb_results_loso_poprawione.txt", "w", encoding="utf-8") as f:
        for i, subject in enumerate(tqdm(unique_subjects, desc="LOSO")):
            test_idx = np.where(groups == subject)[0]
            train_idx = np.where(groups != subject)[0]

            X_train_full_raw, y_train_full = X_df.values[train_idx], y[train_idx]
            X_test_raw, y_test = X_df.values[test_idx], y[test_idx]

            # szybkie info do debugowania
            f.write(
                f"\nFold {i + 1} subject={subject} train_samples={len(X_train_full_raw)} test_samples={len(X_test_raw)} "
                f"test_label_counts={np.bincount(y_test)}\n")

            # fit scaler tylko na train
            fold_scaler = StandardScaler()
            X_train_full = fold_scaler.fit_transform(X_train_full_raw)
            X_test = fold_scaler.transform(X_test_raw)

            # split train/val dla early stopping (stratify jeśli obie klasy są w train)
            stratify_arg = y_train_full if len(np.unique(y_train_full)) > 1 else None
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.1, stratify=stratify_arg, random_state=RANDOM_STATE
            )
            pos_weight = (len(y_train_full) - np.sum(y_train_full)) / (np.sum(y_train_full) + 1e-12)
            clf = XGBClassifier(
                scale_pos_weight=pos_weight,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="auc",
                tree_method="hist",  # szybsze na CPU
                n_jobs=-1,
                verbosity=0
            )

            # early stopping włączone
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[callback.EarlyStopping(
                    rounds=30,
                    save_best=True,
                    maximize=True
                )],
                verbose=False
            )

            probs = clf.predict_proba(X_test)[:, 1]
            preds = clf.predict(X_test)

            # AUC only if both classes present in y_test
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, probs)
            else:
                auc = np.nan
                f.write(f"  WARNING: only one class in test set for subject {subject}; AUC set to nan\n")

            bal_acc = balanced_accuracy_score(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)

            aucs.append(auc)
            bal_accs.append(bal_acc)
            mccs.append(mcc)

            f.write(f"Subject {subject} (Fold {i + 1}): AUC={auc if not np.isnan(auc) else 'nan'}, "
                    f"Balanced Acc={bal_acc:.3f}, MCC={mcc:.3f}\n")

            if hasattr(clf, "feature_importances_"):
                feature_importances += clf.feature_importances_

            joblib.dump(clf, os.path.join(MODEL_DIR, f"xgb_model_loso_subject{subject}.joblib"))
        feature_importances /= len(unique_subjects)
        fi_df = pd.DataFrame({
            "feature": X_df.columns,
            "importance": feature_importances
        }).sort_values("importance", ascending=False)
        fi_df.to_csv("xgb_feature_importances_loso.csv", index=False)

        f.write("\n--- PODSUMOWANIE ---\n")
        f.write(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}\n")
        f.write(f"Balanced acc: {np.mean(bal_accs):.3f}\n")
        f.write(f"MCC: {np.mean(mccs):.3f}\n")

        f.write("\nTop 20 features (by importance):\n")
        f.write(fi_df.head(20).to_string(index=False))
        f.write("\n")

    print("\n--- PODSUMOWANIE ---")
    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Balanced acc: {np.mean(bal_accs):.3f}")
    print(f"MCC: {np.mean(mccs):.3f}")

    print("\nTop 20 features (by importance):")
    print(fi_df.head(20).to_string(index=False))
