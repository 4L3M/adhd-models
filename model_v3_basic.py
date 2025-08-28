# eeg_adhd_feature_pipeline.py
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from tqdm import tqdm

# -------- CONFIG --------
CSV_PATH = "your_dataset.csv"   # <- zmień na swoją ścieżkę
FS = 128                        # sampling rate
EPOCH_SEC = 2                   # długość epoki w sekundach
EPOCH_SAMPLES = int(EPOCH_SEC * FS)
EPOCH_STEP = EPOCH_SAMPLES // 2  # 50% overlap
BANDS = {"delta":(1,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30), "gamma":(30,45)}
RANDOM_STATE = 42

path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")
print("Path to dataset:", path)
data_path = Path(path) / "adhdata.csv"
CSV_PATH = data_path

# -------- Helper functions --------
def bandpower_welch(data, fs, band, nperseg=256):
    f, Pxx = welch(data, fs=fs, nperseg=nperseg)
    low, high = band
    idx_band = np.logical_and(f >= low, f <= high)
    return np.trapz(Pxx[idx_band], f[idx_band])

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

# -------- Feature extraction per epoch --------
# def extract_features_epoch(epoch, channels):
#     n_ch, n_s = epoch.shape
#     feat = {}
#     for ci, ch in enumerate(channels):
#         x = epoch[ci, :]
#         # statystyki czasowe
#         feat[f"{ch}_mean"] = np.mean(x)
#         feat[f"{ch}_std"] = np.std(x)
#         feat[f"{ch}_var"] = np.var(x)
#         feat[f"{ch}_skew"] = skew(x)
#         feat[f"{ch}_kurt"] = kurtosis(x)
#         feat[f"{ch}_rms"] = np.sqrt(np.mean(x**2))
#         feat[f"{ch}_ptp"] = np.ptp(x)
#         feat[f"{ch}_zcr"] = zero_crossing_rate(x)
#
#         # Hjorth
#         a,m,c = hjorth_parameters(x)
#         feat[f"{ch}_hjorth_activity"] = a
#         feat[f"{ch}_hjorth_mobility"] = m
#         feat[f"{ch}_hjorth_complexity"] = c
#
#         # PSD
#         f, Pxx = welch(x, fs=FS, nperseg=min(256, n_s))
#         feat[f"{ch}_spec_entropy"] = spectral_entropy_from_psd(Pxx, f)
#         total_power = 0.0
#         band_pows = {}
#         for band_name, band_range in BANDS.items():
#             bp = bandpower_welch(x, FS, band_range, nperseg=min(256,n_s))
#             band_pows[band_name] = bp
#             feat[f"{ch}_bp_{band_name}"] = bp
#             total_power += bp
#         for band_name in BANDS.keys():
#             denom = total_power if total_power > 0 else 1e-12
#             feat[f"{ch}_relbp_{band_name}"] = band_pows[band_name] / denom
#         feat[f"{ch}_total_power"] = total_power
#
#     # global features (uśrednione po kanałach)
#     for band_name in BANDS.keys():
#         vals = [feat[f"{ch}_bp_{band_name}"] for ch in channels]
#         feat[f"global_mean_bp_{band_name}"] = np.mean(vals)
#         feat[f"global_std_bp_{band_name}"] = np.std(vals)
#     feat["global_mean_spec_entropy"] = np.mean([feat[f"{ch}_spec_entropy"] for ch in channels])
#     return feat

def extract_features_epoch(epoch, channels):
    n_ch, n_s = epoch.shape
    feat = {}
    for ci, ch in enumerate(channels):
        x = epoch[ci, :]
        # statystyki czasowe
        feat[f"{ch}_mean"] = np.mean(x)
        feat[f"{ch}_std"] = np.std(x)
        feat[f"{ch}_var"] = np.var(x)
        feat[f"{ch}_skew"] = skew(x)
        feat[f"{ch}_kurt"] = kurtosis(x)
        feat[f"{ch}_rms"] = np.sqrt(np.mean(x**2))
        feat[f"{ch}_ptp"] = np.ptp(x)
        feat[f"{ch}_zcr"] = zero_crossing_rate(x)

        # Hjorth
        a,m,c = hjorth_parameters(x)
        feat[f"{ch}_hjorth_activity"] = a
        feat[f"{ch}_hjorth_mobility"] = m
        feat[f"{ch}_hjorth_complexity"] = c

        # PSD
        f, Pxx = welch(x, fs=FS, nperseg=min(256, n_s))
        feat[f"{ch}_spec_entropy"] = spectral_entropy_from_psd(Pxx, f)
        total_power = 0.0
        band_pows = {}
        for band_name, band_range in BANDS.items():
            bp = bandpower_welch(x, FS, band_range, nperseg=min(256,n_s))
            band_pows[band_name] = bp
            feat[f"{ch}_bp_{band_name}"] = bp
            total_power += bp
        # relative bandpower
        for band_name in BANDS.keys():
            denom = total_power if total_power > 0 else 1e-12
            feat[f"{ch}_relbp_{band_name}"] = band_pows[band_name] / denom
        feat[f"{ch}_total_power"] = total_power

        # ----- Theta/Beta Ratio -----
        feat[f"{ch}_tbr"] = band_pows["theta"] / (band_pows["beta"] + 1e-12)

    # global features
    for band_name in BANDS.keys():
        vals = [feat[f"{ch}_bp_{band_name}"] for ch in channels]
        feat[f"global_mean_bp_{band_name}"] = np.mean(vals)
        feat[f"global_std_bp_{band_name}"] = np.std(vals)

    # global mean TBR
    tbr_vals = [feat[f"{ch}_tbr"] for ch in channels]
    feat["global_mean_tbr"] = np.mean(tbr_vals)
    feat["global_mean_spec_entropy"] = np.mean([feat[f"{ch}_spec_entropy"] for ch in channels])

    return feat


# -------- Build epochs from CSV --------
def build_epochs_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # automatyczne pobranie kanałów
    exclude = ["Class","ID"]
    channels = [c for c in df.columns if c not in exclude]
    print(f"Wykryto {len(channels)} kanałów EEG: {channels}")

    label_map = {"Control":0, "ADHD":1}
    y_epochs, X_features, groups = [], [], []

    for subject_id, sub_df in tqdm(df.groupby("ID")):
        data = sub_df[channels].values.T  # shape (n_channels, n_time)
        n_samples = data.shape[1]
        if n_samples < EPOCH_SAMPLES:
            continue
        label = sub_df["Class"].mode()[0]
        y_val = label_map.get(label, None)
        if y_val is None: continue

        start = 0
        while start + EPOCH_SAMPLES <= n_samples:
            epoch = data[:, start:start+EPOCH_SAMPLES]
            feats = extract_features_epoch(epoch, channels)
            X_features.append(feats)
            y_epochs.append(y_val)
            groups.append(subject_id)
            start += EPOCH_STEP

    X_df = pd.DataFrame(X_features)
    y = np.array(y_epochs)
    groups = np.array(groups)
    return X_df, y, groups

# -------- Main --------
if __name__ == "__main__":
    X_df, y, groups = build_epochs_from_csv(CSV_PATH)
    print(f"Zbudowano {len(X_df)} epok od {len(np.unique(groups))} pacjentów.")

    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    gkf = GroupKFold(n_splits=5)

    aucs, bal_accs, mccs = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[test_idx])[:,1]
        preds = clf.predict(X[test_idx])
        aucs.append(roc_auc_score(y[test_idx], probs))
        bal_accs.append(balanced_accuracy_score(y[test_idx], preds))
        mccs.append(matthews_corrcoef(y[test_idx], preds))

    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Balanced acc: {np.mean(bal_accs):.3f}")
    print(f"MCC: {np.mean(mccs):.3f}")
