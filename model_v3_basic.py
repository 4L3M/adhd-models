from pathlib import Path

from tqdm import tqdm
import kagglehub
import numpy as np
from numpy.fft import rfft, rfftfreq
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef

# 2 sekundy epoch:
# AUC: nan ± nan
# Balanced acc: 0.786
# MCC: 0.000

# ----
# 4 sek
#   var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
# AUC: nan ± nan
# Balanced acc: 0.787
# MCC: 0.000

# -------- CONFIG --------
FS = 128  # sampling rate
EPOCH_SEC = 4                   # długość epoki w sekundach
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

        for band_name, (low, high) in BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_energy = np.sum(fft_vals[idx]**2)
            total_energy = np.sum(fft_vals**2)
            feat[f"{ch}_fft_energy_{band_name}"] = band_energy
            feat[f"{ch}_fft_rel_energy_{band_name}"] = band_energy / (total_energy + 1e-12)
    return feat


def extract_features_epoch(epoch, channels):
    n_ch, n_s = epoch.shape
    feat = {}
    for ci, ch in enumerate(channels):
        x = epoch[ci, :]

        # -------- Statystyki czasowe --------
        feat[f"{ch}_mean"] = np.mean(x)
        feat[f"{ch}_std"] = np.std(x)
        feat[f"{ch}_var"] = np.var(x)
        feat[f"{ch}_skew"] = skew(x)
        feat[f"{ch}_kurt"] = kurtosis(x)
        feat[f"{ch}_rms"] = np.sqrt(np.mean(x**2))
        feat[f"{ch}_ptp"] = np.ptp(x)
        feat[f"{ch}_zcr"] = zero_crossing_rate(x)

        # -------- Hjorth --------
        a,m,c = hjorth_parameters(x)
        feat[f"{ch}_hjorth_activity"] = a
        feat[f"{ch}_hjorth_mobility"] = m
        feat[f"{ch}_hjorth_complexity"] = c

        # -------- PSD --------
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

        # -------- Theta/Beta Ratio --------
        feat[f"{ch}_tbr"] = band_pows["theta"] / (band_pows["beta"] + 1e-12)

    # -------- Global features --------
    for band_name in BANDS.keys():
        vals = [feat[f"{ch}_bp_{band_name}"] for ch in channels]
        feat[f"global_mean_bp_{band_name}"] = np.mean(vals)
        feat[f"global_std_bp_{band_name}"] = np.std(vals)
    tbr_vals = [feat[f"{ch}_tbr"] for ch in channels]
    feat["global_mean_tbr"] = np.mean(tbr_vals)
    feat["global_mean_spec_entropy"] = np.mean([feat[f"{ch}_spec_entropy"] for ch in channels])

    # -------- FFT features --------
    fft_feat = fft_features_epoch(epoch, channels)
    feat.update(fft_feat)

    return feat


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
        data = sub_df[channels].values.T  # shape (n_channels, n_time)
        n_samples = data.shape[1]
        if n_samples < EPOCH_SAMPLES:
            continue
        label = sub_df["Class"].mode()[0]
        y_val = label_map.get(label, None)
        if y_val is None: continue

        starts = range(0, n_samples - EPOCH_SAMPLES + 1, EPOCH_STEP)
        for start in tqdm(starts, desc=f"Epoki dla pacjenta {subject_id}", leave=False):
            epoch = data[:, start:start+EPOCH_SAMPLES]
            feats = extract_features_epoch(epoch, channels)
            X_features.append(feats)
            y_epochs.append(y_val)
            groups.append(subject_id)

    X_df = pd.DataFrame(X_features)
    y = np.array(y_epochs)
    groups = np.array(groups)
    return X_df, y, groups

if __name__ == "__main__":
    X_df, y, groups = build_epochs_from_csv(CSV_PATH)
    print(f"\nZbudowano {len(X_df)} epok od {len(np.unique(groups))} pacjentów.\n")

    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    subjects = np.unique(groups)
    aucs, bal_accs, mccs = [], [], []

    for i, subject_id in enumerate(tqdm(subjects, desc="LOSO CV")):
        test_idx = np.where(groups == subject_id)[0]
        train_idx = np.where(groups != subject_id)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:,1]
        preds = clf.predict(X_test)

        # Uwaga: przy LOSO może być tylko jedna klasa w testowym foldzie, wtedy AUC/MCC nie mają sensu
        if len(np.unique(y_test)) == 1:
            auc = np.nan
            mcc = 0.0
        else:
            auc = roc_auc_score(y_test, probs)
            mcc = matthews_corrcoef(y_test, preds)

        bal_acc = balanced_accuracy_score(y_test, preds)
        aucs.append(auc)
        bal_accs.append(bal_acc)
        mccs.append(mcc)

        print(f"Subject {i+1}/{len(subjects)}: AUC={auc:.3f}, Balanced Acc={bal_acc:.3f}, MCC={mcc:.3f}")

    print("\n--- PODSUMOWANIE ---")
    print(f"AUC: {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}")
    print(f"Balanced acc: {np.mean(bal_accs):.3f}")
    print(f"MCC: {np.mean(mccs):.3f}")