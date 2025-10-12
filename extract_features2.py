import kagglehub
import pandas as pd
import numpy as np
from scipy.signal import welch
from pathlib import Path
from tqdm import tqdm

# --- KONFIGURACJA ---
FS = 128                # częstotliwość próbkowania EEG (Hz)
EPOCH_SEC = 4           # długość okna w sekundach
EPOCH_SAMPLES = FS * EPOCH_SEC
EPOCH_STEP = EPOCH_SAMPLES // 2  # przesuw o połowę okna
BANDS = {
    "theta": (4, 8),
    "beta": (13, 30)
}

def bandpower_welch(data, fs, band, nperseg=256):
    f, Pxx = welch(data, fs=fs, nperseg=nperseg)
    low, high = band
    idx = np.logical_and(f >= low, f <= high)
    return np.trapz(Pxx[idx], f[idx])

def extract_features_epoch(epoch, channels):
    """Liczy cechy tylko dla theta, beta i TBR, plus FFT."""
    n_ch, n_s = epoch.shape
    feat = {}

    for ci, ch in enumerate(channels):
        x = epoch[ci, :]

        # moc w wybranych pasmach
        band_pows = {}
        total_power = 0.0
        for band_name, band_range in BANDS.items():
            bp = bandpower_welch(x, FS, band_range)
            band_pows[band_name] = bp
            feat[f"{ch}_bp_{band_name}"] = bp
            total_power += bp

        # względne moce
        for band_name in BANDS.keys():
            feat[f"{ch}_relbp_{band_name}"] = band_pows[band_name] / (total_power + 1e-12)

        # TBR
        feat[f"{ch}_TBR"] = band_pows["theta"] / (band_pows["beta"] + 1e-12)

        # FFT amplitudy (proste spektrum)
        fft_vals = np.abs(np.fft.rfft(x))
        feat[f"{ch}_fft_mean"] = np.mean(fft_vals)
        feat[f"{ch}_fft_std"] = np.std(fft_vals)

    return feat

def build_features_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    exclude = ["Class", "ID"]
    channels = [c for c in df.columns if c not in exclude]
    print(f"🔍 Wykryto {len(channels)} kanałów EEG: {channels}")

    label_map = {"Control": 0, "ADHD": 1}
    subjects = list(df["ID"].unique())

    features = []

    for subject_id in tqdm(subjects, desc="📊 Przetwarzanie pacjentów"):
        sub_df = df[df["ID"] == subject_id]
        data = sub_df[channels].values.T
        n_samples = data.shape[1]
        if n_samples < EPOCH_SAMPLES:
            continue

        label = sub_df["Class"].mode()[0]
        y_val = label_map.get(label, None)
        if y_val is None:
            continue

        starts = range(0, n_samples - EPOCH_SAMPLES + 1, EPOCH_STEP)
        for start in starts:
            epoch = data[:, start:start + EPOCH_SAMPLES]
            feats = extract_features_epoch(epoch, channels)
            feats["ID"] = subject_id
            feats["Class"] = y_val
            features.append(feats)

    X_df = pd.DataFrame(features)
    print(f"✅ Zbudowano {len(X_df)} epok z {len(subjects)} pacjentów.")
    return X_df

if __name__ == "__main__":
    path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")
    print("📂 Path to dataset:", path)

    data_path = Path(path) / "adhdata.csv"

    # --- Wyliczenie cech ---
    features_df = build_features_from_csv(data_path)

    # katalog docelowy: tsne_results
    save_dir = Path.home() / "tsne_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    # plik wynikowy CSV
    output_path = save_dir / "features2.csv"
    features_df.to_csv(output_path, index=False)

    print(f"💾 Zapisano cechy EEG w: {output_path.resolve()}")
