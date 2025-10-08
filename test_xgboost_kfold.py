import pandas as pd
import numpy as np
import joblib
import random
from pathlib import Path
from tqdm import tqdm

# Import funkcji i stałych z Twojego pliku trenowania
from model_v7_xgboost_kfold import (
    extract_features_epoch,
    FS, EPOCH_SAMPLES, EPOCH_STEP, CSV_PATH
)

# ---------------- CONFIG ----------------
SCALER_PATH = "scaler_xgb_kfold.joblib"
MODEL_PATH = "xgb_models/xgb_model_kfold_fold1.joblib"  # wybierz np. fold1

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # wczytaj dane
    df = pd.read_csv(CSV_PATH)
    subjects = df["ID"].unique().tolist()

    # losowy pacjent
    patient_id = random.choice(subjects)
    print(f"\nWybrano pacjenta: {patient_id}")

    # dane pacjenta
    sub_df = df[df["ID"] == patient_id]
    channels = [c for c in df.columns if c not in ["Class", "ID"]]
    true_label = sub_df["Class"].mode()[0]
    print(f"Prawdziwa etykieta pacjenta: {true_label}")

    # budowanie epok
    data = sub_df[channels].values.T
    n_samples = data.shape[1]
    epochs = []
    starts = range(0, n_samples - EPOCH_SAMPLES + 1, EPOCH_STEP)
    for start in starts:
        epoch = data[:, start:start+EPOCH_SAMPLES]
        feats = extract_features_epoch(epoch, channels)
        epochs.append(feats)

    if not epochs:
        print("Za mało próbek EEG dla tego pacjenta, pomiń test.")
        exit()

    X_df = pd.DataFrame(epochs)
    print(f"Zbudowano {len(X_df)} epok dla pacjenta.")

    # wczytaj scaler i model
    scaler = joblib.load(SCALER_PATH)
    clf = joblib.load(MODEL_PATH)

    # uzupełnij brakujące cechy (jeśli model oczekuje innego zestawu kolumn)
    feature_names = pd.read_csv("feature_names_kfold.csv")["0"].tolist()
    for col in feature_names:
        if col not in X_df.columns:
            X_df[col] = 0.0
    X_df = X_df[feature_names]

    # przygotowanie danych
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X_df.values)

    # predykcja
    probs = clf.predict_proba(X_scaled)[:, 1]
    preds = clf.predict(X_scaled)

    # wynik dla epok
    print("\n--- Wyniki epok ---")
    for i, (p, pr) in enumerate(zip(preds, probs)):
        print(f"Epoka {i+1:02d}: pred={pr:.3f} -> {'ADHD' if p==1 else 'Control'}")

    # wynik końcowy (średnia prawdopodobieństw + głosowanie większościowe)
    mean_prob = np.mean(probs)
    majority_vote = int(np.round(np.mean(preds)))

    print("\n--- Podsumowanie pacjenta ---")
    print(f"Średnie prawdopodobieństwo ADHD: {mean_prob:.3f}")
    print(f"Wynik większościowy: {'ADHD' if majority_vote==1 else 'Control'}")
    print(f"Prawdziwa etykieta: {true_label}")
