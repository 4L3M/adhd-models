import matplotlib
matplotlib.use('Agg')  # tryb bez GUI, przydatny po SSH
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from datetime import datetime

if __name__ == "__main__":
    # --- 1️⃣ Wczytanie wyliczonych cech ---
    features_path = Path.home() / "tsne_results" / "features.csv"
    df = pd.read_csv(features_path)
    print(f"📂 Wczytano dane cech: {df.shape[0]} próbek, {df.shape[1]} kolumn")

    # --- 2️⃣ Przygotowanie macierzy cech dla wszystkich ---
    available_features = [c for c in df.columns if c not in ["ID", "Class"]]
    print(f"📊 Używanych cech EEG: {len(available_features)}")

    X = df[available_features].values
    y = df["Class"].values  # 0 = Control, 1 = ADHD

    # --- 3️⃣ Skalowanie ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 4️⃣ t-SNE ---
    print("\n🔄 Uruchamianie t-SNE (redukcja wymiarowości)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    print("✅ Redukcja zakończona.")

    # --- 5️⃣ Wizualizacja ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=["ADHD" if label == 1 else "Control" for label in y],
        palette={"ADHD": "crimson", "Control": "royalblue"},
        s=60,
        alpha=0.8,
        edgecolor="white"
    )

    plt.title("t-SNE redukcja wymiarowości EEG (wszyscy pacjenci, wyliczone cechy)", fontsize=14, pad=15)
    plt.xlabel("Wymiar 1 (t-SNE)")
    plt.ylabel("Wymiar 2 (t-SNE)")
    plt.legend(title="Klasa", loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # --- 6️⃣ Zapis wykresu ---
    save_dir = Path.home() / "tsne_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = save_dir / f"tsne_all_patients_features_{timestamp}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"💾 Wykres zapisano w: {save_path}\n")