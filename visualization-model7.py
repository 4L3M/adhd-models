import matplotlib
matplotlib.use('Agg')  # ważne: tryb bez GUI, potrzebny po SSH
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import kagglehub
from tqdm import tqdm
import time
from datetime import datetime

def visualize_tsne(X, y, save_dir: Path):
    print(f"\n📏 Używanych próbek: {len(X)}")

    # --- Udział klas ---
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print("\n📊 Udział klas w podzbiorze:")
    for cls, cnt in zip(unique, counts):
        percent = 100 * cnt / total
        label_name = "ADHD" if cls == 1 else "Nie-ADHD"
        print(f" - {label_name}: {cnt} próbek ({percent:.1f}%)")

    # --- Redukcja wymiarowości ---
    print("\n🔄 Uruchamianie t-SNE (może potrwać kilka minut)...")
    time.sleep(0.5)

    with tqdm(total=100, desc="t-SNE") as pbar:
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30,
            max_iter=1000,
            verbose=0
        )

        for _ in range(10):
            time.sleep(0.2)
            pbar.update(10)

        X_tsne = tsne.fit_transform(X)
        pbar.update(100)

    print("✅ Zakończono redukcję wymiarowości. Rysuję wykres...")

    # --- Wykres t-SNE ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=["ADHD" if label == 1 else "Nie-ADHD" for label in y],
        palette={"ADHD": "crimson", "Nie-ADHD": "royalblue"},
        s=60,
        alpha=0.8,
        edgecolor="white"
    )
    plt.title("Redukcja wymiarowości EEG przy użyciu t-SNE (cały zbiór)", fontsize=14, pad=15)
    plt.xlabel("Wymiar 1 (t-SNE)", fontsize=12)
    plt.ylabel("Wymiar 2 (t-SNE)", fontsize=12)
    plt.legend(title="Klasa", loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # --- Zapis wykresu ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = save_dir / f"tsne_plot_{timestamp}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"💾 Wykres zapisano w: {save_path}\n")


if __name__ == '__main__':
    print("🚀 Uruchamianie wizualizacji EEG t-SNE dla całego zbioru...\n")
    path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")
    print("📂 Path to dataset:", path)

    data_path = Path(path) / "adhdata.csv"
    df = pd.read_csv(data_path)

    # --- Podsumowanie całego zbioru ---
    n_samples = len(df)
    classes = df['Class'].value_counts()
    print("\n--- PODSUMOWANIE ZBIORU ---")
    print(f"📊 Liczba wszystkich próbek: {n_samples}")
    print("🔍 Liczba próbek w każdej klasie:")
    for cls, count in classes.items():
        print(f"   - {cls}: {count}")
    print("---------------------------\n")

    # --- Przygotowanie danych ---
    X = df.drop(columns=['Class', 'ID']).values
    y = df['Class'].apply(lambda c: 1 if c == 'ADHD' else 0).values

    # --- Skalowanie cech ---
    print("⚙️  Skalowanie cech EEG...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Skalowanie zakończone!")

    # --- Katalog zapisu ---
    save_dir = Path.home() / "tsne_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Wyniki będą zapisane w: {save_dir}\n")

    # --- Wizualizacja t-SNE ---
    visualize_tsne(X_scaled, y, save_dir)
