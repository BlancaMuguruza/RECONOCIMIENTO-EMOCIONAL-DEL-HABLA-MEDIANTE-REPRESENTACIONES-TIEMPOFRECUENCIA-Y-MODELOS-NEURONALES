from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "train_01_cnn_baseline.py",
    "train_02_cnn_data_augmentation.py",
    "train_03_cnn_scheduler.py",
    "generate_wavelet_features.py",
    "train_04_cnn_wavelet.py",
    "generate_hubert_embeddings.py",
    "train_05_hubert.py",
]


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\nEjecutando {script}\n")
        subprocess.run([sys.executable, str(here / script)], check=True)