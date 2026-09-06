from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.audio import cargar_audio, normalize_image_tensor, wavelet_scalogram
from src.datos import cargar_o_crear_metadata
from src.entrenamiento import feature_tensor_path
from src.paths import FEATURES_DIR


def wavelet_config_actual() -> dict:
    return {
        "wavelet_name": config.WAVELET_NAME,
        "n_scales": config.N_WAVELET_SCALES,
        "input_samples": config.WAVELET_INPUT_SAMPLES,
        "time_bins": config.WAVELET_TIME_BINS,
    }


def necesita_regenerar(feature_dir: Path, force: bool) -> bool:
    if force:
        return True
    config_path = feature_dir / "wavelet_config.json"
    if not config_path.exists():
        return True
    try:
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return saved_config != wavelet_config_actual()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera escalogramas wavelet una sola vez.")
    parser.add_argument("--force", action="store_true", help="Recalcula aunque el tensor ya exista.")
    args = parser.parse_args()

    df = cargar_o_crear_metadata()
    feature_dir = FEATURES_DIR / config.WAVELET_FEATURE_DIR
    feature_dir.mkdir(parents=True, exist_ok=True)

    regenerar = necesita_regenerar(feature_dir, args.force)
    creadas = 0
    saltadas = 0

    print(f"Generando features wavelet en: {feature_dir}")
    print(f"Configuracion: {wavelet_config_actual()}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="wavelet", unit="audio"):
        output_path = feature_tensor_path(row, feature_dir)
        if output_path.exists() and not regenerar:
            saltadas += 1
            continue

        y = cargar_audio(row["path"])
        x = normalize_image_tensor(wavelet_scalogram(y))
        torch.save(x.cpu().half(), output_path)
        creadas += 1

    (feature_dir / "wavelet_config.json").write_text(
        json.dumps(wavelet_config_actual(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Features wavelet terminadas. Creadas: {creadas}. Ya existian: {saltadas}.")