from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.audio import cargar_audio
from src.datos import cargar_o_crear_metadata
from src.entrenamiento import feature_tensor_path
from src.paths import FEATURES_DIR


def necesita_regenerar(path: Path, force: bool) -> bool:
    if force or not path.exists():
        return True
    tensor = torch.load(path, map_location="cpu")
    return tensor.dim() != 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera secuencias de embeddings HuBERT.")
    parser.add_argument("--force", action="store_true", help="Recalcula aunque el tensor ya exista.")
    args = parser.parse_args()

    df = cargar_o_crear_metadata()
    feature_dir = FEATURES_DIR / "hubert"
    feature_dir.mkdir(parents=True, exist_ok=True)

    missing_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="revisando hubert", unit="audio"):
        output_path = feature_tensor_path(row, feature_dir)
        if necesita_regenerar(output_path, args.force):
            missing_rows.append(row)

    if not missing_rows:
        print(f"Secuencias HuBERT ya generadas en: {feature_dir}")
        raise SystemExit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    hubert.eval()

    print(f"Generando {len(missing_rows)} secuencias HuBERT en: {feature_dir}")
    with torch.no_grad():
        for row in tqdm(missing_rows, total=len(missing_rows), desc="hubert", unit="audio"):
            output_path = feature_tensor_path(row, feature_dir)
            y = cargar_audio(row["path"])
            inputs = extractor(y, sampling_rate=config.SR, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = hubert(**inputs).last_hidden_state.squeeze(0).cpu()
            torch.save(hidden.half(), output_path)

    print("Secuencias HuBERT terminadas.")