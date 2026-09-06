from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.external_datasets import EXTERNAL_METADATA_CSV, crear_metadata_externa


if __name__ == "__main__":
    df = crear_metadata_externa()
    incluidos = int(df["included"].astype(bool).sum())
    excluidos = int((~df["included"].astype(bool)).sum())
    print(f"Metadatos externos guardados en: {EXTERNAL_METADATA_CSV}")
    print(f"Audios incluidos: {incluidos}")
    print(f"Audios excluidos: {excluidos}")
