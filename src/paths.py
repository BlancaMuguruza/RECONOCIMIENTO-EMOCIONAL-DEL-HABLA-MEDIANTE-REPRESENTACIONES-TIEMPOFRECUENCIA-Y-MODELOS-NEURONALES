from __future__ import annotations

import os
from pathlib import Path


# Rutas calculadas desde este archivo para poder renombrar la carpeta.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


RAW_DATA_DIR = _env_path("TFG_AUDIO_DATA_DIR", WORKSPACE_ROOT / "datos_crudos")
GENERATED_DATA_DIR = PROJECT_ROOT / "datos_generados"
SPLITS_DIR = GENERATED_DATA_DIR / "splits"
FEATURES_DIR = GENERATED_DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "resultados"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

METADATA_CSV = GENERATED_DATA_DIR / "metadata_ravdess_tess_savee.csv"


def ensure_project_dirs() -> None:
    for path in [
        GENERATED_DATA_DIR,
        SPLITS_DIR,
        FEATURES_DIR,
        RESULTS_DIR,
        RESULTS_DIR / "eda",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def experiment_dir(name: str) -> Path:
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
