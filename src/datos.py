from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np

from . import config
from .paths import METADATA_CSV, RAW_DATA_DIR, ensure_project_dirs


def _fila(path: Path, dataset: str, speaker_id: str, emotion: str, original_label: str) -> dict:
    return {
        "path": str(path.resolve()),
        "dataset": dataset,
        "speaker_id": str(speaker_id),
        "emotion": emotion,
        "original_label": original_label,
        "file_id": path.stem,
        "file_name": path.name,
        "split": "",
    }


# Cada parser traduce el nombre original a una fila comun de metadata.
def parse_ravdess(path: Path) -> dict | None:
    emotion_map = {"03": "happy", "04": "sad", "05": "angry", "06": "fearful"}
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    emotion = emotion_map.get(parts[2])
    if emotion is None:
        return None
    return _fila(path, "RAVDESS", parts[6], emotion, parts[2])


def parse_savee(path: Path) -> dict | None:
    emotion_map = {"a": "angry", "f": "fearful", "h": "happy", "sa": "sad"}
    match = re.match(r"([A-Z]+)_([a-z]+)(\d+)", path.stem, re.IGNORECASE)
    if not match:
        return None
    code = match.group(2).lower()
    emotion = emotion_map.get(code)
    if emotion is None:
        return None
    return _fila(path, "SAVEE", match.group(1).upper(), emotion, code)


def _normalize_tess_emotion(text: str) -> str | None:
    normalized = text.lower().replace("pleasant-surprise", "pleasant_surprise")
    normalized = normalized.replace("pleasant surprise", "pleasant_surprise")
    emotion_map = {
        "angry": "angry",
        "fear": "fearful",
        "fearful": "fearful",
        "happy": "happy",
        "sad": "sad",
    }
    return emotion_map.get(normalized)


def parse_tess(path: Path) -> dict | None:
    name_parts = path.stem.split("_")
    folder_parts = path.parent.name.split("_")
    speaker = None
    emotion = None

    if len(name_parts) >= 2 and name_parts[0].upper() in {"OAF", "YAF"}:
        speaker = name_parts[0].upper()
        emotion = _normalize_tess_emotion(name_parts[-1])

    if emotion is None and len(folder_parts) >= 2 and folder_parts[0].upper() in {"OAF", "YAF"}:
        speaker = folder_parts[0].upper()
        emotion = _normalize_tess_emotion("_".join(folder_parts[1:]))

    if speaker is None or emotion is None:
        return None
    return _fila(path, "TESS", speaker, emotion, emotion)


def _collect_wavs(folder: Path, parser) -> list[dict]:
    rows = []
    if not folder.exists():
        print(f"[AVISO] No se encuentra la carpeta: {folder}")
        return rows
    for path in sorted(folder.rglob("*.wav")):
        row = parser(path)
        if row is not None and row["emotion"] in config.EMOTIONS:
            rows.append(row)
    return rows


def crear_metadata(raw_data_dir: Path = RAW_DATA_DIR, output_csv: Path = METADATA_CSV) -> pd.DataFrame:
    ensure_project_dirs()
    rows = []
    rows.extend(_collect_wavs(raw_data_dir / "ravdess", parse_ravdess))
    rows.extend(_collect_wavs(raw_data_dir / "tess", parse_tess))
    rows.extend(_collect_wavs(raw_data_dir / "Savee", parse_savee))
    rows.extend(_collect_wavs(raw_data_dir / "savee", parse_savee))

    df = pd.DataFrame(rows).drop_duplicates("path").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No se han encontrado audios validos en {raw_data_dir}")

    df = asignar_split_aleatorio(df, seed=config.SPLIT_SEED)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


def cargar_o_crear_metadata(path: Path = METADATA_CSV) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return crear_metadata(output_csv=path)


# Split inicial aleatorio y estratificado por emocion, sin separar hablantes.
def asignar_split_aleatorio(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    df["split"] = "train"

    for _, group in df.groupby("emotion"):
        indices = rng.permutation(group.index.to_numpy())
        n_total = len(indices)
        n_test = max(1, int(round(n_total * config.TEST_SIZE)))
        n_valid = max(1, int(round(n_total * config.VALID_SIZE)))

        test_idx = indices[:n_test]
        valid_idx = indices[n_test : n_test + n_valid]

        df.loc[test_idx, "split"] = "test"
        df.loc[valid_idx, "split"] = "valid"
    return df.sort_index().reset_index(drop=True)

def split_dataframes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df[df["split"].eq("train")].reset_index(drop=True),
        df[df["split"].eq("valid")].reset_index(drop=True),
        df[df["split"].eq("test")].reset_index(drop=True),
    )


def resumen_datos(df: pd.DataFrame) -> dict:
    return {
        "n_datos": int(len(df)),
        "n_train": int((df["split"] == "train").sum()),
        "n_valid": int((df["split"] == "valid").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "datasets": sorted(df["dataset"].unique().tolist()),
        "emociones": sorted(df["emotion"].unique().tolist()),
    }


def tabla_conteos(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    return df.groupby(list(cols), observed=False).size().reset_index(name="n_datos")
