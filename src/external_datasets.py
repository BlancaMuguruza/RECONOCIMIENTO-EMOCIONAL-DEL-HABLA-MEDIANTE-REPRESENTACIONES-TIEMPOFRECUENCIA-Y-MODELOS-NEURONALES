from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import config
from .paths import PROJECT_ROOT, WORKSPACE_ROOT


EXTERNAL_RESULTS_DIR = PROJECT_ROOT / "resultados" / "08_evaluacion_externa"
EXTERNAL_METADATA_CSV = EXTERNAL_RESULTS_DIR / "metadata_externa.csv"
EXTERNAL_CHECKS_JSON = EXTERNAL_RESULTS_DIR / "metadata_checks.json"
CREMAD_DIR = WORKSPACE_ROOT / "CREMAD" / "AudioWAV"
EMODB_DIR = WORKSPACE_ROOT / "EMODB" / "wav"
EXTERNAL_DATASETS = ["CREMA-D", "EMO-DB"]
TARGET_EMOTIONS = config.EMOTIONS

CREMAD_MAP = {"ANG": "angry", "FEA": "fearful", "HAP": "happy", "SAD": "sad"}
EMODB_MAP = {"W": "angry", "A": "fearful", "F": "happy", "T": "sad"}


def _audio_info(path: Path) -> dict:
    info = sf.info(str(path))
    duration = float(info.frames / info.samplerate) if info.samplerate else np.nan
    return {
        "sample_rate_original": int(info.samplerate),
        "n_channels": int(info.channels),
        "duration_seconds": duration,
    }


def _base_row(path: Path, dataset: str) -> dict:
    try:
        info = _audio_info(path)
        audio_error = ""
    except Exception as exc:
        info = {"sample_rate_original": np.nan, "n_channels": np.nan, "duration_seconds": np.nan}
        audio_error = f"audio_error: {exc}"
    return {
        "dataset": dataset,
        "file_id": path.stem,
        "path": str(path.resolve()),
        "speaker_id": "",
        "emotion_original": "",
        "emotion": "",
        "sample_rate_original": info["sample_rate_original"],
        "n_channels": info["n_channels"],
        "duration_seconds": info["duration_seconds"],
        "included": False,
        "exclusion_reason": audio_error,
    }


def parse_cremad(path: Path) -> dict:
    row = _base_row(path, "CREMA-D")
    parts = path.stem.split("_")
    if len(parts) < 4:
        row["exclusion_reason"] = row["exclusion_reason"] or "patron_no_reconocido"
        return row

    row["speaker_id"] = f"CREMA-D_{parts[0]}"
    row["emotion_original"] = parts[2].upper()
    emotion = CREMAD_MAP.get(row["emotion_original"])
    if emotion is None:
        row["exclusion_reason"] = row["exclusion_reason"] or f"emocion_excluida_{row['emotion_original']}"
        return row

    row["emotion"] = emotion
    row["included"] = row["exclusion_reason"] == ""
    return row


def parse_emodb(path: Path) -> dict:
    row = _base_row(path, "EMO-DB")
    stem = path.stem
    match = re.match(r"^(\d{2})[a-z]\d{2}([A-Z])", stem, re.IGNORECASE)
    if not match:
        row["exclusion_reason"] = row["exclusion_reason"] or "patron_no_reconocido"
        return row

    row["speaker_id"] = f"EMO-DB_{match.group(1)}"
    row["emotion_original"] = match.group(2).upper()
    emotion = EMODB_MAP.get(row["emotion_original"])
    if emotion is None:
        row["exclusion_reason"] = row["exclusion_reason"] or f"emocion_excluida_{row['emotion_original']}"
        return row

    row["emotion"] = emotion
    row["included"] = row["exclusion_reason"] == ""
    return row


def crear_metadata_externa() -> pd.DataFrame:
    rows = []
    for path in sorted(CREMAD_DIR.glob("*.wav")):
        rows.append(parse_cremad(path))
    for path in sorted(EMODB_DIR.glob("*.wav")):
        rows.append(parse_emodb(path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No se han encontrado audios externos.")

    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    finite_duration = np.isfinite(df["duration_seconds"])
    df.loc[~finite_duration, "included"] = False
    df.loc[~finite_duration & df["exclusion_reason"].eq(""), "exclusion_reason"] = "duracion_no_finita"

    EXTERNAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTERNAL_METADATA_CSV, index=False, encoding="utf-8")
    checks = comprobar_metadata_externa(df)
    EXTERNAL_CHECKS_JSON.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    guardar_mapeo_emociones()
    return df


def cargar_metadata_externa() -> pd.DataFrame:
    if EXTERNAL_METADATA_CSV.exists():
        return pd.read_csv(EXTERNAL_METADATA_CSV)
    return crear_metadata_externa()


def metadata_incluida() -> pd.DataFrame:
    df = cargar_metadata_externa()
    included = df[df["included"].astype(bool)].copy().reset_index(drop=True)
    assert set(included["dataset"].unique()) <= set(EXTERNAL_DATASETS)
    assert set(included["emotion"].unique()) <= set(TARGET_EMOTIONS)
    assert included["path"].is_unique
    return included


def comprobar_metadata_externa(df: pd.DataFrame) -> dict:
    included = df[df["included"].astype(bool)].copy()
    checks = {
        "n_total": int(len(df)),
        "n_included": int(len(included)),
        "n_excluded": int((~df["included"].astype(bool)).sum()),
        "datasets": sorted(df["dataset"].unique().tolist()),
        "included_datasets": sorted(included["dataset"].unique().tolist()),
        "included_emotions": sorted(included["emotion"].dropna().unique().tolist()),
        "duplicated_paths": int(df["path"].duplicated().sum()),
        "unrecognized_files": int(df["exclusion_reason"].fillna("").str.contains("patron_no_reconocido").sum()),
        "audio_errors": int(df["exclusion_reason"].fillna("").str.contains("audio_error").sum()),
        "unknown_or_excluded_labels": int(df["exclusion_reason"].fillna("").str.contains("emocion_excluida").sum()),
        "non_finite_duration": int((~np.isfinite(pd.to_numeric(df["duration_seconds"], errors="coerce"))).sum()),
        "n_speakers_by_dataset": included.groupby("dataset")["speaker_id"].nunique().to_dict(),
        "n_included_by_dataset": included["dataset"].value_counts().to_dict(),
        "n_excluded_by_reason": df.loc[~df["included"].astype(bool), "exclusion_reason"].value_counts().to_dict(),
    }
    if checks["duplicated_paths"] != 0:
        raise ValueError("Hay rutas duplicadas en los metadatos externos.")
    if set(included["dataset"].unique()) - set(EXTERNAL_DATASETS):
        raise ValueError("Se han incluido corpus externos no permitidos.")
    if set(included["emotion"].unique()) - set(TARGET_EMOTIONS):
        raise ValueError("Se han incluido emociones no objetivo.")
    return checks


def guardar_mapeo_emociones() -> Path:
    mapping = []
    for original, emotion in CREMAD_MAP.items():
        mapping.append({"dataset": "CREMA-D", "emotion_original": original, "emotion": emotion, "included": True})
    for original, emotion in EMODB_MAP.items():
        mapping.append({"dataset": "EMO-DB", "emotion_original": original, "emotion": emotion, "included": True})
    path = EXTERNAL_RESULTS_DIR / "mapeo_emociones.csv"
    pd.DataFrame(mapping).to_csv(path, index=False, encoding="utf-8")
    return path
