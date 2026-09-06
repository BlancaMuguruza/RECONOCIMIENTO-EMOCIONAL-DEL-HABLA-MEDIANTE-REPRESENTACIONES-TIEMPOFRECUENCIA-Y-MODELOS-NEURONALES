from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.paths import RESULTS_DIR


EXPERIMENTS = [
    "01_cnn_baseline",
    "02_cnn_data_augmentation",
    "03_cnn_scheduler",
    "04_cnn_wavelet",
    "05_hubert",
]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metricas_macro(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rows = _metricas_clase_arrays(y_true, y_pred)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "precision_macro": float(np.mean([row["precision"] for row in rows])),
        "recall_macro": float(np.mean([row["recall"] for row in rows])),
        "f1_macro": float(np.mean([row["f1"] for row in rows])),
    }


def _metricas_clase_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    for idx, emotion in enumerate(config.EMOTIONS):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        support = int((y_true == idx).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "emotion": emotion,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def _metricas_grupo(pred_df: pd.DataFrame, experiment: str, seed: int, group: str) -> dict:
    y_true = pred_df["y_true"].to_numpy(dtype=int)
    y_pred = pred_df["y_pred"].to_numpy(dtype=int)
    metrics = _metricas_macro(y_true, y_pred)
    return {
        "experiment": experiment,
        "seed": seed,
        "group": group,
        "n_datos": int(len(pred_df)),
        **metrics,
    }


def _classification_report_text(pred_df: pd.DataFrame) -> str:
    y_true = pred_df["y_true"].to_numpy(dtype=int)
    y_pred = pred_df["y_pred"].to_numpy(dtype=int)
    rows = _metricas_clase_arrays(y_true, y_pred)
    lines = ["              precision    recall  f1-score   support", ""]
    for row in rows:
        lines.append(
            f"{row['emotion']:>12}       {row['precision']:0.2f}      {row['recall']:0.2f}      {row['f1']:0.2f}{row['support']:10d}"
        )
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    total = int(len(y_true))
    macro = _metricas_macro(y_true, y_pred)
    weighted_precision = _safe_div(sum(row["precision"] * row["support"] for row in rows), total)
    weighted_recall = _safe_div(sum(row["recall"] * row["support"] for row in rows), total)
    weighted_f1 = _safe_div(sum(row["f1"] * row["support"] for row in rows), total)
    lines.extend(
        [
            "",
            f"    accuracy                           {accuracy:0.2f}{total:10d}",
            f"   macro avg       {macro['precision_macro']:0.2f}      {macro['recall_macro']:0.2f}      {macro['f1_macro']:0.2f}{total:10d}",
            f"weighted avg       {weighted_precision:0.2f}      {weighted_recall:0.2f}      {weighted_f1:0.2f}{total:10d}",
            "",
        ]
    )
    return "\n".join(lines)


def _guardar_matrices_si_se_puede(pred_df: pd.DataFrame, seed_dir: Path, experiment: str, seed: int) -> bool:
    try:
        from src.graficas import save_confusion_matrix
    except Exception:
        return False

    save_confusion_matrix(
        pred_df["y_true"],
        pred_df["y_pred"],
        seed_dir / f"confusion_matrix_seed_{seed}.png",
        f"{experiment} seed {seed}",
        normalize=False,
    )
    save_confusion_matrix(
        pred_df["y_true"],
        pred_df["y_pred"],
        seed_dir / f"confusion_matrix_norm_seed_{seed}.png",
        f"{experiment} seed {seed} normalizada",
        normalize=True,
    )
    return True


def regenerar_semilla(experiment: str, seed: int) -> tuple[Path, bool]:
    seed_dir = RESULTS_DIR / experiment / f"seed_{seed}"
    pred_path = seed_dir / f"predicciones_seed_{seed}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"No se encuentra: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    rows = [_metricas_grupo(pred_df, experiment, seed, "test_global")]
    for dataset_name, group_df in pred_df.groupby("dataset"):
        rows.append(_metricas_grupo(group_df, experiment, seed, f"test_{dataset_name}"))
    pd.DataFrame(rows).to_csv(seed_dir / f"metricas_seed_{seed}.csv", index=False, encoding="utf-8")

    class_rows = []
    for row in _metricas_clase_arrays(pred_df["y_true"].to_numpy(dtype=int), pred_df["y_pred"].to_numpy(dtype=int)):
        class_rows.append({"experiment": experiment, "seed": seed, **row})
    pd.DataFrame(class_rows).to_csv(seed_dir / f"metricas_clase_seed_{seed}.csv", index=False, encoding="utf-8")

    (seed_dir / f"classification_report_seed_{seed}.txt").write_text(
        _classification_report_text(pred_df),
        encoding="utf-8",
    )
    matrices_ok = _guardar_matrices_si_se_puede(pred_df, seed_dir, experiment, seed)
    return seed_dir, matrices_ok


if __name__ == "__main__":
    matrices_generadas = True
    for experiment in EXPERIMENTS:
        for seed in config.SEEDS:
            out_dir, matrices_ok = regenerar_semilla(experiment, seed)
            matrices_generadas = matrices_generadas and matrices_ok
            print(f"Metricas internas regeneradas en: {out_dir}")
    if not matrices_generadas:
        print("Aviso: no se han regenerado matrices porque matplotlib/seaborn no estan disponibles en este entorno.")