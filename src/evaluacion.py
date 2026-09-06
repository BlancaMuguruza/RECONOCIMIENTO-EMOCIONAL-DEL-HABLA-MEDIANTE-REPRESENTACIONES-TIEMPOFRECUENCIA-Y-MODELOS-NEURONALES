from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader

from . import config
from .graficas import save_confusion_matrix


# Predicciones en DataFrame para guardar metricas y revisar errores concretos.
def predict(model, dataset, device: torch.device, batch_size: int) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    df = dataset.df.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["emotion_real"] = [config.INV_LABEL_MAP[i] for i in y_true]
    df["prediction"] = [config.INV_LABEL_MAP[i] for i in y_pred]
    return df


def metrics_row(pred_df: pd.DataFrame, experiment: str, seed: int, group: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        pred_df["y_true"],
        pred_df["y_pred"],
        average="macro",
        zero_division=0,
    )
    return {
        "experiment": experiment,
        "seed": seed,
        "group": group,
        "n_datos": int(len(pred_df)),
        "accuracy": accuracy_score(pred_df["y_true"], pred_df["y_pred"]),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def class_metrics(pred_df: pd.DataFrame, experiment: str, seed: int) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=list(range(len(config.EMOTIONS))),
        zero_division=0,
    )
    rows = []
    for idx, emotion in enumerate(config.EMOTIONS):
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "emotion": emotion,
                "precision": precision[idx],
                "recall": recall[idx],
                "f1": f1[idx],
                "support": int(support[idx]),
            }
        )
    return pd.DataFrame(rows)


def save_evaluation(
    pred_df: pd.DataFrame,
    output_dir: Path,
    experiment: str,
    seed: int,
    save_predictions: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        pred_df.to_csv(output_dir / f"predicciones_seed_{seed}.csv", index=False, encoding="utf-8")

    rows = [metrics_row(pred_df, experiment, seed, "test_global")]
    for dataset_name, group_df in pred_df.groupby("dataset"):
        rows.append(metrics_row(group_df, experiment, seed, f"test_{dataset_name}"))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / f"metricas_seed_{seed}.csv", index=False, encoding="utf-8")
    class_metrics(pred_df, experiment, seed).to_csv(
        output_dir / f"metricas_clase_seed_{seed}.csv",
        index=False,
        encoding="utf-8",
    )

    report = classification_report(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=list(range(len(config.EMOTIONS))),
        target_names=config.EMOTIONS,
        zero_division=0,
    )
    (output_dir / f"classification_report_seed_{seed}.txt").write_text(report, encoding="utf-8")

    save_confusion_matrix(
        pred_df["y_true"],
        pred_df["y_pred"],
        output_dir / f"confusion_matrix_seed_{seed}.png",
        f"{experiment} seed {seed}",
        normalize=False,
    )
    save_confusion_matrix(
        pred_df["y_true"],
        pred_df["y_pred"],
        output_dir / f"confusion_matrix_norm_seed_{seed}.png",
        f"{experiment} seed {seed} normalizada",
        normalize=True,
    )
    return metrics