from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import config
from .datos import cargar_o_crear_metadata
from .entrenamiento import TensorFeatureDataset, _class_weight_sampler, _run_epoch, _validate, feature_tensor_path
from .graficas import save_confusion_matrix, save_loss_curve
from .modelos import MLPHuBERT
from .paths import FEATURES_DIR, RESULTS_DIR
from .utilidades import contar_parametros, fijar_semilla, guardar_json


EXPERIMENT_NAME = "06_hubert_ravdess_savee"
PROTOCOLS = ["random_audio", "speaker_disjoint"]
SPLIT_SEEDS = [42, 123, 2024]
MODEL_SEED = 42
TARGET_DATASETS = ["RAVDESS", "SAVEE"]
TARGET_EMOTIONS = ["angry", "fearful", "happy", "sad"]
RESULTS_ROOT = RESULTS_DIR / EXPERIMENT_NAME


def _normalizar_speaker_id(row: pd.Series) -> str:
    dataset = str(row["dataset"]).upper()
    speaker = str(row["speaker_id"]).strip().upper()
    if dataset == "RAVDESS":
        speaker = speaker.zfill(2)
    return f"{dataset}_{speaker}"


def cargar_metadata_experimento() -> pd.DataFrame:
    df = cargar_o_crear_metadata().copy()
    df = df[df["dataset"].isin(TARGET_DATASETS) & df["emotion"].isin(TARGET_EMOTIONS)].copy()
    df["speaker_id_original"] = df["speaker_id"].astype(str)
    df["speaker_id"] = df.apply(_normalizar_speaker_id, axis=1)
    df["embedding_path"] = df.apply(lambda row: str(feature_tensor_path(row, FEATURES_DIR / "hubert")), axis=1)
    df = df.sort_values(["dataset", "speaker_id", "emotion", "file_id"]).reset_index(drop=True)
    return df


def comprobar_metadata_y_embeddings(df: pd.DataFrame) -> dict:
    checks = {
        "n_audios": int(len(df)),
        "datasets": sorted(df["dataset"].unique().tolist()),
        "emotions": sorted(df["emotion"].unique().tolist()),
        "n_speakers_by_dataset": df.groupby("dataset")["speaker_id"].nunique().to_dict(),
        "ravdess_24_speakers": int(df[df["dataset"].eq("RAVDESS")]["speaker_id"].nunique()) == 24,
        "savee_4_speakers": int(df[df["dataset"].eq("SAVEE")]["speaker_id"].nunique()) == 4,
        "speaker_id_nulls": int(df["speaker_id"].isna().sum()),
        "speaker_id_collisions_between_corpora": False,
        "only_target_datasets": set(df["dataset"].unique()) <= set(TARGET_DATASETS),
        "only_target_emotions": set(df["emotion"].unique()) <= set(TARGET_EMOTIONS),
        "embedding_paths_unique": bool(df["embedding_path"].is_unique),
        "path_unique": bool(df["path"].is_unique),
    }

    speakers_by_dataset = df.groupby("dataset")["speaker_id"].apply(set).to_dict()
    if len(speakers_by_dataset) == 2:
        checks["speaker_id_collisions_between_corpora"] = bool(
            speakers_by_dataset.get("RAVDESS", set()) & speakers_by_dataset.get("SAVEE", set())
        )

    missing_embeddings = [path for path in df["embedding_path"] if not Path(path).exists()]
    checks["missing_embeddings"] = missing_embeddings[:10]
    checks["n_missing_embeddings"] = len(missing_embeddings)

    rows = []
    for speaker_id, speaker_df in df.groupby("speaker_id"):
        missing = sorted(set(TARGET_EMOTIONS) - set(speaker_df["emotion"]))
        if missing:
            rows.append({"speaker_id": speaker_id, "missing_emotions": ", ".join(missing)})
    checks["speakers_missing_emotions"] = rows

    if not checks["only_target_datasets"]:
        raise ValueError("El experimento 06 contiene corpus fuera de RAVDESS y SAVEE.")
    if not checks["only_target_emotions"]:
        raise ValueError("El experimento 06 contiene emociones fuera de las cuatro objetivo.")
    if checks["speaker_id_nulls"] != 0:
        raise ValueError("Hay speaker_id nulos en el experimento 06.")
    if checks["speaker_id_collisions_between_corpora"]:
        raise ValueError("Hay colisiones de speaker_id entre corpus.")
    if checks["n_missing_embeddings"] != 0:
        raise FileNotFoundError("Faltan embeddings HuBERT para audios del experimento 06.")
    if not checks["embedding_paths_unique"]:
        raise ValueError("Hay rutas de embedding duplicadas.")
    return checks


def _asignar_random_audio(df: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(split_seed)
    out = df.copy()
    out["split"] = "train"
    ratios = {"RAVDESS": 3 / 24, "SAVEE": 1 / 4}

    for (dataset, emotion), group in out.groupby(["dataset", "emotion"]):
        idx = rng.permutation(group.index.to_numpy())
        n_valid = max(1, int(round(len(idx) * ratios[dataset])))
        n_test = max(1, int(round(len(idx) * ratios[dataset])))
        out.loc[idx[:n_valid], "split"] = "valid"
        out.loc[idx[n_valid : n_valid + n_test], "split"] = "test"
    return out.sort_index().reset_index(drop=True)


def _asignar_speaker_disjoint(df: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(split_seed)
    out = df.copy()
    out["split"] = ""
    plan = {"RAVDESS": (18, 3, 3), "SAVEE": (2, 1, 1)}

    for dataset, (n_train, n_valid, n_test) in plan.items():
        speakers = np.array(sorted(out.loc[out["dataset"].eq(dataset), "speaker_id"].unique()))
        speakers = rng.permutation(speakers)
        train = set(speakers[:n_train])
        valid = set(speakers[n_train : n_train + n_valid])
        test = set(speakers[n_train + n_valid : n_train + n_valid + n_test])
        out.loc[out["speaker_id"].isin(train), "split"] = "train"
        out.loc[out["speaker_id"].isin(valid), "split"] = "valid"
        out.loc[out["speaker_id"].isin(test), "split"] = "test"

    if out["split"].eq("").any():
        raise ValueError("Hay audios sin split en la particion por hablante.")
    return out.sort_index().reset_index(drop=True)


def crear_split(df: pd.DataFrame, protocol: str, split_seed: int) -> pd.DataFrame:
    if protocol == "random_audio":
        split_df = _asignar_random_audio(df, split_seed)
    elif protocol == "speaker_disjoint":
        split_df = _asignar_speaker_disjoint(df, split_seed)
    else:
        raise ValueError(f"Protocolo no soportado: {protocol}")
    split_df["protocol"] = protocol
    split_df["split_seed"] = split_seed
    split_df["model_seed"] = MODEL_SEED
    return split_df


def speaker_assignment(split_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, speaker_id), group in split_df.groupby(["dataset", "speaker_id"]):
        splits = sorted(group["split"].unique().tolist())
        rows.append(
            {
                "dataset": dataset,
                "speaker_id": speaker_id,
                "split": ", ".join(splits),
                "n_audios": int(len(group)),
                "emotions": ", ".join(sorted(group["emotion"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "speaker_id"]).reset_index(drop=True)


def comprobar_split(split_df: pd.DataFrame, protocol: str) -> dict:
    checks = {
        "protocol": protocol,
        "split_seed": int(split_df["split_seed"].iloc[0]),
        "model_seed": MODEL_SEED,
        "n_audios_by_split": split_df["split"].value_counts().reindex(["train", "valid", "test"], fill_value=0).to_dict(),
        "n_speakers_by_split": split_df.groupby("split")["speaker_id"].nunique().reindex(["train", "valid", "test"], fill_value=0).to_dict(),
        "corpus_by_split": split_df.groupby("split")["dataset"].apply(lambda x: sorted(x.unique().tolist())).to_dict(),
        "emotions_by_split": split_df.groupby("split")["emotion"].apply(lambda x: sorted(x.unique().tolist())).to_dict(),
        "tess_present": bool(split_df["dataset"].eq("TESS").any()),
        "only_target_emotions": set(split_df["emotion"].unique()) <= set(TARGET_EMOTIONS),
    }
    split_speakers = {
        split: set(split_df.loc[split_df["split"].eq(split), "speaker_id"])
        for split in ["train", "valid", "test"]
    }
    checks["speaker_overlap_train_valid"] = sorted(split_speakers["train"] & split_speakers["valid"])
    checks["speaker_overlap_train_test"] = sorted(split_speakers["train"] & split_speakers["test"])
    checks["speaker_overlap_valid_test"] = sorted(split_speakers["valid"] & split_speakers["test"])

    for split in ["train", "valid", "test"]:
        if set(checks["emotions_by_split"].get(split, [])) != set(TARGET_EMOTIONS):
            raise ValueError(f"Faltan emociones objetivo en el split {split} de {protocol}.")
        if set(checks["corpus_by_split"].get(split, [])) != set(TARGET_DATASETS):
            raise ValueError(f"Falta algun corpus objetivo en el split {split} de {protocol}.")

    if checks["tess_present"]:
        raise ValueError("TESS aparece en el experimento 06.")
    if not checks["only_target_emotions"]:
        raise ValueError("Aparecen emociones no objetivo en el experimento 06.")
    if protocol == "speaker_disjoint":
        if any(checks[key] for key in ["speaker_overlap_train_valid", "speaker_overlap_train_test", "speaker_overlap_valid_test"]):
            raise ValueError("Hay fuga de hablantes entre splits.")
        if checks["n_speakers_by_split"] != {"train": 20, "valid": 4, "test": 4}:
            raise ValueError("La particion por hablantes no respeta 20/4/4 hablantes.")
    return checks


def guardar_preparacion(protocol: str, split_seed: int, df: pd.DataFrame | None = None) -> dict:
    base_df = cargar_metadata_experimento() if df is None else df
    split_df = crear_split(base_df, protocol, split_seed)
    out_dir = RESULTS_ROOT / protocol / f"split_seed_{split_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment = speaker_assignment(split_df)
    checks = comprobar_split(split_df, protocol)
    checks["n_audios_by_dataset_emotion_split"] = (
        split_df.groupby(["dataset", "emotion", "split"]).size().reset_index(name="n_audios").to_dict(orient="records")
    )

    split_df.to_csv(out_dir / "split_metadata.csv", index=False, encoding="utf-8")
    assignment.to_csv(out_dir / "speaker_assignment.csv", index=False, encoding="utf-8")
    guardar_json(checks, out_dir / "split_checks.json")
    return checks


def preparar_todos_los_splits() -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    df = cargar_metadata_experimento()
    checks = comprobar_metadata_y_embeddings(df)
    df.to_csv(RESULTS_ROOT / "metadata_experimento_06.csv", index=False, encoding="utf-8")
    guardar_json(checks, RESULTS_ROOT / "metadata_checks.json")

    split_checks = {}
    for protocol in PROTOCOLS:
        for split_seed in SPLIT_SEEDS:
            split_checks[f"{protocol}_{split_seed}"] = guardar_preparacion(protocol, split_seed, df)
    return {"metadata_checks": checks, "split_checks": split_checks}


def _predict_con_probabilidades(model, dataset: TensorFeatureDataset, device: torch.device, batch_size: int) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(batch_probs.argmax(axis=1).tolist())
            probs.extend(batch_probs.tolist())

    pred_df = dataset.df.copy().reset_index(drop=True)
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["emotion_real"] = [config.INV_LABEL_MAP[i] for i in y_true]
    pred_df["prediction"] = [config.INV_LABEL_MAP[i] for i in y_pred]
    for i, emotion in enumerate(config.EMOTIONS):
        pred_df[f"prob_{emotion}"] = [row[i] for row in probs]
    return pred_df


def _metricas(pred_df: pd.DataFrame, protocol: str, split_seed: int, group: str, corpus: str, speaker_id: str = "") -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=list(range(len(config.EMOTIONS))),
        average="macro",
        zero_division=0,
    )
    return {
        "protocol": protocol,
        "split_seed": split_seed,
        "model_seed": MODEL_SEED,
        "group": group,
        "corpus": corpus,
        "speaker_id": speaker_id,
        "n_audios": int(len(pred_df)),
        "n_speakers": int(pred_df["speaker_id"].nunique()) if "speaker_id" in pred_df else 0,
        "accuracy": accuracy_score(pred_df["y_true"], pred_df["y_pred"]),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def calcular_metricas_06(pred_df: pd.DataFrame, protocol: str, split_seed: int) -> pd.DataFrame:
    rows = [_metricas(pred_df, protocol, split_seed, "test_global", "GLOBAL")]
    for corpus, group in pred_df.groupby("dataset"):
        rows.append(_metricas(group, protocol, split_seed, f"test_{corpus}", corpus))
    for speaker_id, group in pred_df.groupby("speaker_id"):
        corpus = str(group["dataset"].iloc[0])
        rows.append(_metricas(group, protocol, split_seed, f"speaker_{speaker_id}", corpus, speaker_id))
    return pd.DataFrame(rows)


def _guardar_resumen_hablante(pred_df: pd.DataFrame, path: Path) -> None:
    rows = []
    for speaker_id, group in pred_df.groupby("speaker_id"):
        errores = group[group["emotion_real"] != group["prediction"]]
        if errores.empty:
            confusion = ""
        else:
            confusion = (
                errores.groupby(["emotion_real", "prediction"]).size().sort_values(ascending=False).index[0]
            )
            confusion = f"{confusion[0]} -> {confusion[1]}"
        rows.append(
            {
                "speaker_id": speaker_id,
                "corpus": group["dataset"].iloc[0],
                "n_audios": int(len(group)),
                "n_clases_presentes": int(group["emotion_real"].nunique()),
                "aviso_f1": "" if group["emotion_real"].nunique() == len(TARGET_EMOTIONS) else "No contiene todas las clases.",
                "confusion_principal": confusion,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def entrenar_repeticion(protocol: str, split_seed: int) -> dict:
    out_dir = RESULTS_ROOT / protocol / f"split_seed_{split_seed}"
    split_path = out_dir / "split_metadata.csv"
    if not split_path.exists():
        guardar_preparacion(protocol, split_seed)

    split_df = pd.read_csv(split_path)
    checks = comprobar_split(split_df, protocol)
    guardar_json(checks, out_dir / "split_checks.json")

    train_df = split_df[split_df["split"].eq("train")].reset_index(drop=True)
    valid_df = split_df[split_df["split"].eq("valid")].reset_index(drop=True)
    test_df = split_df[split_df["split"].eq("test")].reset_index(drop=True)

    fijar_semilla(MODEL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dir = FEATURES_DIR / "hubert"
    train_ds = TensorFeatureDataset(train_df, feature_dir)
    valid_ds = TensorFeatureDataset(valid_df, feature_dir)
    test_ds = TensorFeatureDataset(test_df, feature_dir)

    params = config.TrainingParams(use_scheduler=True, weight_decay=1e-4)
    model = MLPHuBERT(input_dim=768, num_classes=len(config.EMOTIONS), dropout=0.45).to(device)
    sampler = _class_weight_sampler(train_df, MODEL_SEED) if params.use_class_weights else None
    train_loader = DataLoader(train_ds, batch_size=params.batch_size, shuffle=sampler is None, sampler=sampler)
    valid_loader = DataLoader(valid_ds, batch_size=params.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=params.lr_patience
    )

    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    best_model_path = out_dir / "best_model.pt"

    for epoch in tqdm(range(1, params.epochs + 1), desc=f"{protocol} split_seed {split_seed}", unit="epoca"):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = _validate(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_accuracy": train_acc,
                "valid_accuracy": valid_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= params.early_stopping_patience:
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8")
    save_loss_curve(history_df, out_dir / "loss_curve.png", f"{protocol} split_seed {split_seed}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    pred_df = _predict_con_probabilidades(model, test_ds, device, params.batch_size)
    pred_df.to_csv(out_dir / "predicciones.csv", index=False, encoding="utf-8")
    metrics = calcular_metricas_06(pred_df, protocol, split_seed)
    metrics.to_csv(out_dir / "metricas.csv", index=False, encoding="utf-8")
    _guardar_resumen_hablante(pred_df, out_dir / "speaker_test_summary.csv")
    save_confusion_matrix(pred_df["y_true"], pred_df["y_pred"], out_dir / "confusion_matrix.png", protocol, normalize=False)
    save_confusion_matrix(
        pred_df["y_true"], pred_df["y_pred"], out_dir / "confusion_matrix_norm.png", protocol, normalize=True
    )

    stats = {
        "experiment": EXPERIMENT_NAME,
        "protocol": protocol,
        "split_seed": split_seed,
        "model_seed": MODEL_SEED,
        "device": str(device),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        **contar_parametros(model),
        "best_valid_loss": float(best_valid_loss),
        "epochs_ran": int(len(history_df)),
        "run_config": {
            "hubert_frozen_embeddings": True,
            "dropout": 0.45,
            "use_scheduler": True,
            "use_weighted_random_sampler": params.use_class_weights,
        },
    }
    guardar_json(stats, out_dir / "stats.json")
    return stats


def _leer_metricas_existentes() -> pd.DataFrame:
    filas = []
    for protocol in PROTOCOLS:
        for split_seed in SPLIT_SEEDS:
            path = RESULTS_ROOT / protocol / f"split_seed_{split_seed}" / "metricas.csv"
            if path.exists():
                filas.append(pd.read_csv(path))
    if not filas:
        raise FileNotFoundError("No hay metricas del experimento 06. Entrena primero las repeticiones.")
    return pd.concat(filas, ignore_index=True)


def generar_resumenes_06() -> dict[str, Path]:
    metrics = _leer_metricas_existentes()
    if metrics["protocol"].nunique() != 2:
        raise ValueError("Falta algun protocolo del experimento 06.")
    if set(metrics["split_seed"].unique()) != set(SPLIT_SEEDS):
        raise ValueError("Cada protocolo debe tener tres repeticiones completas.")

    main = metrics[metrics["corpus"].isin(["GLOBAL", "RAVDESS", "SAVEE"]) & metrics["speaker_id"].fillna("").eq("")]
    summary = (
        main.groupby(["protocol", "corpus"], as_index=False)
        .agg(
            n_repeticiones=("split_seed", "nunique"),
            n_audios_mean=("n_audios", "mean"),
            n_speakers_mean=("n_speakers", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda x: x.std(ddof=1)),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            precision_macro_mean=("precision_macro", "mean"),
            precision_macro_std=("precision_macro", lambda x: x.std(ddof=1)),
            recall_macro_mean=("recall_macro", "mean"),
            recall_macro_std=("recall_macro", lambda x: x.std(ddof=1)),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", lambda x: x.std(ddof=1)),
            f1_macro_min=("f1_macro", "min"),
            f1_macro_max=("f1_macro", "max"),
        )
        .reset_index(drop=True)
    )

    pivot = summary.pivot(index="corpus", columns="protocol", values="f1_macro_mean")
    comparison = pd.DataFrame(
        {
            "corpus": pivot.index,
            "f1_random_audio": pivot["random_audio"],
            "f1_speaker_disjoint": pivot["speaker_disjoint"],
        }
    ).reset_index(drop=True)
    comparison["delta_f1"] = comparison["f1_speaker_disjoint"] - comparison["f1_random_audio"]
    comparison["loss_pp"] = -comparison["delta_f1"] * 100
    comparison["relative_loss_pct"] = -comparison["delta_f1"] / comparison["f1_random_audio"] * 100

    metrics_path = RESULTS_ROOT / "metricas_repeticiones.csv"
    summary_path = RESULTS_ROOT / "resumen_media_std.csv"
    comparison_path = RESULTS_ROOT / "comparacion_protocolos.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    return {"metricas_repeticiones": metrics_path, "resumen_media_std": summary_path, "comparacion_protocolos": comparison_path}
