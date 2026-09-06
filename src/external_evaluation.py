from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from . import config
from .audio import cargar_audio, mel_spectrogram, normalize_image_tensor, wavelet_scalogram
from .entrenamiento import feature_tensor_path
from .external_datasets import EXTERNAL_RESULTS_DIR, metadata_incluida
from .graficas import save_confusion_matrix
from .modelos import MLPHuBERT, crear_cnn_mel, crear_cnn_wavelet
from .paths import RESULTS_DIR
from .utilidades import contar_parametros


SEEDS = [42, 123, 2024]
EXTERNAL_CORPUS = ["CREMA-D", "EMO-DB"]


@dataclass(frozen=True)
class ExternalModelSpec:
    model_id: str
    label: str
    representation: str
    dropout: float


MODEL_SPECS = {
    "01_cnn_baseline": ExternalModelSpec("01_cnn_baseline", "01 CNN Mel baseline", "mel", config.MEL_CONTROLLED_DROPOUT),
    "02_cnn_data_augmentation": ExternalModelSpec("02_cnn_data_augmentation", "02 CNN Mel con aumentacion", "mel", config.MEL_CONTROLLED_DROPOUT),
    "03_cnn_scheduler": ExternalModelSpec("03_cnn_scheduler", "03 CNN Mel con scheduler", "mel", config.MEL_CONTROLLED_DROPOUT),
    "04_cnn_wavelet": ExternalModelSpec("04_cnn_wavelet", "04 CNN Wavelet", "wavelet", 0.50),
    "05_hubert": ExternalModelSpec("05_hubert", "05 HuBERT", "hubert", 0.45),
}


class ExternalAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, representation: str):
        self.df = df.reset_index(drop=True)
        self.representation = representation

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = cargar_audio(row["path"])
        if self.representation == "mel":
            x = normalize_image_tensor(mel_spectrogram(y))
        elif self.representation == "wavelet":
            x = normalize_image_tensor(wavelet_scalogram(y))
        else:
            raise ValueError(f"Representacion no soportada: {self.representation}")
        label = config.LABEL_MAP[row["emotion"]]
        return x, torch.tensor(label, dtype=torch.long)


class ExternalTensorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_dir: Path):
        self.df = df.reset_index(drop=True)
        self.feature_dir = feature_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tensor_path = feature_tensor_path(row, self.feature_dir)
        if not tensor_path.exists():
            raise FileNotFoundError(f"No se encuentra embedding externo: {tensor_path}")
        x = torch.load(tensor_path, map_location="cpu").float()
        label = config.LABEL_MAP[row["emotion"]]
        return x, torch.tensor(label, dtype=torch.long)


def checkpoint_path(model_id: str, seed: int) -> Path:
    return RESULTS_DIR / model_id / f"seed_{seed}" / "best_model.pt"


def comprobar_checkpoints() -> list[dict]:
    rows = []
    for model_id in MODEL_SPECS:
        for seed in SEEDS:
            ckpt = checkpoint_path(model_id, seed)
            stats_path = RESULTS_DIR / model_id / f"seed_{seed}" / "stats.json"
            row = {
                "model_id": model_id,
                "seed": seed,
                "checkpoint": ckpt.exists(),
                "stats": stats_path.exists(),
                "mel_controlado": True,
            }
            if model_id in config.MEL_CONTROLLED_EXPERIMENTS and stats_path.exists():
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                row["mel_controlado"] = stats.get("run_config", {}).get("version") == config.MEL_CONTROLLED_VERSION
            rows.append(row)
    return rows


def assert_checkpoints_validos() -> None:
    rows = comprobar_checkpoints()
    missing = [row for row in rows if not row["checkpoint"] or not row["stats"]]
    if missing:
        raise FileNotFoundError(f"Faltan checkpoints o stats: {missing}")
    mel_bad = [row for row in rows if not row["mel_controlado"]]
    if mel_bad:
        raise ValueError(f"Hay checkpoints Mel que no pertenecen a {config.MEL_CONTROLLED_VERSION}: {mel_bad}")


def crear_modelo(spec: ExternalModelSpec) -> nn.Module:
    if spec.representation == "mel":
        return crear_cnn_mel(num_classes=len(config.EMOTIONS), dropout=spec.dropout)
    if spec.representation == "wavelet":
        return crear_cnn_wavelet(num_classes=len(config.EMOTIONS), dropout=spec.dropout)
    if spec.representation == "hubert":
        return MLPHuBERT(input_dim=768, num_classes=len(config.EMOTIONS), dropout=spec.dropout)
    raise ValueError(f"Modelo externo no soportado: {spec.model_id}")


def cargar_checkpoint(model_id: str, seed: int, device: torch.device) -> nn.Module:
    spec = MODEL_SPECS[model_id]
    model = crear_modelo(spec).to(device)
    state = torch.load(checkpoint_path(model_id, seed), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def external_hubert_cache_dir(dataset: str) -> Path:
    safe_dataset = dataset.lower().replace("-", "_")
    path = EXTERNAL_RESULTS_DIR / "hubert_cache" / safe_dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def asegurar_hubert_embeddings(df: pd.DataFrame, dataset: str) -> Path:
    feature_dir = external_hubert_cache_dir(dataset)
    missing = [row for _, row in df.iterrows() if not feature_tensor_path(row, feature_dir).exists()]
    if not missing:
        return feature_dir

    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    hubert.eval()
    with torch.no_grad():
        for row in missing:
            output_path = feature_tensor_path(row, feature_dir)
            y = cargar_audio(row["path"])
            inputs = extractor(y, sampling_rate=config.SR, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = hubert(**inputs).last_hidden_state.squeeze(0).cpu()
            torch.save(hidden.half(), output_path)
    return feature_dir


def dataset_para_modelo(df: pd.DataFrame, spec: ExternalModelSpec, dataset: str) -> Dataset:
    if spec.representation in {"mel", "wavelet"}:
        return ExternalAudioDataset(df, spec.representation)
    feature_dir = asegurar_hubert_embeddings(df, dataset)
    return ExternalTensorDataset(df, feature_dir)


def predecir_externo(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = config.BATCH_SIZE) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred, probs = [], [], []
    model.eval()
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
    for idx, emotion in enumerate(config.EMOTIONS):
        pred_df[f"prob_{emotion}"] = [row[idx] for row in probs]
    pred_df["confidence"] = pred_df[[f"prob_{emotion}" for emotion in config.EMOTIONS]].max(axis=1)
    pred_df["correct"] = pred_df["y_true"].eq(pred_df["y_pred"])
    prob_sum = pred_df[[f"prob_{emotion}" for emotion in config.EMOTIONS]].sum(axis=1)
    if not prob_sum.between(0.999, 1.001).all():
        raise ValueError("Las probabilidades no suman aproximadamente 1.")
    return pred_df


def metricas_globales(pred_df: pd.DataFrame, model_id: str, seed: int, dataset: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=list(range(len(config.EMOTIONS))),
        average="macro",
        zero_division=0,
    )
    return {
        "model_id": model_id,
        "model": MODEL_SPECS[model_id].label,
        "seed": seed,
        "dataset": dataset,
        "n_audios": int(len(pred_df)),
        "accuracy": accuracy_score(pred_df["y_true"], pred_df["y_pred"]),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def metricas_por_clase(pred_df: pd.DataFrame, model_id: str, seed: int, dataset: str) -> pd.DataFrame:
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
                "model_id": model_id,
                "model": MODEL_SPECS[model_id].label,
                "seed": seed,
                "dataset": dataset,
                "emotion": emotion,
                "precision": precision[idx],
                "recall": recall[idx],
                "f1": f1[idx],
                "support": int(support[idx]),
            }
        )
    return pd.DataFrame(rows)


def output_dir(model_id: str, seed: int, dataset: str) -> Path:
    safe_dataset = dataset.lower().replace("-", "_")
    path = EXTERNAL_RESULTS_DIR / model_id / f"seed_{seed}" / safe_dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluar_combinacion(model_id: str, seed: int, dataset: str) -> dict:
    assert model_id in MODEL_SPECS
    assert seed in SEEDS
    assert dataset in EXTERNAL_CORPUS
    assert_checkpoints_validos()

    df = metadata_incluida()
    df = df[df["dataset"].eq(dataset)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No hay audios incluidos para {dataset}.")

    spec = MODEL_SPECS[model_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cargar_checkpoint(model_id, seed, device)
    before_params = [param.detach().clone() for param in model.parameters()]
    dataset_obj = dataset_para_modelo(df, spec, dataset)
    pred_df = predecir_externo(model, dataset_obj, device)
    after_params = [param.detach() for param in model.parameters()]
    if any(not torch.equal(before, after) for before, after in zip(before_params, after_params)):
        raise RuntimeError("Se ha modificado algun parametro durante la evaluacion.")

    pred_df.insert(0, "model_id", model_id)
    pred_df.insert(1, "model", spec.label)
    pred_df.insert(2, "seed", seed)
    out_dir = output_dir(model_id, seed, dataset)
    pred_df.to_csv(out_dir / "predicciones.csv", index=False, encoding="utf-8")

    metrics = pd.DataFrame([metricas_globales(pred_df, model_id, seed, dataset)])
    class_metrics = metricas_por_clase(pred_df, model_id, seed, dataset)
    metrics.to_csv(out_dir / "metricas.csv", index=False, encoding="utf-8")
    class_metrics.to_csv(out_dir / "metricas_clase.csv", index=False, encoding="utf-8")

    report = classification_report(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=list(range(len(config.EMOTIONS))),
        target_names=config.EMOTIONS,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(pred_df["y_true"], pred_df["y_pred"], out_dir / "confusion_matrix.png", f"{spec.label} {dataset}", normalize=False)
    save_confusion_matrix(pred_df["y_true"], pred_df["y_pred"], out_dir / "confusion_matrix_norm.png", f"{spec.label} {dataset} normalizada", normalize=True)

    run_config = {
        "model_id": model_id,
        "seed": seed,
        "dataset": dataset,
        "representation": spec.representation,
        "checkpoint": str(checkpoint_path(model_id, seed)),
        "no_training": True,
        "eval_mode": True,
        "torch_no_grad": True,
        "augmentation": False,
        "class_order": config.EMOTIONS,
        "parametros": contar_parametros(model),
    }
    (out_dir / "configuracion.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"metricas": metrics.to_dict(orient="records")[0], "output_dir": str(out_dir)}


def generar_resumen_externo() -> dict[str, Path]:
    metric_rows = []
    class_rows = []
    for model_id in MODEL_SPECS:
        for seed in SEEDS:
            for dataset in EXTERNAL_CORPUS:
                out = output_dir(model_id, seed, dataset)
                metrics_path = out / "metricas.csv"
                class_path = out / "metricas_clase.csv"
                if metrics_path.exists():
                    metric_rows.append(pd.read_csv(metrics_path))
                if class_path.exists():
                    class_rows.append(pd.read_csv(class_path))

    if not metric_rows:
        raise FileNotFoundError("No hay resultados externos guardados.")

    metrics = pd.concat(metric_rows, ignore_index=True)
    class_metrics = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    metrics["model"] = metrics["model_id"].map(lambda model_id: MODEL_SPECS[model_id].label)
    if not class_metrics.empty:
        class_metrics["model"] = class_metrics["model_id"].map(lambda model_id: MODEL_SPECS[model_id].label)
    expected = len(MODEL_SPECS) * len(SEEDS) * len(EXTERNAL_CORPUS)
    complete = len(metrics) == expected

    summary = (
        metrics.groupby(["model_id", "model", "dataset"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_audios=("n_audios", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda x: x.std(ddof=1)),
            precision_macro_mean=("precision_macro", "mean"),
            precision_macro_std=("precision_macro", lambda x: x.std(ddof=1)),
            recall_macro_mean=("recall_macro", "mean"),
            recall_macro_std=("recall_macro", lambda x: x.std(ddof=1)),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", lambda x: x.std(ddof=1)),
        )
    )
    class_summary = pd.DataFrame()
    if not class_metrics.empty:
        class_summary = (
            class_metrics.groupby(["model_id", "model", "dataset", "emotion"], as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                support_mean=("support", "mean"),
                precision_mean=("precision", "mean"),
                precision_std=("precision", lambda x: x.std(ddof=1)),
                recall_mean=("recall", "mean"),
                recall_std=("recall", lambda x: x.std(ddof=1)),
                f1_mean=("f1", "mean"),
                f1_std=("f1", lambda x: x.std(ddof=1)),
            )
        )

    EXTERNAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = EXTERNAL_RESULTS_DIR / "metricas_externas.csv"
    summary_path = EXTERNAL_RESULTS_DIR / "resumen_modelo_dataset.csv"
    class_path = EXTERNAL_RESULTS_DIR / "resumen_por_emocion.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    class_summary.to_csv(class_path, index=False, encoding="utf-8")

    status = {
        "complete": complete,
        "n_expected": expected,
        "n_found": int(len(metrics)),
        "ddof": 1,
        "external_only": True,
    }
    (EXTERNAL_RESULTS_DIR / "resumen_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"metricas": metrics_path, "resumen": summary_path, "emociones": class_path}
