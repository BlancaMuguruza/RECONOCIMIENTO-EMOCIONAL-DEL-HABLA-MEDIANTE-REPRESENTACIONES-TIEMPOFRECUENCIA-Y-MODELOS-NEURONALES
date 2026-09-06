from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from . import config
from .audio import augmentation_audio, cargar_audio, mel_spectrogram, normalize_image_tensor, wavelet_scalogram
from .datos import split_dataframes
from .evaluacion import predict, save_evaluation
from .graficas import save_loss_curve
from .utilidades import contar_parametros, fijar_semilla, guardar_json


def feature_tensor_path(row: pd.Series, feature_dir: Path) -> Path:
    dataset = str(row.get("dataset", "audio")).lower()
    file_id = str(row.get("file_id", Path(row["path"]).stem))
    digest = hashlib.sha1(str(Path(row["path"]).resolve()).encode("utf-8")).hexdigest()[:10]
    return feature_dir / f"{dataset}_{file_id}_{digest}.pt"

# Dataset ligero: calcula mel o wavelet al pedir cada muestra.
class AudioFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature: str, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.feature = feature
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = cargar_audio(row["path"])
        if self.augment:
            y = augmentation_audio(y)

        if self.feature == "mel":
            x = mel_spectrogram(y)
        elif self.feature == "wavelet":
            x = wavelet_scalogram(y)
        else:
            raise ValueError(f"Feature no soportada: {self.feature}")

        x = normalize_image_tensor(x)
        label = config.LABEL_MAP[row["emotion"]]
        return x, torch.tensor(label, dtype=torch.long)


class TensorFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_dir: Path):
        self.df = df.reset_index(drop=True)
        self.feature_dir = feature_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tensor_path = feature_tensor_path(row, self.feature_dir)
        if not tensor_path.exists():
            old_path = self.feature_dir / f"{Path(row['path']).stem}.pt"
            tensor_path = old_path if old_path.exists() else tensor_path
        if not tensor_path.exists():
            raise FileNotFoundError(f"No se encuentra la feature guardada: {tensor_path}")
        x = torch.load(tensor_path, map_location="cpu").float()
        label = config.LABEL_MAP[row["emotion"]]
        return x, torch.tensor(label, dtype=torch.long)


def _class_weight_sampler(train_df: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    counts = train_df["emotion"].value_counts()
    weights = train_df["emotion"].map(lambda emotion: 1.0 / counts[emotion]).to_numpy().copy()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _run_epoch(model, loader, criterion, optimizer, device, progress_desc: str | None = None):
    model.train()
    loss_total = 0.0
    correct = 0
    total = 0

    iterator = tqdm(loader, desc=progress_desc, leave=False, unit="batch") if progress_desc else loader

    for inputs, labels in iterator:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return loss_total / max(len(loader), 1), correct / max(total, 1)


def _validate(model, loader, criterion, device, progress_desc: str | None = None):
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0

    iterator = tqdm(loader, desc=progress_desc, leave=False, unit="batch") if progress_desc else loader

    with torch.no_grad():
        for inputs, labels in iterator:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return loss_total / max(len(loader), 1), correct / max(total, 1)



def limpiar_resultados_semilla(seed_dir: Path, seed: int) -> None:
    # Evita mezclar resultados antiguos si se reentrena con otra arquitectura.
    nombres = [
        "best_model.pt",
        "historial_entrenamiento.csv",
        "loss_curve.png",
        "stats.json",
        f"predicciones_seed_{seed}.csv",
        f"metricas_seed_{seed}.csv",
        f"metricas_clase_seed_{seed}.csv",
        f"classification_report_seed_{seed}.txt",
        f"confusion_matrix_seed_{seed}.png",
        f"confusion_matrix_norm_seed_{seed}.png",
    ]
    for nombre in nombres:
        path = seed_dir / nombre
        if path.exists():
            path.unlink()
# Entrena una semilla y deja guardado todo lo que se ve en Marimo.
def entrenar_una_semilla(
    experiment: str,
    model: nn.Module,
    df: pd.DataFrame,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    output_dir: Path,
    seed: int,
    params: config.TrainingParams = config.DEFAULT_TRAINING,
    run_config: dict | None = None,
) -> dict:
    fijar_semilla(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    limpiar_resultados_semilla(seed_dir, seed)

    train_df, valid_df, test_df = split_dataframes(df)
    sampler = _class_weight_sampler(train_df, seed) if params.use_class_weights else None
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    scheduler = None
    if params.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=params.lr_patience,
        )

    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    best_model_path = seed_dir / "best_model.pt"

    epoch_iter = tqdm(range(1, params.epochs + 1), desc=f"{experiment} seed {seed}", unit="epoca")
    for epoch in epoch_iter:
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, progress_desc=f"train epoca {epoch}"
        )
        valid_loss, valid_acc = _validate(
            model, valid_loader, criterion, device, progress_desc=f"valid epoca {epoch}"
        )
        if scheduler is not None:
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

        epoch_iter.set_postfix(
            train_loss=f"{train_loss:.4f}",
            valid_loss=f"{valid_loss:.4f}",
            valid_acc=f"{valid_acc:.3f}",
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
    history_df.to_csv(seed_dir / "historial_entrenamiento.csv", index=False, encoding="utf-8")
    save_loss_curve(history_df, seed_dir / "loss_curve.png", f"{experiment} seed {seed}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    pred_df = predict(model, test_dataset, device, params.batch_size)
    metrics = save_evaluation(pred_df, seed_dir, experiment, seed)

    stats = {
        "experiment": experiment,
        "seed": seed,
        "device": str(device),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        **contar_parametros(model),
        "datos_por_parametro": float(len(train_df) / max(contar_parametros(model)["parametros_entrenables"], 1)),
        "parametros_por_dato": float(contar_parametros(model)["parametros_entrenables"] / max(len(train_df), 1)),
        "best_valid_loss": float(best_valid_loss),
        "epochs_ran": int(len(history_df)),
        "run_config": run_config or {},
    }
    guardar_json(stats, seed_dir / "stats.json")
    return {"stats": stats, "history": history_df, "metrics": metrics, "predictions": pred_df}



def evaluar_modelo_guardado(
    experiment: str,
    model: nn.Module,
    test_dataset: Dataset,
    output_dir: Path,
    seed: int,
    params: config.TrainingParams = config.DEFAULT_TRAINING,
) -> dict:
    seed_dir = output_dir / f"seed_{seed}"
    best_model_path = seed_dir / "best_model.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo guardado: {best_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    history_path = seed_dir / "historial_entrenamiento.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        save_loss_curve(history_df, seed_dir / "loss_curve.png", f"{experiment} seed {seed}")

    # Recalcula test con el mejor modelo sin repetir el entrenamiento.
    pred_df = predict(model, test_dataset, device, params.batch_size)
    metrics = save_evaluation(pred_df, seed_dir, experiment, seed)
    return {"metrics": metrics, "predictions": pred_df, "model_path": best_model_path}


def preparar_datasets_audio(df: pd.DataFrame, feature: str, augment_train: bool = False):
    train_df, valid_df, test_df = split_dataframes(df)
    return (
        AudioFeatureDataset(train_df, feature=feature, augment=augment_train),
        AudioFeatureDataset(valid_df, feature=feature, augment=False),
        AudioFeatureDataset(test_df, feature=feature, augment=False),
    )
