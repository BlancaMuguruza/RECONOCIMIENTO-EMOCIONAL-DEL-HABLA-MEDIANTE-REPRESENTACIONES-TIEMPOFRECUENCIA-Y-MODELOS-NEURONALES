from __future__ import annotations

import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.datos import cargar_o_crear_metadata, split_dataframes
from src.entrenamiento import TensorFeatureDataset, entrenar_una_semilla, feature_tensor_path, preparar_datasets_audio
from src.modelos import MLPHuBERT, crear_cnn_mel, crear_cnn_wavelet
from src.paths import FEATURES_DIR, experiment_dir


def train_audio_experiment(
    experiment: str,
    feature: str,
    model_factory,
    dropout: float,
    params: config.TrainingParams,
    augment_train: bool = False,
    run_config: dict | None = None,
) -> None:
    df = cargar_o_crear_metadata()
    output_dir = experiment_dir(experiment)
    train_ds, valid_ds, test_ds = preparar_datasets_audio(df, feature=feature, augment_train=augment_train)

    for seed in tqdm(config.SEEDS, desc=f"{experiment} semillas", unit="seed"):
        print(f"[{experiment}] entrenando seed {seed}")
        model = model_factory(num_classes=len(config.EMOTIONS), dropout=dropout)
        entrenar_una_semilla(
            experiment,
            model,
            df,
            train_ds,
            valid_ds,
            test_ds,
            output_dir,
            seed,
            params,
            run_config=run_config,
        )

    print(f"[{experiment}] terminado. Resultados en: {output_dir}")


def _comprobar_features(df, feature_dir: Path, comando: str) -> None:
    missing = [feature_tensor_path(row, feature_dir) for _, row in df.iterrows() if not feature_tensor_path(row, feature_dir).exists()]
    if missing:
        ejemplo = missing[0]
        raise FileNotFoundError(
            f"Faltan {len(missing)} features en {feature_dir}. "
            f"Ejecuta primero: {comando}. Ejemplo que falta: {ejemplo}"
        )


def train_tensor_experiment(
    experiment: str,
    feature_name: str,
    model_factory,
    dropout: float,
    params: config.TrainingParams,
    comando_generacion: str,
    run_config: dict | None = None,
) -> None:
    df = cargar_o_crear_metadata()
    output_dir = experiment_dir(experiment)
    feature_dir = FEATURES_DIR / feature_name
    _comprobar_features(df, feature_dir, comando_generacion)

    train_df, valid_df, test_df = split_dataframes(df)
    train_ds = TensorFeatureDataset(train_df, feature_dir)
    valid_ds = TensorFeatureDataset(valid_df, feature_dir)
    test_ds = TensorFeatureDataset(test_df, feature_dir)

    for seed in tqdm(config.SEEDS, desc=f"{experiment} semillas", unit="seed"):
        print(f"[{experiment}] entrenando seed {seed}")
        model = model_factory(num_classes=len(config.EMOTIONS), dropout=dropout)
        entrenar_una_semilla(
            experiment,
            model,
            df,
            train_ds,
            valid_ds,
            test_ds,
            output_dir,
            seed,
            params,
            run_config=run_config,
        )

    print(f"[{experiment}] terminado. Resultados en: {output_dir}")

def train_hubert_experiment(experiment: str, dropout: float, params: config.TrainingParams) -> None:
    df = cargar_o_crear_metadata()
    output_dir = experiment_dir(experiment)
    feature_dir = FEATURES_DIR / "hubert"
    train_df, valid_df, test_df = split_dataframes(df)

    train_ds = TensorFeatureDataset(train_df, feature_dir)
    valid_ds = TensorFeatureDataset(valid_df, feature_dir)
    test_ds = TensorFeatureDataset(test_df, feature_dir)

    for seed in tqdm(config.SEEDS, desc=f"{experiment} semillas", unit="seed"):
        print(f"[{experiment}] entrenando seed {seed}")
        model = MLPHuBERT(input_dim=768, num_classes=len(config.EMOTIONS), dropout=dropout)
        entrenar_una_semilla(experiment, model, df, train_ds, valid_ds, test_ds, output_dir, seed, params)

    print(f"[{experiment}] terminado. Resultados en: {output_dir}")
