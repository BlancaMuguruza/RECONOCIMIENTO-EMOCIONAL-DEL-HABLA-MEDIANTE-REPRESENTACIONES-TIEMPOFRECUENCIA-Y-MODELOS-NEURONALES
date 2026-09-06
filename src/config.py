from __future__ import annotations

from dataclasses import dataclass


# Configuracion comun de esta primera version del experimento.
DATASETS = ["RAVDESS", "TESS", "SAVEE"]
EMOTIONS = ["angry", "fearful", "happy", "sad"]
LABEL_MAP = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
INV_LABEL_MAP = {idx: emotion for emotion, idx in LABEL_MAP.items()}

SPLIT_SEED = 42
SEEDS = [42, 123, 2024]

TEST_SIZE = 0.15
VALID_SIZE = 0.15

SR = 16000
DURATION_SECONDS = 3.0
N_SAMPLES = int(SR * DURATION_SECONDS)
HUBERT_FRAME_SECONDS = 0.02
HUBERT_FRAMES_PER_AUDIO = int(DURATION_SECONDS / HUBERT_FRAME_SECONDS)

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

N_WAVELET_SCALES = 64
WAVELET_INPUT_SAMPLES = 8192
WAVELET_TIME_BINS = 1024
WAVELET_FEATURE_DIR = "wavelet_64x1024"
WAVELET_NAME = "morl"

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 12
LR_PATIENCE = 3


@dataclass(frozen=True)
class TrainingParams:
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    lr_patience: int = LR_PATIENCE
    use_scheduler: bool = False
    # Nombre conservado por compatibilidad: activa WeightedRandomSampler, no pesos en CrossEntropyLoss.
    use_class_weights: bool = True


DEFAULT_TRAINING = TrainingParams()

MEL_CONTROLLED_DROPOUT = 0.40
MEL_CONTROLLED_VERSION = "mel_controlada_v1"
MEL_CONTROLLED_EXPERIMENTS = {
    "01_cnn_baseline": {
        "dropout": MEL_CONTROLLED_DROPOUT,
        "augment_train": False,
        "use_scheduler": False,
    },
    "02_cnn_data_augmentation": {
        "dropout": MEL_CONTROLLED_DROPOUT,
        "augment_train": True,
        "use_scheduler": False,
    },
    "03_cnn_scheduler": {
        "dropout": MEL_CONTROLLED_DROPOUT,
        "augment_train": False,
        "use_scheduler": True,
    },
}

