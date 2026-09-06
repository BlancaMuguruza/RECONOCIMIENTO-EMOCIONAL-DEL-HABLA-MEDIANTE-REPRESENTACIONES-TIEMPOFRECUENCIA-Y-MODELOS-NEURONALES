from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

TRAIN_SCRIPTS = {
    "01_cnn_baseline": PROJECT_ROOT / "scripts" / "train_01_cnn_baseline.py",
    "02_cnn_data_augmentation": PROJECT_ROOT / "scripts" / "train_02_cnn_data_augmentation.py",
    "03_cnn_scheduler": PROJECT_ROOT / "scripts" / "train_03_cnn_scheduler.py",
}
BACKUP_DIR = PROJECT_ROOT / "resultados_originales_mel_configuracion_no_controlada"


def main() -> None:
    assert config.MEL_CONTROLLED_DROPOUT == 0.40
    assert config.SPLIT_SEED == 42
    assert config.SEEDS == [42, 123, 2024]
    assert config.WEIGHT_DECAY == 1e-4
    assert config.BATCH_SIZE == 32
    assert config.LEARNING_RATE == 1e-3
    assert config.EPOCHS == 60
    assert config.EARLY_STOPPING_PATIENCE == 12
    assert config.LR_PATIENCE == 3

    expected = {
        "01_cnn_baseline": {"dropout": 0.40, "augment_train": False, "use_scheduler": False},
        "02_cnn_data_augmentation": {"dropout": 0.40, "augment_train": True, "use_scheduler": False},
        "03_cnn_scheduler": {"dropout": 0.40, "augment_train": False, "use_scheduler": True},
    }
    assert config.MEL_CONTROLLED_EXPERIMENTS == expected

    for experiment, script_path in TRAIN_SCRIPTS.items():
        text = script_path.read_text(encoding="utf-8")
        assert f'config.MEL_CONTROLLED_EXPERIMENTS["{experiment}"]' in text
        assert "crear_cnn_mel" in text
        assert "run_config=" in text

    common = (PROJECT_ROOT / "scripts" / "_common.py").read_text(encoding="utf-8")
    assert "run_config=run_config" in common

    entrenamiento = (PROJECT_ROOT / "src" / "entrenamiento.py").read_text(encoding="utf-8")
    assert "AudioFeatureDataset(valid_df, feature=feature, augment=False)" in entrenamiento
    assert "AudioFeatureDataset(test_df, feature=feature, augment=False)" in entrenamiento
    assert "nn.CrossEntropyLoss()" in entrenamiento
    assert "WeightedRandomSampler" in entrenamiento

    for experiment in TRAIN_SCRIPTS:
        assert (BACKUP_DIR / experiment).exists(), f"Falta copia antigua de {experiment}"

    print("Configuracion Mel controlada verificada.")


if __name__ == "__main__":
    main()
