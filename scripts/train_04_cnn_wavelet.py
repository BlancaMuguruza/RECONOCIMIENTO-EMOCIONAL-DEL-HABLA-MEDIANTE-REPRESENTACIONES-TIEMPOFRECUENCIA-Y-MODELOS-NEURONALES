from __future__ import annotations

from _common import config, crear_cnn_wavelet, train_tensor_experiment


if __name__ == "__main__":
    train_tensor_experiment(
        experiment="04_cnn_wavelet",
        feature_name=config.WAVELET_FEATURE_DIR,
        model_factory=crear_cnn_wavelet,
        dropout=0.50,
        params=config.TrainingParams(batch_size=8, use_scheduler=True, weight_decay=2e-4),
        comando_generacion="python scripts/generate_wavelet_features.py",
    )