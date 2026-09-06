from __future__ import annotations

from _common import config, crear_cnn_mel, train_audio_experiment


if __name__ == "__main__":
    mel_cfg = config.MEL_CONTROLLED_EXPERIMENTS["01_cnn_baseline"]
    train_audio_experiment(
        experiment="01_cnn_baseline",
        feature="mel",
        model_factory=crear_cnn_mel,
        dropout=mel_cfg["dropout"],
        params=config.TrainingParams(use_scheduler=mel_cfg["use_scheduler"], weight_decay=1e-4),
        augment_train=mel_cfg["augment_train"],
        run_config={"version": config.MEL_CONTROLLED_VERSION, **mel_cfg},
    )
