from __future__ import annotations

from _common import config, train_hubert_experiment


if __name__ == "__main__":
    train_hubert_experiment(
        experiment="05_hubert",
        dropout=0.45,
        params=config.TrainingParams(use_scheduler=True, weight_decay=1e-4),
    )
