from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from . import config


PALETA_PASTEL = {
    "RAVDESS": "#9ED2F0",
    "SAVEE": "#A8E6BD",
    "TESS": "#F7A8C4",
    "CREMA-D": "#C8B6FF",
    "EMO-DB": "#FFD6A5",
    "angry": "#F7A8C4",
    "fearful": "#C8B6FF",
    "happy": "#FFD6A5",
    "sad": "#9ED2F0",
    "train": "#A8E6BD",
    "valid": "#9ED2F0",
    "test": "#F7A8C4",
}

COLORES_BARRAS = ["#9ED2F0", "#A8E6BD", "#F7A8C4", "#C8B6FF", "#FFD6A5"]
CMAP_MAGNITUD = LinearSegmentedColormap.from_list("magnitud_pastel", ["#F7FFF9", "#A8E6BD", "#2E8B73"])
CMAP_PORCENTAJE = LinearSegmentedColormap.from_list("porcentaje_pastel", ["#F8FBFF", "#9ED2F0", "#4A6FA5"])


# Mismo estilo visual que experimento_final para mantener continuidad en la memoria.
def set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "axes.edgecolor": "#D7DEE8",
            "grid.color": "#E9EEF6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def save_loss_curve(history: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train", color=PALETA_PASTEL["train"], linewidth=2)
    axes[0].plot(history["epoch"], history["valid_loss"], label="valid", color=PALETA_PASTEL["valid"], linewidth=2)
    axes[0].set_title(f"{title} - loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train", color=PALETA_PASTEL["train"], linewidth=2)
    axes[1].plot(history["epoch"], history["valid_accuracy"], label="valid", color=PALETA_PASTEL["valid"], linewidth=2)
    axes[1].set_title(f"{title} - accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, path: Path, title: str, normalize: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_style()
    labels = list(range(len(config.EMOTIONS)))
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )
    if normalize:
        matrix = matrix * 100
    fmt = ".1f" if normalize else "d"

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=CMAP_PORCENTAJE if normalize else CMAP_MAGNITUD,
        vmin=0,
        vmax=100 if normalize else None,
        xticklabels=config.EMOTIONS,
        yticklabels=config.EMOTIONS,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Porcentaje por clase real" if normalize else "Numero de muestras"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_countplot(df: pd.DataFrame, x: str, path: Path, title: str, hue: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if hue is None:
        order = sorted(df[x].dropna().unique().tolist())
        palette = [PALETA_PASTEL.get(value, COLORES_BARRAS[idx % len(COLORES_BARRAS)]) for idx, value in enumerate(order)]
        sns.countplot(data=df, x=x, hue=x, order=order, palette=palette, legend=False, ax=ax)
    else:
        sns.countplot(data=df, x=x, hue=hue, palette=PALETA_PASTEL, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
