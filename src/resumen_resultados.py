from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config
from .paths import RESULTS_DIR


MODEL_LABELS = {
    "01_cnn_baseline": "01 CNN Mel baseline",
    "02_cnn_data_augmentation": "02 CNN Mel con aumentacion",
    "03_cnn_scheduler": "03 CNN Mel con scheduler",
    "04_cnn_wavelet": "04 CNN Wavelet",
    "05_hubert": "05 HuBERT",
}
MEL_MODELS = ["01_cnn_baseline", "02_cnn_data_augmentation", "03_cnn_scheduler"]
CORPUS_ORDER = ["GLOBAL", "RAVDESS", "SAVEE", "TESS"]
PALETTE = {"RAVDESS": "#9ED2F0", "SAVEE": "#A8E6BD", "TESS": "#F7A8C4"}


def _metric_files(results_dir: Path = RESULTS_DIR) -> list[Path]:
    return sorted(results_dir.glob("*/seed_*/metricas_seed_*.csv"))


def cargar_metricas_semillas(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    filas = []
    for path in _metric_files(results_dir):
        model_id = path.parents[1].name
        if model_id not in MODEL_LABELS:
            continue
        metrics = pd.read_csv(path)
        metrics["model_id"] = model_id
        metrics["model"] = MODEL_LABELS[model_id]
        filas.append(metrics)

    if not filas:
        raise FileNotFoundError(f"No se han encontrado metricas en {results_dir}")

    metrics = pd.concat(filas, ignore_index=True)
    metrics = metrics[metrics["group"].str.startswith("test_")].copy()
    metrics["corpus"] = metrics["group"].str.replace("test_", "", regex=False)
    metrics.loc[metrics["group"].eq("test_global"), "corpus"] = "GLOBAL"
    return metrics


def mel_controlado_disponible(results_dir: Path = RESULTS_DIR) -> tuple[bool, list[str]]:
    problemas = []
    for model_id in MEL_MODELS:
        for seed in config.SEEDS:
            stats_path = results_dir / model_id / f"seed_{seed}" / "stats.json"
            if not stats_path.exists():
                problemas.append(f"Falta {stats_path}")
                continue
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            run_config = stats.get("run_config", {})
            if run_config.get("version") != config.MEL_CONTROLLED_VERSION:
                problemas.append(f"{stats_path} no pertenece a {config.MEL_CONTROLLED_VERSION}")
    return not problemas, problemas


def crear_tabla_media_std(metrics: pd.DataFrame) -> pd.DataFrame:
    expected_seeds = len(config.SEEDS)
    seeds = (
        metrics.groupby(["model_id", "corpus"])["seed"]
        .nunique()
        .reset_index(name="n_seeds")
    )
    if not seeds["n_seeds"].eq(expected_seeds).all():
        raise ValueError("Hay grupos sin las tres semillas necesarias para calcular desviacion tipica.")

    grouped = (
        metrics.groupby(["model_id", "model", "corpus"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_macro_mean=("precision_macro", "mean"),
            precision_macro_std=("precision_macro", "std"),
            recall_macro_mean=("recall_macro", "mean"),
            recall_macro_std=("recall_macro", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    grouped["corpus_order"] = grouped["corpus"].map({name: i for i, name in enumerate(CORPUS_ORDER)})
    grouped["model_order"] = grouped["model_id"].map({name: i for i, name in enumerate(MODEL_LABELS)})
    return (
        grouped.sort_values(["model_order", "corpus_order"])
        .drop(columns=["model_order", "corpus_order"])
        .reset_index(drop=True)
    )


def crear_tabla_f1_descriptivo(metrics: pd.DataFrame) -> pd.DataFrame:
    corpus_metrics = metrics[metrics["corpus"].isin(["RAVDESS", "SAVEE", "TESS"])].copy()
    pivot = corpus_metrics.pivot_table(
        index=["model_id", "model", "seed"],
        columns="corpus",
        values="f1_macro",
        aggfunc="mean",
    ).reset_index()
    pivot["f1_equal_corpus"] = pivot[["RAVDESS", "SAVEE", "TESS"]].mean(axis=1)
    pivot["f1_ravdess_savee"] = pivot[["RAVDESS", "SAVEE"]].mean(axis=1)
    resumen = (
        pivot.groupby(["model_id", "model"], as_index=False)
        .agg(
            f1_mean_equal_corpus=("f1_equal_corpus", "mean"),
            f1_std_equal_corpus=("f1_equal_corpus", "std"),
            f1_mean_ravdess_savee=("f1_ravdess_savee", "mean"),
            f1_std_ravdess_savee=("f1_ravdess_savee", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    resumen["model_order"] = resumen["model_id"].map({name: i for i, name in enumerate(MODEL_LABELS)})
    return resumen.sort_values("model_order").drop(columns=["model_order"]).reset_index(drop=True)


def guardar_figura_f1_por_corpus(resumen: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = resumen[resumen["corpus"].isin(["RAVDESS", "SAVEE", "TESS"])].copy()
    model_order = list(MODEL_LABELS.values())
    y_base = {model: i for i, model in enumerate(model_order)}
    offsets = {"RAVDESS": -0.18, "SAVEE": 0.0, "TESS": 0.18}

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for corpus, corpus_df in data.groupby("corpus"):
        y = [y_base[model] + offsets[corpus] for model in corpus_df["model"]]
        ax.errorbar(
            corpus_df["f1_macro_mean"],
            y,
            xerr=corpus_df["f1_macro_std"],
            fmt="o",
            capsize=4,
            label=corpus,
            color=PALETTE[corpus],
            markeredgecolor="#1f2937",
        )
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 macro medio")
    ax.set_title("F1 macro por modelo y corpus")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(title="Corpus")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def guardar_figura_delta_mel(resumen: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    mel = resumen[
        resumen["model_id"].isin(MEL_MODELS)
        & resumen["corpus"].isin(["GLOBAL", "RAVDESS", "SAVEE", "TESS"])
    ].copy()
    pivot = mel.pivot_table(index="corpus", columns="model_id", values="f1_macro_mean", aggfunc="mean")
    deltas = pd.DataFrame(
        {
            "corpus": pivot.index,
            "Aumentacion - baseline": pivot["02_cnn_data_augmentation"] - pivot["01_cnn_baseline"],
            "Scheduler - baseline": pivot["03_cnn_scheduler"] - pivot["01_cnn_baseline"],
        }
    )
    deltas["corpus_order"] = deltas["corpus"].map({name: i for i, name in enumerate(CORPUS_ORDER)})
    deltas = deltas.sort_values("corpus_order")
    deltas["Aumentacion - baseline"] = deltas["Aumentacion - baseline"] * 100
    deltas["Scheduler - baseline"] = deltas["Scheduler - baseline"] * 100

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = list(range(len(deltas)))
    ax.axvline(0, color="#1f2937", linewidth=1)
    ax.scatter(deltas["Aumentacion - baseline"], [v - 0.12 for v in y], label="Aumentacion - baseline", color="#A8E6BD")
    ax.scatter(deltas["Scheduler - baseline"], [v + 0.12 for v in y], label="Scheduler - baseline", color="#9ED2F0")
    for y_value, row in zip(y, deltas.to_dict(orient="records")):
        valor_aug = row["Aumentacion - baseline"]
        valor_sched = row["Scheduler - baseline"]
        ax.text(valor_aug, y_value - 0.03, f"{valor_aug:.1f} pp", ha="center", va="bottom", fontsize=8)
        ax.text(valor_sched, y_value + 0.21, f"{valor_sched:.1f} pp", ha="center", va="bottom", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(deltas["corpus"])
    ax.set_xlabel("Diferencia media de F1 macro en puntos porcentuales")
    ax.set_title("Efecto de los cambios Mel respecto al baseline")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def generar_resumenes(results_dir: Path = RESULTS_DIR, require_mel_controlado: bool = True) -> dict[str, Path]:
    if require_mel_controlado:
        ok, problemas = mel_controlado_disponible(results_dir)
        if not ok:
            raise RuntimeError(
                "Los resultados Mel controlados todavia no estan completos. "
                "Reentrena 01, 02 y 03 antes de generar las tablas finales.\n"
                + "\n".join(problemas)
            )

    metrics = cargar_metricas_semillas(results_dir)
    resumen = crear_tabla_media_std(metrics)
    descriptivo = crear_tabla_f1_descriptivo(metrics)

    resumen_path = results_dir / "resumen_metricas_modelo_corpus.csv"
    descriptivo_path = results_dir / "resumen_f1_descriptivo.csv"
    fig_dir = results_dir / "figuras_resumen"
    f1_path = fig_dir / "f1_modelo_corpus_std.png"
    delta_path = fig_dir / "delta_f1_mel_controlado.png"

    resumen.to_csv(resumen_path, index=False, encoding="utf-8")
    descriptivo.to_csv(descriptivo_path, index=False, encoding="utf-8")
    guardar_figura_f1_por_corpus(resumen, f1_path)
    guardar_figura_delta_mel(resumen, delta_path)
    return {
        "resumen_metricas": resumen_path,
        "resumen_f1_descriptivo": descriptivo_path,
        "figura_f1_corpus": f1_path,
        "figura_delta_mel": delta_path,
    }
