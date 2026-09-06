from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .datos import resumen_datos, tabla_conteos
from .utilidades import contar_parametros


def add_project_to_path() -> None:
    import sys

    # Permite importar src aunque el notebook se abra desde otra carpeta.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _numero(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    return str(value)


def _decimal(value: float) -> str:
    if value >= 100:
        text = f"{value:,.0f}"
    elif value >= 10:
        text = f"{value:.1f}"
    elif value >= 1:
        text = f"{value:.2f}"
    else:
        text = f"{value:.4f}"
    return text.replace(",", ".")


def _porcentaje(value: float) -> str:
    return f"{value * 100:.1f}%"


def _cards(values: dict):
    import marimo as mo

    cards = "\n".join(
        f"""
        <div style="border:1px solid #d8dee9; border-radius:8px; padding:12px 14px; background:#fbfcfe;">
          <div style="font-size:12px; color:#5f6b7a;">{key}</div>
          <div style="font-size:24px; font-weight:700; color:#1f2937;">{value}</div>
        </div>
        """
        for key, value in values.items()
    )
    return mo.Html(
        f"""
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px;">
        {cards}
        </div>
        """
    )


def _observaciones_entrada(n_train: int, representation: str | None) -> tuple[str, int] | None:
    if representation == "mel":
        frames_audio = 1 + config.N_SAMPLES // config.HOP_LENGTH
        return "Frames train", n_train * frames_audio
    if representation == "wavelet":
        return "Posiciones wavelet", n_train * config.WAVELET_TIME_BINS
    if representation in {"hubert_mean", "hubert_sequence"}:
        return "Frames HuBERT aprox.", n_train * config.HUBERT_FRAMES_PER_AUDIO
    return None


def stats_cards(df: pd.DataFrame, model=None, representation: str | None = None, extra: dict | None = None):
    summary = resumen_datos(df)
    n_train = max(summary["n_train"], 1)
    values = {
        "Datos totales": _numero(summary["n_datos"]),
        "Muestras train": _numero(summary["n_train"]),
        "Valid": _numero(summary["n_valid"]),
        "Test": _numero(summary["n_test"]),
        "Clases": _numero(len(config.EMOTIONS)),
        "Seeds": _numero(len(config.SEEDS)),
    }

    if model is not None:
        model_params = contar_parametros(model)
        trainable = model_params["parametros_entrenables"]
        values.update(
            {
                "Parametros": _numero(trainable),
                "Param/muestra": _decimal(trainable / n_train),
            }
        )
        entrada = _observaciones_entrada(n_train, representation)
        if entrada is not None:
            label, n_obs = entrada
            values[label] = _numero(n_obs)
            values["Datos/param"] = _decimal(n_obs / max(trainable, 1))

    for key, value in (extra or {}).items():
        values[key] = _numero(value)
    return _cards(values)


def mostrar_conteos(df: pd.DataFrame):
    import marimo as mo

    return mo.accordion(
        {
            "Conteos por split y dataset": mo.ui.table(tabla_conteos(df, "split", "dataset")),
            "Conteos por split y emocion": mo.ui.table(tabla_conteos(df, "split", "emotion")),
        }
    )


def _metricas_globales(output_dir: Path) -> pd.DataFrame:
    filas = []
    for seed_dir in sorted(output_dir.glob("seed_*")):
        seed = seed_dir.name.replace("seed_", "")
        metrics_path = seed_dir / f"metricas_seed_{seed}.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            filas.append(metrics[metrics["group"].eq("test_global")])
    if not filas:
        return pd.DataFrame()
    metrics = pd.concat(filas, ignore_index=True)
    cols = ["seed", "n_datos", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    return metrics[cols].round(4)


def _imagen(path: Path, title: str):
    import marimo as mo

    if not path.exists():
        return mo.md(f"**{title}**\n\nPendiente de generar.")
    return mo.vstack([mo.md(f"#### {title}"), mo.image(str(path))])


def _metricas_cards(metrics: pd.DataFrame):
    resumen = metrics.drop(columns=["seed"]).mean(numeric_only=True)
    values = {
        "Test": _numero(int(round(resumen.get("n_datos", 0)))),
        "Accuracy media": _porcentaje(float(resumen.get("accuracy", 0))),
        "F1 macro medio": _porcentaje(float(resumen.get("f1_macro", 0))),
        "Precision macro": _porcentaje(float(resumen.get("precision_macro", 0))),
        "Recall macro": _porcentaje(float(resumen.get("recall_macro", 0))),
    }
    return _cards(values)


def mostrar_resultados_experimento(output_dir: Path):
    import marimo as mo

    metrics = _metricas_globales(output_dir)
    bloques = [mo.md("## Resultados")]

    if metrics.empty:
        bloques.append(mo.md("Todavia no hay resultados guardados. Entrena el modelo para ver curvas y matrices aqui."))
        return mo.vstack(bloques)

    resumen = metrics.drop(columns=["seed"]).mean(numeric_only=True).to_frame("media").T.round(4)
    bloques.extend(
        [
            _metricas_cards(metrics),
            mo.accordion(
                {
                    "Metricas por semilla": mo.ui.table(metrics),
                    "Media de las 3 semillas": mo.ui.table(resumen),
                }
            ),
        ]
    )

    for seed in metrics["seed"].tolist():
        seed_dir = output_dir / f"seed_{seed}"
        bloques.append(mo.md(f"### Semilla {seed}"))
        bloques.append(
            mo.hstack(
                [
                    _imagen(seed_dir / "loss_curve.png", "Curva de entrenamiento"),
                    _imagen(seed_dir / f"confusion_matrix_norm_seed_{seed}.png", "Matriz normalizada"),
                ],
                widths="equal",
            )
        )
        bloques.append(_imagen(seed_dir / f"confusion_matrix_seed_{seed}.png", "Matriz absoluta"))

    return mo.vstack(bloques)


def mostrar_imagenes_eda(eda_dir: Path):
    import marimo as mo

    return mo.vstack(
        [
            mo.md("## Graficas generadas"),
            mo.hstack(
                [
                    _imagen(eda_dir / "conteo_dataset.png", "Datos por dataset"),
                    _imagen(eda_dir / "conteo_emocion.png", "Datos por emocion"),
                ],
                widths="equal",
            ),
            _imagen(eda_dir / "conteo_split.png", "Datos por split"),
        ]
    )

