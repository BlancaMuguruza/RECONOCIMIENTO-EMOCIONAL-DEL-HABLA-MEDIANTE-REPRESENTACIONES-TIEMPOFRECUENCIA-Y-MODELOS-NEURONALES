import marimo

__generated_with = "0.23.11"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src import config
    from src.external_datasets import (
        EXTERNAL_CHECKS_JSON,
        EXTERNAL_METADATA_CSV,
        EXTERNAL_RESULTS_DIR,
        cargar_metadata_externa,
    )
    from src.external_evaluation import MODEL_SPECS, SEEDS
    from src.modelos import MLPHuBERT, crear_cnn_mel, crear_cnn_wavelet
    from src.paths import RESULTS_DIR
    from src.utilidades import contar_parametros

    return (
        EXTERNAL_CHECKS_JSON,
        EXTERNAL_RESULTS_DIR,
        MLPHuBERT,
        MODEL_SPECS,
        RESULTS_DIR,
        SEEDS,
        cargar_metadata_externa,
        config,
        contar_parametros,
        crear_cnn_mel,
        crear_cnn_wavelet,
        json,
        mo,
        pd,
    )


@app.cell
def _(
    EXTERNAL_RESULTS_DIR,
    MLPHuBERT,
    MODEL_SPECS,
    SEEDS,
    config,
    contar_parametros,
    crear_cnn_mel,
    crear_cnn_wavelet,
    mo,
    pd,
):
    def leer_csv_08(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def bloque_imagen_08(path, titulo):
        if not path.exists():
            return mo.md(f"#### {titulo}\n\nPendiente de generar.")
        return mo.vstack([mo.md(f"#### {titulo}"), mo.image(str(path))])

    def porcentaje_08(value):
        return f"{float(value) * 100:.1f}%".replace(".", ",")

    def numero_08(value):
        if isinstance(value, float):
            return f"{value:.2f}".replace(".", ",")
        return f"{int(value):,}".replace(",", ".")

    def card_grid_08(values, columnas=3):
        cards = "\n".join(
            f"""
            <div style="border:1px solid #d8dee9; border-radius:8px; padding:10px 12px; background:#fbfcfe;">
              <div style="font-size:12px; color:#5f6b7a;">{key}</div>
              <div style="font-size:22px; font-weight:700; color:#1f2937;">{value}</div>
            </div>
            """
            for key, value in values.items()
        )
        return mo.Html(
            f"""
            <div style="display:grid; grid-template-columns:repeat({columnas},minmax(90px,1fr)); gap:8px;">
            {cards}
            </div>
            """
        )

    def resumen_modelo_08(model_id):
        resumen = leer_csv_08(EXTERNAL_RESULTS_DIR / "resumen_modelo_dataset.csv")
        if resumen.empty:
            return pd.DataFrame()
        resumen = resumen.copy()
        resumen["model"] = resumen["model_id"].map(lambda value: MODEL_SPECS[value].label)
        return resumen[resumen["model_id"].eq(model_id)].copy()

    def metricas_modelo_08(model_id):
        metrics = leer_csv_08(EXTERNAL_RESULTS_DIR / "metricas_externas.csv")
        if metrics.empty:
            filas = []
            for _seed_08 in SEEDS:
                for _dataset_08 in ["CREMA-D", "EMO-DB"]:
                    _safe_dataset_08 = _dataset_08.lower().replace("-", "_")
                    _path_08 = EXTERNAL_RESULTS_DIR / model_id / f"seed_{_seed_08}" / _safe_dataset_08 / "metricas.csv"
                    if _path_08.exists():
                        filas.append(pd.read_csv(_path_08))
            metrics = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
        if not metrics.empty:
            metrics = metrics.copy()
            metrics["model"] = metrics["model_id"].map(lambda value: MODEL_SPECS[value].label)
        return metrics[metrics["model_id"].eq(model_id)].copy() if not metrics.empty else metrics

    def clases_modelo_08(model_id):
        class_summary = leer_csv_08(EXTERNAL_RESULTS_DIR / "resumen_por_emocion.csv")
        if class_summary.empty:
            return pd.DataFrame()
        class_summary = class_summary.copy()
        class_summary["model"] = class_summary["model_id"].map(lambda value: MODEL_SPECS[value].label)
        return class_summary[class_summary["model_id"].eq(model_id)].copy()

    def modelo_preview_08(model_id):
        if model_id in {"01_cnn_baseline", "02_cnn_data_augmentation", "03_cnn_scheduler"}:
            return crear_cnn_mel(num_classes=len(config.EMOTIONS), dropout=config.MEL_CONTROLLED_DROPOUT)
        if model_id == "04_cnn_wavelet":
            return crear_cnn_wavelet(num_classes=len(config.EMOTIONS), dropout=0.50)
        return MLPHuBERT(input_dim=768, num_classes=len(config.EMOTIONS), dropout=0.45)

    def bloque_modelo_08(model_id):
        spec = MODEL_SPECS[model_id]
        resumen = resumen_modelo_08(model_id)
        metrics = metricas_modelo_08(model_id)
        class_summary = clases_modelo_08(model_id)
        params = contar_parametros(modelo_preview_08(model_id))["parametros_entrenables"]
        valores = {
            "Parametros": numero_08(params),
            "Seeds": ", ".join(map(str, SEEDS)),
            "Reentrenado": "No",
            "Representacion": spec.representation,
        }
        if not resumen.empty:
            for _dataset_08 in ["CREMA-D", "EMO-DB"]:
                _row_08 = resumen[resumen["dataset"].eq(_dataset_08)]
                if not _row_08.empty:
                    valores[f"F1 {_dataset_08}"] = porcentaje_08(_row_08.iloc[0].f1_macro_mean)
                    valores[f"Acc {_dataset_08}"] = porcentaje_08(_row_08.iloc[0].accuracy_mean)

        bloques = [mo.md(f"## {spec.label}"), card_grid_08(valores, columnas=2)]
        if resumen.empty:
            bloques.append(mo.md("Resultados pendientes. Ejecuta la evaluacion externa para este modelo."))
            return mo.vstack(bloques)

        _pivot_08 = resumen.pivot(index="model_id", columns="dataset", values="f1_macro_mean")
        if {"CREMA-D", "EMO-DB"} <= set(_pivot_08.columns):
            _delta_08 = (_pivot_08["CREMA-D"].iloc[0] - _pivot_08["EMO-DB"].iloc[0]) * 100
            bloques.append(card_grid_08({"Diferencia F1": f"{_delta_08:.1f} pp".replace(".", ",")}, columnas=1))

        bloques.append(
            mo.accordion(
                {
                    "Resultados por semilla": mo.ui.table(metrics.sort_values(["dataset", "seed"]).round(4)),
                    "Media y desviacion": mo.ui.table(resumen.round(4)),
                    "Resultados por emocion": mo.ui.table(class_summary.round(4)) if not class_summary.empty else mo.md("Pendiente."),
                }
            )
        )

        _fig_dir_08 = EXTERNAL_RESULTS_DIR / "figuras"
        _fig_dir_08.mkdir(parents=True, exist_ok=True)
        _emotion_fig_08 = _fig_dir_08 / f"{model_id}_f1_por_emocion.png"
        if not class_summary.empty:
            import matplotlib.pyplot as _plt_emotion_08

            _fig_08, _ax_08 = _plt_emotion_08.subplots(figsize=(7.5, 4))
            for _dataset_08, _group_08 in class_summary.groupby("dataset"):
                _ax_08.plot(_group_08["emotion"], _group_08["f1_mean"], marker="o", label=_dataset_08)
            _ax_08.set_ylim(0, 1.05)
            _ax_08.set_ylabel("F1")
            _ax_08.set_title(f"{spec.label}: F1 por emocion")
            _ax_08.grid(axis="y", linestyle="--", alpha=0.35)
            _ax_08.legend()
            _fig_08.tight_layout()
            _fig_08.savefig(_emotion_fig_08, dpi=250, bbox_inches="tight")
            _plt_emotion_08.close(_fig_08)

        bloques.append(bloque_imagen_08(_emotion_fig_08, "F1 por emocion"))

        matrices = {}
        for _dataset_08 in ["CREMA-D", "EMO-DB"]:
            for _seed_08 in SEEDS:
                _safe_dataset_08 = _dataset_08.lower().replace("-", "_")
                _path_08 = EXTERNAL_RESULTS_DIR / model_id / f"seed_{_seed_08}" / _safe_dataset_08 / "confusion_matrix_norm.png"
                matrices[f"{_dataset_08} seed {_seed_08}"] = bloque_imagen_08(_path_08, f"{_dataset_08} seed {_seed_08}")
        bloques.append(mo.accordion(matrices))

        if not class_summary.empty:
            _worst_08 = class_summary.sort_values("f1_mean").iloc[0]
            bloques.append(
                mo.md(
                    f"Confusion principal a revisar: el peor F1 medio aparece en "
                    f"{_worst_08.dataset}, emocion `{_worst_08.emotion}`."
                )
            )
        return mo.vstack(bloques)

    return (
        bloque_imagen_08,
        bloque_modelo_08,
        card_grid_08,
        leer_csv_08,
        numero_08,
    )


@app.cell
def _(
    EXTERNAL_CHECKS_JSON,
    EXTERNAL_RESULTS_DIR,
    MODEL_SPECS,
    RESULTS_DIR,
    bloque_imagen_08,
    card_grid_08,
    cargar_metadata_externa,
    json,
    leer_csv_08,
    mo,
    numero_08,
    pd,
):
    _df_ext_08 = cargar_metadata_externa()
    _included_08 = _df_ext_08[_df_ext_08["included"].astype(bool)].copy()
    _checks_08 = json.loads(EXTERNAL_CHECKS_JSON.read_text(encoding="utf-8")) if EXTERNAL_CHECKS_JSON.exists() else {}
    _mapping_08 = leer_csv_08(EXTERNAL_RESULTS_DIR / "mapeo_emociones.csv")
    _resumen_08 = leer_csv_08(EXTERNAL_RESULTS_DIR / "resumen_modelo_dataset.csv")
    _interno_08 = leer_csv_08(RESULTS_DIR / "resumen_metricas_modelo_corpus.csv")
    _model_labels_08 = {model_id: spec.label for model_id, spec in MODEL_SPECS.items()}
    _model_order_08 = [_model_labels_08[model_id] for model_id in MODEL_SPECS]
    if not _resumen_08.empty:
        _resumen_08 = _resumen_08.copy()
        _resumen_08["model"] = _resumen_08["model_id"].map(_model_labels_08)
    if not _interno_08.empty:
        _interno_08 = _interno_08.copy()
        _interno_08["model"] = _interno_08["model_id"].map(_model_labels_08)

    _cards_08 = {
        "Audios encontrados": numero_08(len(_df_ext_08)),
        "Incluidos": numero_08(len(_included_08)),
        "Excluidos": numero_08(len(_df_ext_08) - len(_included_08)),
        "Hablantes": numero_08(_included_08["speaker_id"].nunique()),
        "Corpus": numero_08(_included_08["dataset"].nunique()),
        "Emociones": numero_08(_included_08["emotion"].nunique()),
    }

    _fig_dir_08 = EXTERNAL_RESULTS_DIR / "figuras"
    _fig_dir_08.mkdir(parents=True, exist_ok=True)
    _points_fig_08 = _fig_dir_08 / "f1_externo_modelos.png"
    if not _resumen_08.empty:
        import matplotlib.pyplot as _plt_points_08

        _offsets_08 = {"CREMA-D": -0.12, "EMO-DB": 0.12}
        _colors_08 = {"CREMA-D": "#C8B6FF", "EMO-DB": "#FFD6A5"}
        _fig_08, _ax_08 = _plt_points_08.subplots(figsize=(9, 4.8))
        for _dataset_08, _group_08 in _resumen_08.groupby("dataset"):
            _group_08 = _group_08.set_index("model").reindex(_model_order_08).reset_index()
            _y_08 = [idx + _offsets_08[_dataset_08] for idx in range(len(_group_08))]
            _ax_08.errorbar(
                _group_08["f1_macro_mean"],
                _y_08,
                xerr=_group_08["f1_macro_std"],
                fmt="o",
                capsize=4,
                label=_dataset_08,
                color=_colors_08[_dataset_08],
                markeredgecolor="#1f2937",
            )
        _ax_08.set_yticks(range(len(_model_order_08)))
        _ax_08.set_yticklabels(_model_order_08)
        _ax_08.set_xlim(0, 1.05)
        _ax_08.set_xlabel("F1 macro")
        _ax_08.set_title("Evaluacion externa por modelo")
        _ax_08.grid(axis="x", linestyle="--", alpha=0.35)
        _ax_08.legend()
        _fig_08.tight_layout()
        _fig_08.savefig(_points_fig_08, dpi=250, bbox_inches="tight")
        _plt_points_08.close(_fig_08)

    _heatmap_fig_08 = _fig_dir_08 / "heatmap_f1_interno_externo.png"
    if not _resumen_08.empty and not _interno_08.empty:
        import matplotlib.pyplot as _plt_heat_08
        import seaborn as _sns_heat_08

        _internal_map_08 = _interno_08[["model_id", "model", "corpus", "f1_macro_mean"]].copy()
        _external_map_08 = _resumen_08[["model_id", "model", "dataset", "f1_macro_mean"]].rename(columns={"dataset": "corpus"})
        _heat_df_08 = pd.concat([_internal_map_08, _external_map_08], ignore_index=True)
        _columns_08 = ["RAVDESS", "SAVEE", "TESS", "CREMA-D", "EMO-DB"]
        _pivot_heat_08 = _heat_df_08.pivot_table(index="model", columns="corpus", values="f1_macro_mean", aggfunc="mean")
        _pivot_heat_08 = _pivot_heat_08.reindex(index=_model_order_08, columns=_columns_08)
        _fig_heat_08, _ax_heat_08 = _plt_heat_08.subplots(figsize=(8.5, 4.8))
        _sns_heat_08.heatmap(
            _pivot_heat_08,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            ax=_ax_heat_08,
            cbar_kws={"label": "F1 macro medio"},
        )
        _ax_heat_08.set_title("F1 interno y externo")
        _ax_heat_08.set_xlabel("Corpus")
        _ax_heat_08.set_ylabel("Modelo")
        _fig_heat_08.tight_layout()
        _fig_heat_08.savefig(_heatmap_fig_08, dpi=250, bbox_inches="tight")
        _plt_heat_08.close(_fig_heat_08)

    mo.vstack(
        [
            mo.md("## Evaluacion externa"),
            mo.md(
                "CREMA-D y EMO-DB se usan solo como test externo. No se reentrena, no se ajustan hiperparametros "
                "y no se seleccionan checkpoints con estos datos."
            ),
            card_grid_08(_cards_08, columnas=2),
            mo.accordion(
                {
                    "Mapeo de emociones": mo.ui.table(_mapping_08),
                    "Calidad de metadatos": mo.ui.table(pd.DataFrame([_checks_08])),
                    "Distribucion por emocion": mo.ui.table(
                        _included_08.groupby(["dataset", "emotion"]).size().reset_index(name="n_audios")
                    ),
                    "Duracion y frecuencia": mo.ui.table(
                        _included_08.groupby("dataset").agg(
                            duration_mean=("duration_seconds", "mean"),
                            duration_min=("duration_seconds", "min"),
                            duration_max=("duration_seconds", "max"),
                            sample_rates=("sample_rate_original", "nunique"),
                        ).reset_index().round(4)
                    ),
                    "Audios por hablante": mo.ui.table(
                        _included_08.groupby(["dataset", "speaker_id"]).size().reset_index(name="n_audios")
                    ),
                    "Excluidos": mo.ui.table(
                        _df_ext_08[~_df_ext_08["included"].astype(bool)][
                            ["dataset", "file_id", "emotion_original", "exclusion_reason"]
                        ].head(200)
                    ),
                }
            ),
            bloque_imagen_08(_points_fig_08, "F1 externo con barras de error"),
            bloque_imagen_08(_heatmap_fig_08, "Mapa de calor interno y externo"),
            mo.md("La evaluacion interna y externa se muestran juntas solo como referencia descriptiva; no son el mismo protocolo."),
        ]
    )
    return


@app.cell(column=1)
def _():
    #
    return


@app.cell(column=2)
def _(bloque_modelo_08):
    bloque_modelo_08("01_cnn_baseline")
    return


@app.cell(column=3)
def _():
    #
    return


@app.cell(column=4)
def _(bloque_modelo_08):
    bloque_modelo_08("02_cnn_data_augmentation")
    return


@app.cell(column=5)
def _():
    #
    return


@app.cell(column=6)
def _(bloque_modelo_08):
    bloque_modelo_08("03_cnn_scheduler")
    return


@app.cell(column=7)
def _():
    #
    return


@app.cell(column=8)
def _(bloque_modelo_08):
    bloque_modelo_08("04_cnn_wavelet")
    return


@app.cell(column=9)
def _():
    #
    return


@app.cell(column=10)
def _(bloque_modelo_08):
    bloque_modelo_08("05_hubert")
    return


@app.cell(column=11)
def _():
    #
    return


if __name__ == "__main__":
    app.run()
