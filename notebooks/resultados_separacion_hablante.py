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
    from src.hubert_speaker_analysis import (
        MODEL_SEED,
        PROTOCOLS,
        RESULTS_ROOT,
        SPLIT_SEEDS,
        TARGET_DATASETS,
        TARGET_EMOTIONS,
        cargar_metadata_experimento,
        comprobar_metadata_y_embeddings,
    )

    return (
        MODEL_SEED,
        PROTOCOLS,
        RESULTS_ROOT,
        SPLIT_SEEDS,
        TARGET_DATASETS,
        cargar_metadata_experimento,
        comprobar_metadata_y_embeddings,
        config,
        json,
        mo,
        pd,
    )


@app.cell
def _(mo, pd):
    def bloque_imagen_07(path, titulo):
        if not path.exists():
            return mo.md(f"#### {titulo}\n\nPendiente de generar.")
        return mo.vstack([mo.md(f"#### {titulo}"), mo.image(str(path))])

    def leer_csv_07(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def numero_07(value):
        if isinstance(value, float):
            return f"{value:.2f}".replace(".", ",")
        return f"{int(value):,}".replace(",", ".")

    def porcentaje_07(value):
        return f"{float(value) * 100:.1f}%".replace(".", ",")

    def card_grid_07(valores, columnas=3):
        cards = "\n".join(
            f"""
            <div style="border:1px solid #d8dee9; border-radius:8px; padding:10px 12px; background:#fbfcfe;">
              <div style="font-size:12px; color:#5f6b7a;">{key}</div>
              <div style="font-size:22px; font-weight:700; color:#1f2937;">{value}</div>
            </div>
            """
            for key, value in valores.items()
        )
        return mo.Html(
            f"""
            <div style="display:grid; grid-template-columns:repeat({columnas},minmax(88px,1fr)); gap:8px;">
            {cards}
            </div>
            """
        )

    def tabla_bonita_corpus_07(media, output_dir, titulo):
        if media.empty:
            return mo.md("Comparacion por corpus pendiente.")
        swatches = {"GLOBAL": "#D8DEE9", "RAVDESS": "#9ED2F0", "SAVEE": "#A8E6BD"}
        filas = []
        for row in media.itertuples(index=False):
            color = swatches.get(row.corpus, "#d8dee9")
            filas.append(
                f"""
                <tr>
                  <td><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{color}; margin-right:7px;"></span><strong>{row.corpus}</strong></td>
                  <td>{int(row.n_audios)}</td>
                  <td>{int(row.n_speakers)}</td>
                  <td>{porcentaje_07(row.accuracy)}</td>
                  <td>{porcentaje_07(row.precision_macro)}</td>
                  <td>{porcentaje_07(row.recall_macro)}</td>
                  <td><strong>{porcentaje_07(row.f1_macro)}</strong></td>
                </tr>
                """
            )
        return mo.Html(
            f"""
            <div style="border:1px solid #d8dee9; border-radius:8px; overflow:hidden; background:white;">
              <table style="width:100%; border-collapse:collapse; font-size:13px;">
                <thead>
                  <tr style="background:#eef4fb; color:#1f2937;">
                    <th style="text-align:left; padding:9px 10px;">Corpus</th>
                    <th style="padding:9px 10px;">Audios</th>
                    <th style="padding:9px 10px;">Hablantes</th>
                    <th style="padding:9px 10px;">Accuracy</th>
                    <th style="padding:9px 10px;">Precision</th>
                    <th style="padding:9px 10px;">Recall</th>
                    <th style="padding:9px 10px;">F1 macro</th>
                  </tr>
                </thead>
                <tbody>{''.join(filas)}</tbody>
              </table>
            </div>
            <div style="font-size:12px; color:#5f6b7a; margin-top:4px;">{titulo}</div>
            """
        )

    def metricas_protocol_07(results_root, protocol, split_seeds):
        filas = []
        for _split_seed in split_seeds:
            path = results_root / protocol / f"split_seed_{_split_seed}" / "metricas.csv"
            if path.exists():
                metricas = pd.read_csv(path)
                filas.append(metricas)
        return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()

    def metricas_principales_07(metrics):
        if metrics.empty:
            return metrics
        return metrics[
            metrics["corpus"].isin(["GLOBAL", "RAVDESS", "SAVEE"])
            & metrics["speaker_id"].fillna("").eq("")
        ].copy()

    def resumen_metricas_protocol_07(metrics):
        principales = metricas_principales_07(metrics)
        if principales.empty:
            return pd.DataFrame()
        return (
            principales.groupby("corpus", as_index=False)
            .agg(
                n_audios=("n_audios", "mean"),
                n_speakers=("n_speakers", "mean"),
                accuracy=("accuracy", "mean"),
                precision_macro=("precision_macro", "mean"),
                recall_macro=("recall_macro", "mean"),
                f1_macro=("f1_macro", "mean"),
            )
            .assign(
                orden=lambda df: df["corpus"].map({"GLOBAL": 0, "RAVDESS": 1, "SAVEE": 2})
            )
            .sort_values("orden")
            .drop(columns=["orden"])
        )

    def split_checks_table_07(results_root, protocol, split_seeds, json_module):
        rows = []
        for _split_seed in split_seeds:
            path = results_root / protocol / f"split_seed_{_split_seed}" / "split_checks.json"
            if not path.exists():
                continue
            checks = json_module.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "split_seed": _split_seed,
                    "train": checks["n_audios_by_split"]["train"],
                    "valid": checks["n_audios_by_split"]["valid"],
                    "test": checks["n_audios_by_split"]["test"],
                    "hablantes_train": checks["n_speakers_by_split"]["train"],
                    "hablantes_valid": checks["n_speakers_by_split"]["valid"],
                    "hablantes_test": checks["n_speakers_by_split"]["test"],
                    "fuga_train_valid": len(checks["speaker_overlap_train_valid"]),
                    "fuga_train_test": len(checks["speaker_overlap_train_test"]),
                    "fuga_valid_test": len(checks["speaker_overlap_valid_test"]),
                }
            )
        return pd.DataFrame(rows)

    def bloque_protocolo_07(results_root, protocol, split_seeds, json_module):
        metrics = metricas_protocol_07(results_root, protocol, split_seeds)
        principales = metricas_principales_07(metrics)
        checks = split_checks_table_07(results_root, protocol, split_seeds, json_module)
        media = resumen_metricas_protocol_07(metrics)

        bloques = [mo.md(f"## {protocol}")]
        if checks.empty:
            bloques.append(mo.md("Ejecuta `python scripts/train_06_hubert_speaker_analysis.py --prepare-only` para ver los splits."))
            return mo.vstack(bloques)

        valores = {
            "Repeticiones": numero_07(len(checks)),
            "Train medio": numero_07(checks["train"].mean()),
            "Valid medio": numero_07(checks["valid"].mean()),
            "Test medio": numero_07(checks["test"].mean()),
            "Hablantes train": numero_07(checks["hablantes_train"].mean()),
            "Fugas max": numero_07(checks[["fuga_train_valid", "fuga_train_test", "fuga_valid_test"]].max().max()),
        }
        if not principales.empty:
            global_rows = principales[principales["corpus"].eq("GLOBAL")]
            valores["Accuracy"] = porcentaje_07(global_rows["accuracy"].mean())
            valores["F1 macro"] = porcentaje_07(global_rows["f1_macro"].mean())
        bloques.append(card_grid_07(valores, columnas=2))

        if not principales.empty:
            bloques.extend(
                [
                    mo.md("### Comparacion por corpus"),
                    tabla_bonita_corpus_07(media, results_root / protocol, f"{protocol}: media de repeticiones"),
                ]
            )
        else:
            bloques.append(mo.md("Resultados pendientes. Los splits ya estan preparados, pero faltan entrenamientos."))

        acordeon = {
            "Comprobaciones de splits": mo.ui.table(checks),
        }
        for _split_seed in split_seeds:
            base = results_root / protocol / f"split_seed_{_split_seed}"
            assignment = leer_csv_07(base / "speaker_assignment.csv")
            split_metadata = leer_csv_07(base / "split_metadata.csv")
            metricas = leer_csv_07(base / "metricas.csv")
            if not assignment.empty:
                acordeon[f"Hablantes split_seed {_split_seed}"] = mo.ui.table(assignment)
            if not split_metadata.empty:
                dist = (
                    split_metadata.groupby(["split", "dataset", "emotion"])
                    .size()
                    .reset_index(name="n_audios")
                )
                acordeon[f"Distribucion split_seed {_split_seed}"] = mo.ui.table(dist)
            if not metricas.empty:
                acordeon[f"Metricas split_seed {_split_seed}"] = mo.ui.table(metricas.round(4))
        bloques.append(mo.accordion(acordeon))

        curvas = []
        matrices = []
        for _split_seed in split_seeds:
            base = results_root / protocol / f"split_seed_{_split_seed}"
            curvas.append(bloque_imagen_07(base / "loss_curve.png", f"Curva split_seed {_split_seed}"))
            matrices.append(
                mo.vstack(
                    [
                        mo.md(f"#### Matrices split_seed {_split_seed}"),
                        mo.hstack(
                            [
                                bloque_imagen_07(base / "confusion_matrix_norm.png", "Normalizada"),
                                bloque_imagen_07(base / "confusion_matrix.png", "Absoluta"),
                            ],
                            widths="equal",
                        ),
                    ]
                )
            )

        bloques.extend(
            [
                mo.md("### Curvas de entrenamiento"),
                mo.vstack(curvas),
                mo.md("### Matrices de confusion"),
                mo.vstack(matrices),
            ]
        )
        return mo.vstack(bloques)

    return (
        bloque_imagen_07,
        bloque_protocolo_07,
        card_grid_07,
        leer_csv_07,
        numero_07,
        porcentaje_07,
    )


@app.cell
def _(
    MODEL_SEED,
    SPLIT_SEEDS,
    TARGET_DATASETS,
    card_grid_07,
    cargar_metadata_experimento,
    comprobar_metadata_y_embeddings,
    config,
    mo,
    numero_07,
    pd,
):
    _df_meta_07 = cargar_metadata_experimento()
    _checks_meta_07 = comprobar_metadata_y_embeddings(_df_meta_07)
    _valores_07 = {
        "Audios": numero_07(len(_df_meta_07)),
        "Corpus": ", ".join(TARGET_DATASETS),
        "Hablantes": numero_07(_df_meta_07["speaker_id"].nunique()),
        "Emociones": numero_07(_df_meta_07["emotion"].nunique()),
        "Repeticiones": numero_07(len(SPLIT_SEEDS)),
        "model_seed": numero_07(MODEL_SEED),
        "Batch": numero_07(config.BATCH_SIZE),
        "Epochs max": numero_07(config.EPOCHS),
    }
    mo.vstack(
        [
            mo.md("## Objetivo"),
            mo.md(
                "Evaluar si la cabeza MLP sobre embeddings HuBERT congelados mantiene rendimiento cuando "
                "los hablantes de test no aparecen en entrenamiento. TESS queda fuera para que la comparacion "
                "sea solo RAVDESS + SAVEE."
            ),
            card_grid_07(_valores_07, columnas=2),
            mo.accordion(
                {
                    "Comprobaciones": mo.ui.table(
                        pd.DataFrame([_checks_meta_07]).drop(columns=["missing_embeddings", "speakers_missing_emotions"])
                    ),
                    "Excepciones por hablante": mo.ui.table(pd.DataFrame(_checks_meta_07["speakers_missing_emotions"])),
                    "Metadatos del experimento": mo.ui.table(_df_meta_07[["dataset", "speaker_id", "emotion", "file_id"]].head(30)),
                }
            ),
        ]
    )
    return


@app.cell(column=1)
def _():
    #
    return


@app.cell(column=2)
def _(
    RESULTS_ROOT,
    SPLIT_SEEDS,
    bloque_imagen_07,
    card_grid_07,
    config,
    leer_csv_07,
    mo,
    pd,
    porcentaje_07,
):
    _summary_path_07 = RESULTS_ROOT / "resumen_media_std.csv"
    _comparison_path_07 = RESULTS_ROOT / "comparacion_protocolos.csv"
    _fig_dir_07 = RESULTS_ROOT / "figuras"
    _fig_dir_07.mkdir(parents=True, exist_ok=True)
    _summary_07 = leer_csv_07(_summary_path_07)
    _comparison_07 = leer_csv_07(_comparison_path_07)
    _fig_f1_07 = _fig_dir_07 / "f1_protocolos_errorbar.png"
    _fig_delta_07 = _fig_dir_07 / "delta_speaker_disjoint.png"
    _fig_matrix_media_07 = _fig_dir_07 / "confusion_matrix_norm_media.png"

    if not _summary_07.empty:
        import matplotlib.pyplot as _plt_summary_07

        _corpus_order_07 = ["GLOBAL", "RAVDESS", "SAVEE"]
        _offsets_07 = {"random_audio": -0.12, "speaker_disjoint": 0.12}
        _colors_07 = {"random_audio": "#9ED2F0", "speaker_disjoint": "#F7A8C4"}
        _fig_07, _ax_07 = _plt_summary_07.subplots(figsize=(8, 4.2))
        for _protocol_name_07, _group_07 in _summary_07.groupby("protocol"):
            _group_07 = _group_07.set_index("corpus").loc[_corpus_order_07].reset_index()
            _y_07 = [idx + _offsets_07[_protocol_name_07] for idx in range(len(_group_07))]
            _ax_07.errorbar(
                _group_07["f1_macro_mean"],
                _y_07,
                xerr=_group_07["f1_macro_std"],
                fmt="o",
                capsize=4,
                label=_protocol_name_07,
                color=_colors_07[_protocol_name_07],
                markeredgecolor="#1f2937",
            )
        _ax_07.set_yticks(range(len(_corpus_order_07)))
        _ax_07.set_yticklabels(_corpus_order_07)
        _ax_07.set_xlim(0, 1.05)
        _ax_07.set_xlabel("F1 macro")
        _ax_07.set_title("F1 macro por protocolo y corpus")
        _ax_07.grid(axis="x", linestyle="--", alpha=0.35)
        _ax_07.legend()
        _fig_07.tight_layout()
        _fig_07.savefig(_fig_f1_07, dpi=250, bbox_inches="tight")
        _plt_summary_07.close(_fig_07)

    if not _comparison_07.empty:
        import matplotlib.pyplot as _plt_delta_07

        _plot_delta_07 = _comparison_07.copy()
        _plot_delta_07["delta_pp"] = _plot_delta_07["delta_f1"] * 100
        _plot_delta_07 = _plot_delta_07.set_index("corpus").loc[["GLOBAL", "RAVDESS", "SAVEE"]].reset_index()
        _fig_delta_local_07, _ax_delta_07 = _plt_delta_07.subplots(figsize=(7.8, 4))
        _y_delta_07 = list(range(len(_plot_delta_07)))
        _ax_delta_07.axvline(0, color="#1f2937", linewidth=1)
        _ax_delta_07.scatter(_plot_delta_07["delta_pp"], _y_delta_07, color="#F7A8C4", edgecolor="#1f2937")
        for _row_07, _y_value_07 in zip(_plot_delta_07.itertuples(index=False), _y_delta_07):
            _ax_delta_07.text(_row_07.delta_pp, _y_value_07 + 0.08, f"{_row_07.delta_pp:.1f} pp", ha="center", fontsize=9)
        _ax_delta_07.set_yticks(_y_delta_07)
        _ax_delta_07.set_yticklabels(_plot_delta_07["corpus"])
        _ax_delta_07.set_xlabel("speaker_disjoint - random_audio en puntos porcentuales de F1")
        _ax_delta_07.set_title("Cambio al exigir hablantes no vistos")
        _ax_delta_07.grid(axis="x", linestyle="--", alpha=0.35)
        _fig_delta_local_07.tight_layout()
        _fig_delta_local_07.savefig(_fig_delta_07, dpi=250, bbox_inches="tight")
        _plt_delta_07.close(_fig_delta_local_07)

    _pred_paths_07 = [
        RESULTS_ROOT / "speaker_disjoint" / f"split_seed_{_split_seed_07}" / "predicciones.csv"
        for _split_seed_07 in SPLIT_SEEDS
    ]
    if all(_path_07.exists() for _path_07 in _pred_paths_07):
        import matplotlib.pyplot as _plt_matrix_07
        import seaborn as _sns_matrix_07
        from sklearn.metrics import confusion_matrix as _confusion_matrix_07

        _matrices_07 = []
        for _pred_path_07 in _pred_paths_07:
            _pred_df_07 = pd.read_csv(_pred_path_07)
            _matrix_07 = _confusion_matrix_07(
                _pred_df_07["y_true"],
                _pred_df_07["y_pred"],
                labels=list(range(len(config.EMOTIONS))),
                normalize="true",
            )
            _matrices_07.append(_matrix_07 * 100)
        _mean_matrix_07 = sum(_matrices_07) / len(_matrices_07)
        _fig_matrix_07, _ax_matrix_07 = _plt_matrix_07.subplots(figsize=(6.5, 5.8))
        _sns_matrix_07.heatmap(
            _mean_matrix_07,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            vmin=0,
            vmax=100,
            xticklabels=config.EMOTIONS,
            yticklabels=config.EMOTIONS,
            linewidths=0.5,
            linecolor="white",
            ax=_ax_matrix_07,
            cbar_kws={"label": "Porcentaje medio por clase real"},
        )
        _ax_matrix_07.set_title("Matriz normalizada media")
        _ax_matrix_07.set_xlabel("Prediccion")
        _ax_matrix_07.set_ylabel("Real")
        _fig_matrix_07.tight_layout()
        _fig_matrix_07.savefig(_fig_matrix_media_07, dpi=250, bbox_inches="tight")
        _plt_matrix_07.close(_fig_matrix_07)

    _bloques_07 = [mo.md("## Resumen final")]
    if _summary_07.empty or _comparison_07.empty:
        _bloques_07.append(
            mo.md(
                "Pendiente de entrenar y generar resumenes. Cuando ejecutes `--summaries`, aqui apareceran "
                "las medias, desviaciones y la perdida por separacion de hablantes."
            )
        )
    else:
        _global_random_07 = _summary_07[
            _summary_07["protocol"].eq("random_audio") & _summary_07["corpus"].eq("GLOBAL")
        ].iloc[0]
        _global_speaker_07 = _summary_07[
            _summary_07["protocol"].eq("speaker_disjoint") & _summary_07["corpus"].eq("GLOBAL")
        ].iloc[0]
        _delta_global_07 = _comparison_07[_comparison_07["corpus"].eq("GLOBAL")].iloc[0]
        _bloques_07.append(
            card_grid_07(
                {
                    "F1 random": porcentaje_07(_global_random_07.f1_macro_mean),
                    "F1 hablantes": porcentaje_07(_global_speaker_07.f1_macro_mean),
                    "Delta F1": f"{_delta_global_07.delta_f1 * 100:.1f} pp".replace(".", ","),
                    "Perdida relativa": f"{_delta_global_07.relative_loss_pct:.1f}%".replace(".", ","),
                },
                columnas=2,
            )
        )
        _vista_07 = _summary_07.copy()
        _vista_07["F1 macro"] = _vista_07.apply(
            lambda row: f"{row.f1_macro_mean * 100:.1f} +/- {row.f1_macro_std * 100:.1f}%".replace(".", ","),
            axis=1,
        )
        _bloques_07.append(
            mo.accordion(
                {
                    "Media y desviacion": mo.ui.table(
                        _vista_07[["protocol", "corpus", "n_repeticiones", "accuracy_mean", "f1_macro_mean", "F1 macro"]].round(4)
                    ),
                    "Comparacion protocolos": mo.ui.table(_comparison_07.round(4)),
                }
            )
        )

    _bloques_07.extend(
        [
            bloque_imagen_07(_fig_f1_07, "F1 por protocolo y corpus"),
            bloque_imagen_07(_fig_delta_07, "Delta por hablantes no vistos"),
            bloque_imagen_07(_fig_matrix_media_07, "Matriz media normalizada"),
        ]
    )
    mo.vstack(_bloques_07)
    return


@app.cell(column=3)
def _():
    #
    return


@app.cell(column=4)
def _(RESULTS_ROOT, SPLIT_SEEDS, bloque_protocolo_07, json):
    bloque_protocolo_07(RESULTS_ROOT, "random_audio", SPLIT_SEEDS, json)
    return


@app.cell(column=5)
def _():
    #
    return


@app.cell(column=6)
def _(RESULTS_ROOT, SPLIT_SEEDS, bloque_protocolo_07, json):
    bloque_protocolo_07(RESULTS_ROOT, "speaker_disjoint", SPLIT_SEEDS, json)
    return


@app.cell(column=7)
def _():
    #
    return


@app.cell(column=8)
def _(
    PROTOCOLS,
    RESULTS_ROOT,
    SPLIT_SEEDS,
    bloque_imagen_07,
    leer_csv_07,
    mo,
    pd,
):
    _speaker_rows_07 = []
    for _protocol_07 in PROTOCOLS:
        for _split_seed_07 in SPLIT_SEEDS:
            _base_07 = RESULTS_ROOT / _protocol_07 / f"split_seed_{_split_seed_07}"
            _summary_speaker_07 = leer_csv_07(_base_07 / "speaker_test_summary.csv")
            _metrics_07 = leer_csv_07(_base_07 / "metricas.csv")
            if _summary_speaker_07.empty or _metrics_07.empty:
                continue
            _speaker_metrics_07 = _metrics_07[_metrics_07["speaker_id"].fillna("").ne("")]
            _speaker_rows_07.append(
                _speaker_metrics_07.merge(_summary_speaker_07, on=["speaker_id", "corpus", "n_audios"], how="left")
            )

    _speaker_df_07 = pd.concat(_speaker_rows_07, ignore_index=True) if _speaker_rows_07 else pd.DataFrame()
    _fig_speaker_07 = RESULTS_ROOT / "figuras" / "f1_por_hablante.png"
    if not _speaker_df_07.empty:
        import matplotlib.pyplot as _plt_speaker_07

        _fig_speaker_07.parent.mkdir(parents=True, exist_ok=True)
        _plot_df_07 = _speaker_df_07.sort_values(["protocol", "corpus", "speaker_id"])
        _colors_07 = _plot_df_07["corpus"].map({"RAVDESS": "#9ED2F0", "SAVEE": "#A8E6BD"}).tolist()
        _fig_07, _ax_07 = _plt_speaker_07.subplots(figsize=(9, 4.6))
        _x_07 = list(range(len(_plot_df_07)))
        _ax_07.scatter(_x_07, _plot_df_07["f1_macro"], color=_colors_07, edgecolor="#1f2937")
        _ax_07.set_xticks(_x_07)
        _ax_07.set_xticklabels(_plot_df_07["speaker_id"], rotation=70, ha="right", fontsize=8)
        _ax_07.set_ylim(0, 1.05)
        _ax_07.set_ylabel("F1 macro")
        _ax_07.set_title("F1 por hablante de test")
        _ax_07.grid(axis="y", linestyle="--", alpha=0.35)
        _fig_07.tight_layout()
        _fig_07.savefig(_fig_speaker_07, dpi=250, bbox_inches="tight")
        _plt_speaker_07.close(_fig_07)

    _bloques_07 = [
        mo.md("## Rendimiento por hablante"),
        bloque_imagen_07(_fig_speaker_07, "F1 por hablante"),
    ]
    if _speaker_df_07.empty:
        _bloques_07.append(mo.md("Pendiente de entrenar las repeticiones."))
    else:
        _bloques_07.append(
            mo.accordion(
                {
                    "Metricas por hablante": mo.ui.table(_speaker_df_07.round(4)),
                    "Avisos de clases por hablante": mo.ui.table(
                        _speaker_df_07[
                            ["protocol", "split_seed", "speaker_id", "corpus", "n_audios", "n_clases_presentes", "aviso_f1", "confusion_principal"]
                        ]
                    ),
                }
            )
        )
    mo.vstack(_bloques_07)
    return


@app.cell(column=9)
def _():
    #
    return


@app.cell(column=10)
def _(RESULTS_ROOT, leer_csv_07, mo):
    _comparison_07 = leer_csv_07(RESULTS_ROOT / "comparacion_protocolos.csv")
    if _comparison_07.empty:
        _texto_07 = "Pendiente de entrenar y generar resumenes."
    else:
        _global_07 = _comparison_07[_comparison_07["corpus"].eq("GLOBAL")].iloc[0]
        _worst_07 = _comparison_07.sort_values("delta_f1").iloc[0]
        _texto_07 = (
            f"El F1 global cambia {_global_07.delta_f1 * 100:.1f} puntos porcentuales al exigir hablantes no vistos. "
            f"El corpus con mayor caida es {_worst_07.corpus}. "
            "Con tres repeticiones se describe variabilidad, pero no se afirma significacion estadistica."
        )
    mo.vstack([mo.md("## Interpretacion breve"), mo.md(_texto_07)])
    return


if __name__ == "__main__":
    app.run()
