import marimo

__generated_with = "0.23.11"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return mo, pd


@app.cell
def _():
    from src import config
    from src.datos import cargar_o_crear_metadata, tabla_conteos
    from src.graficas import save_countplot
    from src.modelos import MLPHuBERT, crear_cnn_mel, crear_cnn_wavelet
    from src.notebook_utils import stats_cards
    from src.paths import RESULTS_DIR, experiment_dir
    from src.utilidades import contar_parametros

    return (
        MLPHuBERT,
        RESULTS_DIR,
        cargar_o_crear_metadata,
        config,
        contar_parametros,
        crear_cnn_mel,
        crear_cnn_wavelet,
        experiment_dir,
        save_countplot,
        stats_cards,
        tabla_conteos,
    )


@app.cell
def _(config, contar_parametros, mo, pd):
    def bloque_imagen(path, titulo):
        if not path.exists():
            return mo.md(f"#### {titulo}\n\nPendiente de generar.")
        return mo.vstack([mo.md(f"#### {titulo}"), mo.image(str(path))])

    def seed_dirs_ordenados(output_dir):
        return sorted(output_dir.glob("seed_*"), key=lambda path: int(path.name.replace("seed_", "")))

    def metricas_globales(output_dir):
        filas = []
        for seed_dir in seed_dirs_ordenados(output_dir):
            seed = seed_dir.name.replace("seed_", "")
            metrics_path = seed_dir / f"metricas_seed_{seed}.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
                filas.append(metrics[metrics["group"].eq("test_global")])
        if not filas:
            return pd.DataFrame()
        metrics = pd.concat(filas, ignore_index=True)
        cols = ["seed", "n_datos", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
        return metrics[cols]

    def metricas_por_dataset(output_dir):
        filas = []
        for seed_dir in seed_dirs_ordenados(output_dir):
            seed = seed_dir.name.replace("seed_", "")
            metrics_path = seed_dir / f"metricas_seed_{seed}.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
                dataset_metrics = metrics[
                    metrics["group"].str.startswith("test_")
                    & ~metrics["group"].eq("test_global")
                ].copy()
                dataset_metrics["dataset"] = dataset_metrics["group"].str.replace("test_", "", regex=False)
                filas.append(dataset_metrics)
        if not filas:
            return pd.DataFrame()
        metrics = pd.concat(filas, ignore_index=True)
        cols = ["seed", "dataset", "n_datos", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
        return metrics[cols]

    def media_dataset(metrics_dataset):
        if metrics_dataset.empty:
            return pd.DataFrame()
        orden = ["RAVDESS", "SAVEE", "TESS"]
        media = (
            metrics_dataset
            .groupby("dataset", as_index=False)
            .agg(
                n_datos=("n_datos", "mean"),
                accuracy=("accuracy", "mean"),
                precision_macro=("precision_macro", "mean"),
                recall_macro=("recall_macro", "mean"),
                f1_macro=("f1_macro", "mean"),
            )
        )
        media["orden"] = media["dataset"].map({name: idx for idx, name in enumerate(orden)})
        media = media.sort_values(["orden", "dataset"]).drop(columns=["orden"])
        media["n_datos"] = media["n_datos"].round().astype(int)
        return media

    def _numero(value):
        if isinstance(value, float):
            return f"{value:.2f}".replace(".", ",")
        return f"{int(value):,}".replace(",", ".")

    def _porcentaje(value):
        return f"{float(value) * 100:.1f}%".replace(".", ",")

    def _obs_entrada(n_train, representation):
        if representation == "mel":
            return "Frames train", n_train * (1 + config.N_SAMPLES // config.HOP_LENGTH)
        if representation == "wavelet":
            return "Posiciones train", n_train * config.WAVELET_TIME_BINS
        if representation == "hubert_sequence":
            return "Embeddings train", n_train * config.HUBERT_FRAMES_PER_AUDIO
        return "Entradas train", n_train

    def tarjetas_modelo(df, model_preview, representation):
        n_train = max(int((df["split"] == "train").sum()), 1)
        params = contar_parametros(model_preview)["parametros_entrenables"]
        obs_label, obs_value = _obs_entrada(n_train, representation)
        valores = {
            "Parametros": _numero(params),
            obs_label: _numero(obs_value),
            "Obs./param": f"{obs_value / max(params, 1):.2f}x".replace(".", ","),
        }
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
            <div style="display:grid; grid-template-columns:repeat(3,minmax(84px,1fr)); gap:8px;">
            {cards}
            </div>
            """
        )

    def guardar_tabla_dataset(media, path, titulo):
        if media.empty:
            return
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        tabla = media.copy()
        tabla = tabla.rename(
            columns={
                "dataset": "Base de datos",
                "n_datos": "Test",
                "accuracy": "Accuracy",
                "precision_macro": "Precision",
                "recall_macro": "Recall",
                "f1_macro": "F1 macro",
            }
        )
        for col in ["Accuracy", "Precision", "Recall", "F1 macro"]:
            tabla[col] = tabla[col].map(_porcentaje)

        fig, ax = plt.subplots(figsize=(8.8, 1.6 + 0.42 * len(tabla)))
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.set_title(titulo, fontsize=13, fontweight="bold", color="#1f2937", pad=12)
        table = ax.table(
            cellText=tabla.values,
            colLabels=tabla.columns,
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.45)
        dataset_colors = {"RAVDESS": "#9ED2F0", "SAVEE": "#A8E6BD", "TESS": "#F7A8C4"}
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d8dee9")
            cell.set_linewidth(0.7)
            if row == 0:
                cell.set_facecolor("#eef4fb")
                cell.set_text_props(weight="bold", color="#1f2937")
            else:
                cell.set_facecolor("#ffffff" if row % 2 else "#fbfcfe")
                if col == 0:
                    dataset = str(tabla.iloc[row - 1, 0])
                    cell.set_facecolor(dataset_colors.get(dataset, "#fbfcfe"))
                    cell.set_text_props(weight="bold", color="#1f2937")
        fig.tight_layout()
        fig.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    def tabla_dataset_bonita(media, output_dir, titulo):
        if media.empty:
            return mo.md("Comparacion por base de datos pendiente.")
        image_path = output_dir / "tabla_comparacion_datasets.png"
        guardar_tabla_dataset(media, image_path, titulo)
        filas = []
        swatches = {"RAVDESS": "#9ED2F0", "SAVEE": "#A8E6BD", "TESS": "#F7A8C4"}
        for row in media.itertuples(index=False):
            color = swatches.get(row.dataset, "#d8dee9")
            filas.append(
                f"""
                <tr>
                  <td><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{color}; margin-right:7px;"></span><strong>{row.dataset}</strong></td>
                  <td>{int(row.n_datos)}</td>
                  <td>{_porcentaje(row.accuracy)}</td>
                  <td>{_porcentaje(row.precision_macro)}</td>
                  <td>{_porcentaje(row.recall_macro)}</td>
                  <td><strong>{_porcentaje(row.f1_macro)}</strong></td>
                </tr>
                """
            )
        return mo.vstack(
            [
                mo.Html(
                    f"""
                    <style>
                      .tabla-dataset-{output_dir.name} td {{
                        padding:9px 10px;
                        border-top:1px solid #edf1f7;
                        text-align:center;
                        color:#334155;
                      }}
                      .tabla-dataset-{output_dir.name} td:first-child {{
                        text-align:left;
                      }}
                      .tabla-dataset-{output_dir.name} tbody tr:nth-child(even) {{
                        background:#fbfcfe;
                      }}
                    </style>
                    <div style="border:1px solid #d8dee9; border-radius:8px; overflow:hidden; background:white;">
                      <table class="tabla-dataset-{output_dir.name}" style="width:100%; border-collapse:collapse; font-size:13px;">
                        <thead>
                          <tr style="background:#eef4fb; color:#1f2937;">
                            <th style="text-align:left; padding:9px 10px;">Base de datos</th>
                            <th style="padding:9px 10px;">Test</th>
                            <th style="padding:9px 10px;">Accuracy</th>
                            <th style="padding:9px 10px;">Precision</th>
                            <th style="padding:9px 10px;">Recall</th>
                            <th style="padding:9px 10px;">F1 macro</th>
                          </tr>
                        </thead>
                        <tbody>{''.join(filas)}</tbody>
                      </table>
                    </div>
                    """
                ),
                mo.Html(
                    f"""
                    <a href="{image_path.as_uri()}" download style="font-size:12px; color:#4a6fa5; text-decoration:none;">
                      Descargar tabla como PNG
                    </a>
                    """
                ),
            ]
        )

    def _formato_media_std(media, std):
        return f"{float(media) * 100:.1f} ± {float(std) * 100:.1f}%".replace(".", ",")

    def bloque_resumen_controlado(results_dir, bloque_imagen):
        resumen_path = results_dir / "resumen_metricas_modelo_corpus.csv"
        descriptivo_path = results_dir / "resumen_f1_descriptivo.csv"
        fig_dir = results_dir / "figuras_resumen"
        if not resumen_path.exists() or not descriptivo_path.exists():
            return mo.vstack(
                [
                    mo.md("## Resumen final"),
                    mo.md(
                        "Pendiente de generar. Reentrena los tres modelos Mel controlados y ejecuta "
                        "`python scripts/generate_result_summaries.py`."
                    ),
                ]
            )

        resumen = pd.read_csv(resumen_path)
        vista = resumen.copy()
        vista["Accuracy"] = vista.apply(lambda row: _formato_media_std(row.accuracy_mean, row.accuracy_std), axis=1)
        vista["Precision"] = vista.apply(
            lambda row: _formato_media_std(row.precision_macro_mean, row.precision_macro_std),
            axis=1,
        )
        vista["Recall"] = vista.apply(lambda row: _formato_media_std(row.recall_macro_mean, row.recall_macro_std), axis=1)
        vista["F1 macro"] = vista.apply(lambda row: _formato_media_std(row.f1_macro_mean, row.f1_macro_std), axis=1)
        vista = vista[["model", "corpus", "Accuracy", "Precision", "Recall", "F1 macro", "n_seeds"]]

        descriptivo = pd.read_csv(descriptivo_path)
        return mo.vstack(
            [
                mo.md("## Resumen final"),
                mo.md("Media ± desviacion tipica muestral calculada con `ddof=1` sobre las tres semillas."),
                bloque_imagen(fig_dir / "f1_modelo_corpus_std.png", "F1 por modelo y corpus"),
                bloque_imagen(fig_dir / "delta_f1_mel_controlado.png", "Efecto de los cambios Mel"),
                mo.accordion(
                    {
                        "Metricas por modelo y corpus": mo.ui.table(vista),
                        "Analisis descriptivo de F1": mo.ui.table(descriptivo),
                    }
                ),
            ]
        )

    def tarjetas_metricas(metrics):
        if metrics.empty:
            return mo.md("Resultados pendientes.")
        resumen = metrics.drop(columns=["seed"]).mean(numeric_only=True)
        valores = {
            "Accuracy": f"{float(resumen.get('accuracy', 0)) * 100:.1f}%",
            "F1 macro": f"{float(resumen.get('f1_macro', 0)) * 100:.1f}%",
        }
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
            <div style="display:grid; grid-template-columns:repeat(2,minmax(88px,1fr)); gap:8px;">
            {cards}
            </div>
            """
        )

    def resultados_mel_controlados(output_dir):
        if output_dir.name not in config.MEL_CONTROLLED_EXPERIMENTS:
            return True
        import json

        for seed in config.SEEDS:
            stats_path = output_dir / f"seed_{seed}" / "stats.json"
            if not stats_path.exists():
                return False
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if stats.get("run_config", {}).get("version") != config.MEL_CONTROLLED_VERSION:
                return False
        return True

    def bloque_modelo(titulo, output_dir, df, model_preview, representation):
        metrics = metricas_globales(output_dir)
        dataset_metrics = metricas_por_dataset(output_dir)
        bloques = [
            mo.md(f"## {titulo}"),
            tarjetas_modelo(df, model_preview, representation),
        ]

        if not resultados_mel_controlados(output_dir):
            bloques.append(
                mo.md(
                    "Resultados Mel antiguos detectados. Estan conservados en "
                    "`resultados_originales_mel_configuracion_no_controlada/`. "
                    "Reentrena este modelo para ver la comparacion controlada."
                )
            )
            return mo.vstack(bloques)

        bloques.append(tarjetas_metricas(metrics))

        if metrics.empty:
            bloques.append(mo.md("Entrena este modelo desde `scripts/` para ver aqui sus resultados."))
            return mo.vstack(bloques)

        media = metrics.drop(columns=["seed"]).mean(numeric_only=True).to_frame("media").T
        media_datasets = media_dataset(dataset_metrics)
        bloques.append(
            mo.accordion(
                {
                    "Metricas por semilla": mo.ui.table(metrics.round(4)),
                    "Media de semillas": mo.ui.table(media.round(4)),
                    "Dataset por semilla": mo.ui.table(dataset_metrics.round(4)),
                }
            )
        )

        curvas = []
        matrices = []
        for seed in metrics["seed"].tolist():
            seed_dir = output_dir / f"seed_{seed}"
            curvas.append(bloque_imagen(seed_dir / "loss_curve.png", f"Curva seed {seed}"))
            matrices.append(
                mo.vstack(
                    [
                        mo.md(f"#### Matrices seed {seed}"),
                        mo.hstack(
                            [
                                bloque_imagen(seed_dir / f"confusion_matrix_norm_seed_{seed}.png", "Normalizada"),
                                bloque_imagen(seed_dir / f"confusion_matrix_seed_{seed}.png", "Absoluta"),
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
                mo.md("### Comparacion por base de datos"),
                tabla_dataset_bonita(media_datasets, output_dir, f"{titulo}: comparacion por base de datos"),
            ]
        )
        return mo.vstack(bloques)

    return bloque_imagen, bloque_modelo, bloque_resumen_controlado


@app.cell
def _(
    RESULTS_DIR,
    bloque_imagen,
    cargar_o_crear_metadata,
    mo,
    save_countplot,
    stats_cards,
    tabla_conteos,
):
    _df = cargar_o_crear_metadata()
    _eda_dir = RESULTS_DIR / "eda"
    _eda_dir.mkdir(parents=True, exist_ok=True)

    save_countplot(_df, "dataset", _eda_dir / "conteo_dataset.png", "Datos por dataset")
    save_countplot(_df, "emotion", _eda_dir / "conteo_emocion.png", "Datos por emocion")
    save_countplot(_df, "split", _eda_dir / "conteo_split.png", "Datos por split", hue="dataset")

    mo.vstack(
        [
            mo.md("## EDA"),
            stats_cards(_df),
            bloque_imagen(_eda_dir / "conteo_dataset.png", "Datos por dataset"),
            bloque_imagen(_eda_dir / "conteo_emocion.png", "Datos por emocion"),
            bloque_imagen(_eda_dir / "conteo_split.png", "Datos por split"),
            mo.accordion(
                {
                    "Conteo por dataset y emocion": mo.ui.table(tabla_conteos(_df, "dataset", "emotion")),
                    "Conteo por split y dataset": mo.ui.table(tabla_conteos(_df, "split", "dataset")),
                    "Conteo por split y emocion": mo.ui.table(tabla_conteos(_df, "split", "emotion")),
                }
            ),
        ]
    )
    return


@app.cell(column=1)
def _(
    MLPHuBERT,
    RESULTS_DIR,
    bloque_imagen,
    bloque_resumen_controlado,
    cargar_o_crear_metadata,
    config,
    contar_parametros,
    crear_cnn_mel,
    crear_cnn_wavelet,
    mo,
):
    _df = cargar_o_crear_metadata()
    _n_train = int((_df["split"] == "train").sum())
    _n_valid = int((_df["split"] == "valid").sum())
    _n_test = int((_df["split"] == "test").sum())
    _mel_model = crear_cnn_mel(num_classes=len(config.EMOTIONS), dropout=config.MEL_CONTROLLED_DROPOUT)
    _wavelet_model = crear_cnn_wavelet(num_classes=len(config.EMOTIONS), dropout=0.50)
    _hubert_model = MLPHuBERT(input_dim=768, num_classes=len(config.EMOTIONS), dropout=0.45)

    _modelos = [
        (
            "Modelo 1",
            "CNN Mel baseline",
            "Mel-espectrogramas. 4 bloques convolucionales con canales 8, 16, 32 y 32, pooling adaptativo 2x2 y capa densa de 32 neuronas.",
            contar_parametros(_mel_model)["parametros_entrenables"],
        ),
        (
            "Modelo 2",
            "CNN Mel con aumento",
            "Misma CNN que el baseline, pero entrenada con ruido, desplazamiento temporal y cambios suaves de ganancia.",
            contar_parametros(_mel_model)["parametros_entrenables"],
        ),
        (
            "Modelo 3",
            "CNN Mel con scheduler",
            "Misma CNN Mel, sin aumento de datos, con reduccion automatica del learning rate cuando la validacion se estanca.",
            contar_parametros(_mel_model)["parametros_entrenables"],
        ),
        (
            "Modelo 4",
            "CNN Wavelet",
            "Escalogramas wavelet. 4 bloques convolucionales con canales 16, 32, 64 y 64, kernels 5x3 y capa densa de 64 neuronas.",
            contar_parametros(_wavelet_model)["parametros_entrenables"],
        ),
        (
            "Modelo 5",
            "HuBERT congelado",
            "Embeddings HuBERT cada 20 ms. El clasificador promedia la secuencia y usa una MLP de 64 neuronas.",
            contar_parametros(_hubert_model)["parametros_entrenables"],
        ),
    ]

    _cards = "\n".join(
        f"""
        <div style="border:1px solid #d8dee9; border-radius:8px; padding:11px 12px; background:#fbfcfe;">
          <div style="font-size:12px; color:#5f6b7a;">{numero}</div>
          <div style="font-size:15px; font-weight:700; color:#1f2937; margin:2px 0 6px;">{nombre}</div>
          <div style="font-size:12px; line-height:1.45; color:#334155;">{texto}</div>
          <div style="font-size:12px; color:#5f6b7a; margin-top:7px;">Parametros: <strong>{format(params, ",").replace(",", ".")}</strong></div>
        </div>
        """
        for numero, nombre, texto, params in _modelos
    )

    mo.vstack(
        [
            mo.md("## Modelos"),
            mo.md(
                "Todos los modelos se entrenan con el mismo particionado aleatorio estratificado por emocion. "
                "La comparacion principal se hace sobre test y despues se revisa por base de datos."
            ),
            mo.Html(
                f"""
                <div style="border:1px solid #d8dee9; border-radius:8px; padding:12px 14px; background:white; margin-bottom:10px;">
                  <div style="font-weight:700; color:#1f2937; margin-bottom:8px;">Configuracion comun</div>
                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:13px; color:#334155;">
                    <div>Train: <strong>{_n_train}</strong></div>
                    <div>Valid: <strong>{_n_valid}</strong></div>
                    <div>Test: <strong>{_n_test}</strong></div>
                    <div>Seeds: <strong>{len(config.SEEDS)}</strong></div>
                    <div>Epochs max: <strong>{config.DEFAULT_TRAINING.epochs}</strong></div>
                    <div>Batch Mel/HuBERT: <strong>{config.DEFAULT_TRAINING.batch_size}</strong></div>
                    <div>Early stopping: <strong>{config.DEFAULT_TRAINING.early_stopping_patience}</strong></div>
                    <div>Batch Wavelet: <strong>8</strong></div>
                  </div>
                </div>
                """
            ),
            mo.Html(f"<div style='display:grid; gap:9px;'>{_cards}</div>"),
            bloque_resumen_controlado(RESULTS_DIR, bloque_imagen),
        ]
    )
    return


@app.cell(column=2)
def _(
    bloque_modelo,
    cargar_o_crear_metadata,
    config,
    crear_cnn_mel,
    experiment_dir,
):
    _df = cargar_o_crear_metadata()
    _model = crear_cnn_mel(num_classes=len(config.EMOTIONS), dropout=config.MEL_CONTROLLED_DROPOUT)
    bloque_modelo(
        "Modelo 1: CNN baseline",
        experiment_dir("01_cnn_baseline"),
        _df,
        _model,
        "mel",
    )
    return


@app.cell(column=3)
def _():
    return


@app.cell(column=4)
def _(
    bloque_modelo,
    cargar_o_crear_metadata,
    config,
    crear_cnn_mel,
    experiment_dir,
):
    _df = cargar_o_crear_metadata()
    _model = crear_cnn_mel(num_classes=len(config.EMOTIONS), dropout=config.MEL_CONTROLLED_DROPOUT)
    bloque_modelo(
        "Modelo 2: CNN augmentation",
        experiment_dir("02_cnn_data_augmentation"),
        _df,
        _model,
        "mel",
    )
    return


@app.cell(column=5)
def _():
    return


@app.cell(column=6)
def _(
    bloque_modelo,
    cargar_o_crear_metadata,
    config,
    crear_cnn_mel,
    experiment_dir,
):
    _df = cargar_o_crear_metadata()
    _model = crear_cnn_mel(num_classes=len(config.EMOTIONS), dropout=config.MEL_CONTROLLED_DROPOUT)
    bloque_modelo(
        "Modelo 3: CNN scheduler",
        experiment_dir("03_cnn_scheduler"),
        _df,
        _model,
        "mel",
    )
    return


@app.cell(column=7)
def _():
    return


@app.cell(column=8)
def _(
    bloque_modelo,
    cargar_o_crear_metadata,
    config,
    crear_cnn_wavelet,
    experiment_dir,
):
    _df = cargar_o_crear_metadata()
    _model = crear_cnn_wavelet(num_classes=len(config.EMOTIONS), dropout=0.50)
    bloque_modelo(
        "Modelo 4: Wavelet CNN",
        experiment_dir("04_cnn_wavelet"),
        _df,
        _model,
        "wavelet",
    )
    return


@app.cell(column=9)
def _():
    return


@app.cell(column=10)
def _(
    MLPHuBERT,
    bloque_modelo,
    cargar_o_crear_metadata,
    config,
    experiment_dir,
):
    _df = cargar_o_crear_metadata()
    _model = MLPHuBERT(input_dim=768, num_classes=len(config.EMOTIONS), dropout=0.45)
    bloque_modelo(
        "Modelo 5: HuBERT",
        experiment_dir("05_hubert"),
        _df,
        _model,
        "hubert_sequence",
    )
    return


@app.cell(column=11)
def _():
    return


if __name__ == "__main__":
    app.run()
