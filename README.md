# Reconocimiento emocional del habla

Repositorio asociado al Trabajo Fin de Grado **Reconocimiento emocional del habla mediante representaciones tiempo-frecuencia y modelos neuronales**.

El proyecto estudia el reconocimiento emocional del habla a partir de cinco corpus públicos, tres representaciones de audio y varios protocolos de evaluación. El experimento principal utiliza RAVDESS, TESS y SAVEE, y compara cinco modelos:

- CNN con espectrogramas Mel.
- CNN con Mel y aumento de datos.
- CNN con Mel y scheduler.
- CNN con escalogramas Wavelet.
- Clasificador sobre embeddings HuBERT.

También se incluyen dos análisis adicionales: una prueba con separación por hablante para HuBERT y una evaluación externa con CREMA-D y EMO-DB usando modelos ya entrenados.

El objetivo del repositorio es conservar el código y la configuración experimental necesarios para revisar y reproducir el flujo principal del trabajo.

## Estructura

```text
src/                Módulos comunes del proyecto
scripts/            Scripts de preparación, entrenamiento y evaluación
notebooks/          Vistas finales en Marimo
datos_generados/    Metadatos y datos derivados ligeros
```

Las rutas internas se calculan desde la ubicación del proyecto. Los datos crudos se esperan fuera del repositorio, por defecto en una carpeta `datos_crudos` situada al mismo nivel que este proyecto. También se puede indicar otra ruta mediante la variable de entorno `TFG_AUDIO_DATA_DIR`.

## Instalación del entorno

Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

También se incluye `pyproject.toml` con las dependencias principales.

## Preparar datos

La metadata principal se crea la primera vez que un script llama a `cargar_o_crear_metadata()`. El archivo generado es:

```text
datos_generados/metadata_ravdess_tess_savee.csv
```

Para los modelos Wavelet y HuBERT es necesario generar previamente las representaciones:

```powershell
python scripts/generate_wavelet_features.py
python scripts/generate_hubert_embeddings.py
```

Para forzar la regeneración de estas representaciones:

```powershell
python scripts/generate_wavelet_features.py --force
python scripts/generate_hubert_embeddings.py --force
```

## Entrenar los cinco modelos

Los tres modelos basados en espectrogramas Mel se entrenan directamente desde los audios preprocesados:

```powershell
python scripts/train_01_cnn_baseline.py
python scripts/train_02_cnn_data_augmentation.py
python scripts/train_03_cnn_scheduler.py
```

El modelo Wavelet requiere generar antes sus features:

```powershell
python scripts/generate_wavelet_features.py
python scripts/train_04_cnn_wavelet.py
```

El modelo HuBERT requiere generar antes sus embeddings:

```powershell
python scripts/generate_hubert_embeddings.py
python scripts/train_05_hubert.py
```

Para ejecutar el flujo principal completo:

```powershell
python scripts/train_all.py
```

Cada script ejecuta las tres semillas definidas en `src/config.py`. Durante el entrenamiento se guardan métricas, curvas de pérdida, matrices de confusión, predicciones y checkpoints.

Las métricas internas pueden recalcularse desde las predicciones guardadas:

```powershell
python scripts/regenerate_internal_metrics.py
```

## Resúmenes y comprobaciones

Comprobación de la configuración Mel controlada:

```powershell
python scripts/check_mel_controlled_config.py
```

Comprobación de la comparación Mel:

```powershell
python scripts/generate_result_summaries.py --check-only
```

Generación de resúmenes internos:

```powershell
python scripts/generate_result_summaries.py
```

## Separación por hablante

Este análisis utiliza HuBERT sobre RAVDESS y SAVEE. Compara una partición aleatoria por audios con una partición en la que los hablantes del test no aparecen en entrenamiento.

Preparación de particiones:

```powershell
python scripts/train_06_hubert_speaker_analysis.py --prepare-only
```

Ejecución completa:

```powershell
python scripts/train_06_hubert_speaker_analysis.py --all
```

Generación de resúmenes:

```powershell
python scripts/train_06_hubert_speaker_analysis.py --summaries
```

Ejecución de una combinación concreta:

```powershell
python scripts/train_06_hubert_speaker_analysis.py --protocol random_audio --split-seed 42
python scripts/train_06_hubert_speaker_analysis.py --protocol speaker_disjoint --split-seed 42
```

## Evaluación externa

La evaluación externa utiliza CREMA-D y EMO-DB. Requiere disponer de los checkpoints entrenados de los cinco modelos principales.

Preparación de metadata externa:

```powershell
python scripts/prepare_external_metadata.py
```

Comprobación de metadata y checkpoints:

```powershell
python scripts/evaluate_external_datasets.py --check-only
```

Evaluación completa:

```powershell
python scripts/evaluate_external_datasets.py --all
```

Generación de resúmenes:

```powershell
python scripts/evaluate_external_datasets.py --summaries
```

Evaluación de una combinación concreta:

```powershell
python scripts/evaluate_external_datasets.py --model 05_hubert --seed 42 --dataset CREMA-D
```

## Vistas Marimo

Las vistas finales se abren con Marimo:

```powershell
python -m marimo edit notebooks/vista_general_modelos.py
python -m marimo edit notebooks/resultados_separacion_hablante.py
python -m marimo edit notebooks/resultados_evaluacion_externa.py
```

`vista_general_modelos.py` muestra EDA, resumen de modelos, curvas, matrices y resultados de los cinco modelos principales.

`resultados_separacion_hablante.py` muestra el análisis de HuBERT con y sin separación por hablante.

`resultados_evaluacion_externa.py` muestra la evaluación externa con CREMA-D y EMO-DB.
