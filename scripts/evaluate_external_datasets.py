from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.external_datasets import crear_metadata_externa
from src.external_evaluation import (
    EXTERNAL_CORPUS,
    MODEL_SPECS,
    SEEDS,
    assert_checkpoints_validos,
    comprobar_checkpoints,
    evaluar_combinacion,
    generar_resumen_externo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion externa de modelos entrenados sobre CREMA-D y EMO-DB.")
    parser.add_argument("--check-only", action="store_true", help="Comprueba metadatos y checkpoints sin inferencia.")
    parser.add_argument("--summaries", action="store_true", help="Genera resumenes desde resultados ya guardados.")
    parser.add_argument("--all", action="store_true", help="Evalua las 30 combinaciones completas.")
    parser.add_argument("--model", choices=list(MODEL_SPECS), help="Modelo concreto.")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Semilla concreta.")
    parser.add_argument("--dataset", choices=EXTERNAL_CORPUS, help="Corpus externo concreto.")
    args = parser.parse_args()

    crear_metadata_externa()
    assert_checkpoints_validos()

    if args.check_only:
        print("Checkpoints encontrados:")
        for row in comprobar_checkpoints():
            print(row)
        print("Comprobaciones externas terminadas.")
        return

    if args.summaries:
        rutas = generar_resumen_externo()
        print("Resumenes externos generados:")
        for name, path in rutas.items():
            print(f"- {name}: {path}")
        return

    if args.all:
        for model_id in MODEL_SPECS:
            for seed in SEEDS:
                for dataset in EXTERNAL_CORPUS:
                    print(f"Evaluando {model_id} seed {seed} en {dataset}")
                    evaluar_combinacion(model_id, seed, dataset)
        generar_resumen_externo()
        return

    if args.model is None or args.seed is None or args.dataset is None:
        raise SystemExit("Indica --check-only, --summaries, --all o una combinacion con --model, --seed y --dataset.")

    resultado = evaluar_combinacion(args.model, args.seed, args.dataset)
    print(f"Evaluacion guardada en: {resultado['output_dir']}")


if __name__ == "__main__":
    main()
