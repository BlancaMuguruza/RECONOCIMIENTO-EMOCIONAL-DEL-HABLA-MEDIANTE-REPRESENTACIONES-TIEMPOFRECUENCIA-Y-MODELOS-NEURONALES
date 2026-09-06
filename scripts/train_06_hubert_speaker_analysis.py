from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hubert_speaker_analysis import (
    PROTOCOLS,
    SPLIT_SEEDS,
    entrenar_repeticion,
    generar_resumenes_06,
    preparar_todos_los_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimento 06: HuBERT en RAVDESS+SAVEE con particion por audios y por hablantes."
    )
    parser.add_argument("--prepare-only", action="store_true", help="Solo crea metadatos, splits y comprobaciones.")
    parser.add_argument("--summaries", action="store_true", help="Genera los CSV resumen a partir de metricas ya entrenadas.")
    parser.add_argument("--all", action="store_true", help="Entrena los dos protocolos y las tres repeticiones.")
    parser.add_argument("--protocol", choices=PROTOCOLS, help="Protocolo concreto que se quiere entrenar.")
    parser.add_argument("--split-seed", type=int, choices=SPLIT_SEEDS, help="Repeticion de particion que se quiere entrenar.")
    args = parser.parse_args()

    if args.prepare_only:
        preparar_todos_los_splits()
        print("Preparacion del experimento 06 terminada.")
        return

    if args.summaries:
        rutas = generar_resumenes_06()
        print("Resumenes del experimento 06 generados:")
        for nombre, path in rutas.items():
            print(f"- {nombre}: {path}")
        return

    preparar_todos_los_splits()

    if args.all:
        for protocol in PROTOCOLS:
            for split_seed in SPLIT_SEEDS:
                entrenar_repeticion(protocol, split_seed)
        generar_resumenes_06()
        return

    if args.protocol is None or args.split_seed is None:
        raise SystemExit("Indica --prepare-only, --summaries, --all o bien --protocol y --split-seed.")

    entrenar_repeticion(args.protocol, args.split_seed)


if __name__ == "__main__":
    main()
