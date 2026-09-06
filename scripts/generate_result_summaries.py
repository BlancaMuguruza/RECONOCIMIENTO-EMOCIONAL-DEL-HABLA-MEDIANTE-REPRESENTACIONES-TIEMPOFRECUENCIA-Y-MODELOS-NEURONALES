from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.resumen_resultados import generar_resumenes, mel_controlado_disponible


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera tablas y figuras resumen a partir de metricas por semilla.")
    parser.add_argument("--check-only", action="store_true", help="Solo comprueba si los Mel controlados estan listos.")
    args = parser.parse_args()

    ok, problemas = mel_controlado_disponible()
    if args.check_only:
        if ok:
            print("Los resultados Mel controlados estan completos.")
        else:
            print("Los resultados Mel controlados aun no estan completos:")
            for problema in problemas:
                print(f"- {problema}")
        raise SystemExit(0 if ok else 1)

    rutas = generar_resumenes(require_mel_controlado=True)
    print("Resumenes generados:")
    for nombre, path in rutas.items():
        print(f"- {nombre}: {path}")
