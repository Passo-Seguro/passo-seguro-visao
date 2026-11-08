"""Utilidades comuns aos scripts (caminhos, dispositivo, log simples)."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

DADOS = RAIZ / "dados"
BRUTOS = DADOS / "brutos"
INTERMEDIARIO = DADOS / "intermediario"
ANOTADO = DADOS / "anotado"
DATASET = DADOS / "dataset"
MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"
EXECUCOES = RAIZ / "execucoes"  # saída bruta do Ultralytics (pesada, fora do git)


def dispositivo_auto() -> str:
    """'0' (primeira GPU CUDA), 'mps' (Mac) ou 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def titulo(texto: str) -> None:
    print(f"\n=== {texto} ===", flush=True)
