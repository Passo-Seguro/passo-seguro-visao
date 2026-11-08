"""Classes do detector, lidas de config/classes.yaml (cópia do contrato do repositório principal)."""

from functools import lru_cache
from pathlib import Path

import yaml

from visao import RAIZ

ARQUIVO_CLASSES = RAIZ / "config" / "classes.yaml"

# Nomes para mostrar na tela (vídeo e tempo real). O nome técnico continua o do contrato.
ROTULOS = {
    "buraco": "Buraco",
    "piso_danificado": "Piso danificado",
    "obstaculo": "Obstaculo",  # sem acento: o putText do OpenCV não desenha acentos
    "veiculo_na_calcada": "Veiculo na calcada",
}

# Cores BGR fixas por classe, para o vídeo ficar consistente.
CORES = {
    "buraco": (0, 0, 230),
    "piso_danificado": (0, 165, 255),
    "obstaculo": (230, 160, 0),
    "veiculo_na_calcada": (200, 0, 200),
}


@lru_cache(maxsize=1)
def carregar_classes(caminho: Path = ARQUIVO_CLASSES) -> list[str]:
    """Lista de nomes na ordem do id (0 a 3). A posição é o índice da classe no YOLO."""
    dados = yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))
    itens = sorted(dados["classes"], key=lambda c: c["id"])
    ids = [c["id"] for c in itens]
    if ids != list(range(len(itens))):
        raise ValueError(f"ids de classe devem ser 0..n-1 sem buracos; recebi {ids}")
    return [c["nome"] for c in itens]
