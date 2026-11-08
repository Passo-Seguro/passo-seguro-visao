"""Leitura de imagens com a orientação certa (EXIF) e gravação SEM metadados (LGPD: sem GPS)."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_imagens(pasta: Path) -> list[Path]:
    return sorted(p for p in Path(pasta).rglob("*") if p.suffix.lower() in EXTENSOES)


def abrir_rgb(origem) -> np.ndarray:
    """Abre de caminho ou bytes, aplica a rotação do EXIF e devolve RGB uint8."""
    if isinstance(origem, (bytes, bytearray)):
        origem = BytesIO(origem)
    with Image.open(origem) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"))


def salvar_sem_metadados(rgb: np.ndarray, destino: Path, qualidade: int = 92) -> None:
    """Grava JPEG novo a partir dos pixels: nenhum EXIF (GPS, aparelho, data) é copiado."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(destino, "JPEG", quality=qualidade, optimize=True)


def tem_gps(caminho: Path) -> bool:
    """True se a imagem ainda tiver o bloco GPS no EXIF (usado na checagem antes de publicar)."""
    try:
        with Image.open(caminho) as img:
            exif = img.getexif()
            return bool(exif.get_ifd(0x8825))  # GPSInfo
    except Exception:
        return False
