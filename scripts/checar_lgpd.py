"""Checagem antes de publicar o dataset: nenhuma foto própria pode ter GPS no EXIF.

    python scripts/checar_lgpd.py [pasta]   (padrão: dados/dataset e dados/intermediario/proprias)

Sai com código 1 se achar alguma foto com GPS. Rostos e placas são conferidos a olho
(ver PASSO_A_PASSO.md, passo 3).
"""

import sys
from pathlib import Path

from _comum import DATASET, INTERMEDIARIO
from visao.imagens import listar_imagens, tem_gps


def main() -> int:
    pastas = [Path(p) for p in sys.argv[1:]] or [DATASET, INTERMEDIARIO / "proprias"]
    com_gps = []
    total = 0
    for pasta in pastas:
        for img in listar_imagens(pasta):
            total += 1
            if tem_gps(img):
                com_gps.append(img)
    for img in com_gps[:20]:
        print(f"COM GPS: {img}")
    print(f"{total} imagens conferidas, {len(com_gps)} com GPS.")
    return 1 if com_gps else 0


if __name__ == "__main__":
    sys.exit(main())
