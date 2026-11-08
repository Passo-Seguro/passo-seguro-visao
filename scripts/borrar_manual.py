"""Borra à mão regiões que os detectores automáticos deixaram passar (rosto, placa, documento).

    python scripts/borrar_manual.py dados/intermediario/proprias/<trecho>/<foto>.jpg

Na janela: arraste um retângulo e aperte ENTER (ou ESPAÇO) para cada região; ESC termina.
A foto é regravada no lugar, sem metadados.
"""

import sys
from pathlib import Path

import cv2

from _comum import RAIZ  # noqa: F401  (ajusta o sys.path)
from visao.imagens import abrir_rgb, salvar_sem_metadados


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    caminho = Path(sys.argv[1])
    rgb = abrir_rgb(caminho)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    escala = min(1.0, 1200 / max(bgr.shape[:2]))
    tela = cv2.resize(bgr, None, fx=escala, fy=escala)
    regioes = cv2.selectROIs("Selecione e ENTER; ESC termina", tela, showCrosshair=False)
    cv2.destroyAllWindows()
    if len(regioes) == 0:
        print("Nada selecionado.")
        return 0
    for x, y, w, h in regioes:
        x, y, w, h = (int(v / escala) for v in (x, y, w, h))
        k = max(31, (max(w, h) // 3) | 1)
        rgb[y : y + h, x : x + w] = cv2.GaussianBlur(rgb[y : y + h, x : x + w], (k, k), 0)
    salvar_sem_metadados(rgb, caminho)
    print(f"{len(regioes)} região(ões) borrada(s) em {caminho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
