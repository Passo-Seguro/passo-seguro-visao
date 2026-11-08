"""Prepara as fotos da caminhada: orientação, tamanho, LGPD (rostos, placas e GPS).

Organize as fotos por TRECHO da caminhada, uma pasta por trecho (é o "local" da divisão
70/20/10, para fotos do mesmo lugar não caírem em treino e teste ao mesmo tempo):

    dados/brutos/proprias/
        paulista-trianon/   IMG_0001.jpg ...
        consolacao-mackenzie/ ...
        lins-fiap/ ...

O script:
1. aplica a rotação do EXIF e reduz o lado maior para 1600 px;
2. borra rostos (Haar frontal e de perfil do OpenCV) e placas (Haar de placas do OpenCV);
3. grava um JPEG NOVO sem nenhum metadado (sem GPS, sem modelo do celular);
4. escreve dados/intermediario/proprias/manifesto.csv com o que foi borrado.

Os detectores Haar erram: depois de rodar, confira as fotos com pessoas e carros perto e use
`python scripts/borrar_manual.py <foto>` para borrar o que escapou. A verificação final antes
de publicar é `python scripts/checar_lgpd.py`.

    python scripts/02_preparar_fotos_proprias.py
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from _comum import BRUTOS, INTERMEDIARIO, titulo
from visao.imagens import abrir_rgb, listar_imagens, salvar_sem_metadados

ENTRADA = BRUTOS / "proprias"
SAIDA = INTERMEDIARIO / "proprias"


def carregar_cascatas():
    base = Path(cv2.data.haarcascades)
    nomes = {
        "rosto": ["haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"],
        "placa": ["haarcascade_russian_plate_number.xml", "haarcascade_license_plate_rus_16stages.xml"],
    }
    cascatas = {}
    for tipo, arquivos in nomes.items():
        cascatas[tipo] = [cv2.CascadeClassifier(str(base / a)) for a in arquivos if (base / a).exists()]
    return cascatas


def borrar(rgb: np.ndarray, caixas) -> np.ndarray:
    out = rgb.copy()
    for x, y, w, h in caixas:
        m = int(0.15 * max(w, h))  # margem para não sobrar borda do rosto/placa
        x0, y0 = max(x - m, 0), max(y - m, 0)
        x1, y1 = min(x + w + m, out.shape[1]), min(y + h + m, out.shape[0])
        regiao = out[y0:y1, x0:x1]
        if regiao.size == 0:
            continue
        k = max(31, (max(x1 - x0, y1 - y0) // 3) | 1)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(regiao, (k, k), 0)
    return out


def detectar(cinza, cascatas, min_lado: int):
    achados = []
    for c in cascatas:
        r = c.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(min_lado, min_lado))
        achados.extend([tuple(int(v) for v in b) for b in r])
    return achados


def reduzir(rgb: np.ndarray, lado_max: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    escala = lado_max / max(h, w)
    if escala >= 1:
        return rgb
    return cv2.resize(rgb, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lado-max", type=int, default=1600)
    args = ap.parse_args()

    fotos = listar_imagens(ENTRADA)
    if not fotos:
        print(f"Nenhuma foto em {ENTRADA}. Crie uma subpasta por trecho e copie as fotos.")
        return 1
    soltas = [f for f in fotos if f.parent == ENTRADA]
    if soltas:
        print(f"ERRO: {len(soltas)} fotos estão direto em {ENTRADA}. Coloque cada uma na pasta do trecho.")
        return 1

    cascatas = carregar_cascatas()
    SAIDA.mkdir(parents=True, exist_ok=True)
    registros = []
    titulo(f"{len(fotos)} fotos")
    for i, foto in enumerate(fotos, 1):
        local = foto.relative_to(ENTRADA).parts[0]
        rgb = reduzir(abrir_rgb(foto), args.lado_max)
        cinza = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        min_lado = max(20, min(rgb.shape[:2]) // 60)
        rostos = detectar(cinza, cascatas["rosto"], min_lado)
        # o Haar de perfil só pega um lado do rosto: roda de novo na imagem espelhada
        espelho = cv2.flip(cinza, 1)
        for x, y, w, h in detectar(espelho, cascatas["rosto"][1:], min_lado):
            rostos.append((rgb.shape[1] - x - w, y, w, h))
        placas = detectar(cinza, cascatas["placa"], max(12, min_lado // 2))
        rgb = borrar(rgb, rostos + placas)
        nome = f"{local}__{foto.stem}.jpg"
        salvar_sem_metadados(rgb, SAIDA / local / nome)
        registros.append({"arquivo": nome, "local": local, "origem": foto.name,
                          "rostos_borrados": len(rostos), "placas_borradas": len(placas)})
        if i % 25 == 0 or i == len(fotos):
            print(f"{i}/{len(fotos)}")

    with open(SAIDA / "manifesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(registros[0]))
        w.writeheader()
        w.writerows(registros)
    locais = sorted({r["local"] for r in registros})
    print(f"\n{len(registros)} fotos em {len(locais)} trechos: {', '.join(locais)}")
    print(f"Rostos borrados: {sum(r['rostos_borrados'] for r in registros)}, "
          f"placas: {sum(r['placas_borradas'] for r in registros)}")
    if len(locais) < 5:
        print("AVISO: menos de 5 trechos. A divisão por local fica grosseira; tente separar a caminhada em mais trechos.")
    print(f"Saída: {SAIDA}  (próximo passo: 03_pre_anotar.py e anotação no Roboflow/CVAT)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
