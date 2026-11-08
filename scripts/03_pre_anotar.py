"""Pré-anotação automática das fotos próprias, para acelerar o trabalho no Roboflow ou CVAT.

O YOLO pré-treinado no COCO já conhece carro, moto, ônibus, caminhão e bicicleta. Este script
marca essas caixas como `veiculo_na_calcada` (id 3). ATENÇÃO: ele marca TODO veículo, inclusive
os parados corretamente na rua. Na revisão, apague as caixas de veículos que não estão sobre a
calçada e desenhe à mão buraco, piso_danificado e obstaculo (o COCO não tem essas classes).

Saída em dados/pre_anotado/:
    roboflow/  images/ labels/ data.yaml   -> Roboflow: Upload > arraste a pasta inteira
    cvat_yolo_1.1.zip                      -> CVAT: Actions > Upload annotations > YOLO 1.1

    python scripts/03_pre_anotar.py [--confianca 0.35]
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

from _comum import DADOS, INTERMEDIARIO, PESOS_COCO, dispositivo_auto, titulo
from visao.classes import carregar_classes
from visao.imagens import listar_imagens

COCO_PARA_VEICULO = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confianca", type=float, default=0.35)
    ap.add_argument("--modelo", default=str(PESOS_COCO / "yolo11n.pt"), help="baixado automaticamente na 1ª vez")
    args = ap.parse_args()

    from ultralytics import YOLO

    classes = carregar_classes()
    id_veiculo = classes.index("veiculo_na_calcada")
    fotos = listar_imagens(INTERMEDIARIO / "proprias")
    if not fotos:
        print("Rode antes o 02_preparar_fotos_proprias.py.")
        return 1

    saida = DADOS / "pre_anotado"
    if saida.exists():
        shutil.rmtree(saida)
    rf = saida / "roboflow"
    (rf / "images").mkdir(parents=True)
    (rf / "labels").mkdir(parents=True)

    PESOS_COCO.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(args.modelo)
    titulo(f"Pré-anotando {len(fotos)} fotos")
    total = 0
    for i in range(0, len(fotos), 16):
        lote = fotos[i : i + 16]
        resultados = modelo.predict([str(p) for p in lote], conf=args.confianca,
                                    classes=list(COCO_PARA_VEICULO), device=dispositivo_auto(), verbose=False)
        for foto, r in zip(lote, resultados):
            linhas = [f"{id_veiculo} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in r.boxes.xywhn.tolist()]
            total += len(linhas)
            shutil.copy2(foto, rf / "images" / foto.name)
            (rf / "labels" / f"{foto.stem}.txt").write_text("\n".join(linhas), encoding="utf-8")
    (rf / "data.yaml").write_text(
        yaml.safe_dump({"names": classes, "nc": len(classes), "train": "images", "val": "images"},
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # CVAT, formato YOLO 1.1: obj.names, obj.data, train.txt e obj_train_data/
    zip_cvat = saida / "cvat_yolo_1.1.zip"
    with zipfile.ZipFile(zip_cvat, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(classes) + "\n")
        z.writestr("obj.data", f"classes = {len(classes)}\nnames = data/obj.names\ntrain = data/train.txt\n")
        nomes = []
        for rotulo in sorted((rf / "labels").glob("*.txt")):
            z.write(rotulo, f"obj_train_data/{rotulo.name}")
            img = next((rf / "images").glob(rotulo.stem + ".*"))
            nomes.append(f"data/obj_train_data/{img.name}")
        z.writestr("train.txt", "\n".join(nomes) + "\n")

    print(f"{total} caixas de veículo sugeridas.")
    print(f"Roboflow: suba a pasta {rf}")
    print(f"CVAT: crie a tarefa com as imagens de {rf / 'images'} e importe {zip_cvat}")
    print("Depois de anotar, exporte em formato YOLOv8 e descompacte em dados/anotado/proprias/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
