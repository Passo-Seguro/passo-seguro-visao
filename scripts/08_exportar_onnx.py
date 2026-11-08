"""Exporta o modelo escolhido para ONNX e grava modelos/detector.onnx + modelos/detector.json.

    python scripts/08_exportar_onnx.py --nome yolo11n_final_640

O serviço `POST /deteccoes` e o script de tempo real leem modelos/detector.onnx. O .json ao lado
guarda as classes (na ordem do contrato), o tamanho de entrada e as métricas de teste, para o
README e o /saude do serviço mostrarem qual modelo está no ar.

Confere também que o ONNX dá o mesmo resultado que o .pt numa imagem de teste.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone

from _comum import DATASET, MODELOS, RESULTADOS, titulo
from visao.classes import carregar_classes
from visao.detector_onnx import DetectorOnnx
from visao.imagens import abrir_rgb, listar_imagens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--destino", default="detector")
    args = ap.parse_args()

    from ultralytics import YOLO

    classes = carregar_classes()
    pt = MODELOS / "pesos" / f"{args.nome}.pt"
    modelo = YOLO(str(pt))
    nomes_modelo = [modelo.names[i] for i in sorted(modelo.names)]
    if nomes_modelo != classes:
        print(f"ERRO: classes do modelo {nomes_modelo} diferentes do contrato {classes}.")
        return 1

    titulo("Exportando")
    caminho = modelo.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False, opset=12)
    destino = MODELOS / f"{args.destino}.onnx"
    shutil.copy2(caminho, destino)

    metricas = {}
    arq = RESULTADOS / args.nome / "metricas_teste.json"
    if arq.exists():
        metricas = json.loads(arq.read_text(encoding="utf-8"))
    meta = {
        "nome": args.nome,
        "classes": classes,
        "imgsz": args.imgsz,
        "exportado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metricas_teste": {k: metricas.get(k) for k in ("mAP50", "mAP50_95", "precisao", "recall")} if metricas else None,
    }
    destino.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{destino} ({destino.stat().st_size / 1e6:.1f} MB)")

    titulo("Conferindo ONNX × PyTorch")
    imagens = listar_imagens(DATASET / "test" / "images")[:5]
    det = DetectorOnnx(destino, classes, limiar_confianca=0.4)
    diferencas = 0
    for img in imagens:
        a = det.detectar(abrir_rgb(img))
        b = modelo.predict(str(img), conf=0.4, imgsz=args.imgsz, verbose=False)[0]
        ca = sorted(d.classe for d in a)
        cb = sorted(classes[int(c)] for c in b.boxes.cls.tolist())
        ok = ca == cb
        diferencas += not ok
        print(f"{'ok ' if ok else 'DIF'} {img.name}: onnx={ca} pt={cb}")
    if diferencas:
        print("Pequenas diferenças perto do limiar são normais (arredondamento); muitas indicam problema no export.")
    print("\nPróximo: python scripts/09_latencia.py  e  docker build (ver README)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
