"""Treina o YOLO na sua GPU (ou CPU, se não houver) com o dataset de dados/dataset/.

Exemplos:
    # passo 4 — baseline rápido, só para validar o pipeline de ponta a ponta
    python scripts/05_treinar.py --preset baseline

    # passo 5 — treino final, um de cada para comparar
    python scripts/05_treinar.py --preset final --modelo n
    python scripts/05_treinar.py --preset final --modelo s

    # sem GPU ou com pouca VRAM
    python scripts/05_treinar.py --preset final --modelo n --imgsz 512 --batch 8

Saídas:
    execucoes/<nome>/          tudo o que o Ultralytics gera (curvas, matriz de confusão, pesos)
    modelos/pesos/<nome>.pt    melhor peso (best.pt), usado pelos passos seguintes
    resultados/<nome>/treino.json  resumo (melhor época, mAP50, mAP50-95, tempo, GPU)
"""

import argparse
import json
import platform
import shutil
import sys
import time

from _comum import DATASET, EXECUCOES, MODELOS, RESULTADOS, dispositivo_auto, titulo

PRESETS = {
    "baseline": {"epocas": 20, "paciencia": 0},
    "final": {"epocas": 100, "paciencia": 20},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="final")
    ap.add_argument("--modelo", choices=["n", "s", "m"], default="n", help="n = nano, s = small")
    ap.add_argument("--familia", default="yolo11", help="yolo11 ou yolov8")
    ap.add_argument("--epocas", type=int)
    ap.add_argument("--paciencia", type=int, help="parada antecipada: épocas sem melhorar")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = automático pela VRAM (só GPU)")
    ap.add_argument("--device", default=None, help="0, cpu, mps (padrão: detecta)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--nome", default=None)
    ap.add_argument("--retomar", action="store_true", help="continua um treino interrompido (mesmo --nome)")
    ap.add_argument("--do-zero", action="store_true",
                    help="sem pesos do COCO (para comparar treino do zero × transfer learning)")
    args = ap.parse_args()

    data_yaml = DATASET / "data.yaml"
    if not data_yaml.exists():
        print("dados/dataset/data.yaml não existe. Rode o 04_montar_dataset.py.")
        return 1

    preset = PRESETS[args.preset]
    epocas = args.epocas or preset["epocas"]
    paciencia = preset["paciencia"] if args.paciencia is None else args.paciencia
    device = args.device or dispositivo_auto()
    batch = args.batch
    if device == "cpu" and batch == -1:
        batch = 8  # o batch automático só funciona com CUDA
    # no Windows, muitos workers no DataLoader travam ou estouram a RAM
    workers = args.workers if args.workers is not None else (2 if platform.system() == "Windows" else 8)
    nome = args.nome or f"{args.familia}{args.modelo}_{args.preset}_{args.imgsz}" + ("_do_zero" if args.do_zero else "")

    from ultralytics import YOLO

    titulo(f"Treinando {nome} em device={device} ({epocas} épocas, paciência {paciencia}, batch {batch})")
    if args.retomar:
        modelo = YOLO(str(EXECUCOES / nome / "weights" / "last.pt"))
        inicio = time.time()
        modelo.train(resume=True)
    else:
        # .pt = pré-treinado no COCO (transfer learning); .yaml = mesma arquitetura, pesos aleatórios
        modelo = YOLO(f"{args.familia}{args.modelo}.{'yaml' if args.do_zero else 'pt'}")
        inicio = time.time()
        modelo.train(
            data=str(data_yaml),
            epochs=epocas,
            patience=paciencia,
            imgsz=args.imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=str(EXECUCOES),
            name=nome,
            exist_ok=True,
            seed=42,
            deterministic=False,
            plots=True,
            # augmentation: padrão do Ultralytics + ajustes para foto de rua feita no celular
            fliplr=0.5,
            flipud=0.0,      # calçada de cabeça para baixo não existe
            degrees=5.0,     # celular levemente torto
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,  # sol, sombra, fim de tarde
            mosaic=1.0,
            close_mosaic=10,
            mixup=0.05,
            amp=device != "cpu",
        )
    duracao_min = (time.time() - inicio) / 60

    pasta = EXECUCOES / nome
    best = pasta / "weights" / "best.pt"
    (MODELOS / "pesos").mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, MODELOS / "pesos" / f"{nome}.pt")

    resumo = {"nome": nome, "pre_treino": "nenhum" if args.do_zero else "COCO", "familia": args.familia, "modelo": args.modelo, "preset": args.preset,
              "epocas_pedidas": epocas, "imgsz": args.imgsz, "batch": batch, "device": device,
              "duracao_min": round(duracao_min, 1)}
    try:
        import pandas as pd

        res = pd.read_csv(pasta / "results.csv")
        res.columns = [c.strip() for c in res.columns]
        melhor = res.loc[res["metrics/mAP50-95(B)"].idxmax()]
        resumo.update({
            "epocas_rodadas": int(res["epoch"].max()),
            "melhor_epoca": int(melhor["epoch"]),
            "val_mAP50": round(float(melhor["metrics/mAP50(B)"]), 4),
            "val_mAP50_95": round(float(melhor["metrics/mAP50-95(B)"]), 4),
            "val_precisao": round(float(melhor["metrics/precision(B)"]), 4),
            "val_recall": round(float(melhor["metrics/recall(B)"]), 4),
        })
    except Exception as erro:  # resumo é bônus; o peso já foi salvo
        print(f"(não consegui ler results.csv: {erro})")
    try:
        import torch

        if torch.cuda.is_available():
            resumo["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    destino = RESULTADOS / nome
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "treino.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    titulo("Pronto")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(f"\nPróximo: python scripts/06_avaliar.py --nome {nome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
