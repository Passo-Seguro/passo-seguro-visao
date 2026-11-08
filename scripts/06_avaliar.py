"""Avalia os modelos treinados no conjunto de TESTE (nunca visto no treino nem na escolha).

    python scripts/06_avaliar.py --nome yolo11n_final_640 --nome yolo11s_final_640
    python scripts/06_avaliar.py --todos            # todos os modelos em modelos/pesos/
    python scripts/06_avaliar.py --todos --coco     # + linha de base: COCO sem ajuste (só veículo)

Para cada modelo grava em resultados/<nome>/:
    metricas_teste.json        mAP50, mAP50-95, precisão e recall gerais e por classe
    metricas_por_classe.csv    a mesma tabela, para o PDF
    confusion_matrix*.png      matriz de confusão (absoluta e normalizada)
E resultados/comparacao.md com todos lado a lado (nano × small × COCO).
"""

import argparse
import csv
import json
import shutil
import sys

import numpy as np

from _comum import DATASET, EXECUCOES, MODELOS, PESOS_COCO, RESULTADOS, dispositivo_auto, titulo
from visao.caixas import cxcywh_para_xyxy, iou, ler_rotulo_yolo
from visao.classes import carregar_classes
from visao.imagens import listar_imagens


def avaliar_modelo(nome: str, classes: list[str], device: str, imgsz: int) -> dict:
    from ultralytics import YOLO

    modelo = YOLO(str(MODELOS / "pesos" / f"{nome}.pt"))
    m = modelo.val(data=str(DATASET / "data.yaml"), split="test", imgsz=imgsz, device=device,
                   project=str(EXECUCOES), name=f"{nome}_teste", exist_ok=True, plots=True, verbose=False)
    por_classe = {}
    for i, idx in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        por_classe[classes[int(idx)]] = {"precisao": round(float(p), 4), "recall": round(float(r), 4),
                                          "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    vel = m.speed  # ms por imagem: preprocess, inference, postprocess
    resultado = {
        "nome": nome,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precisao": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "inferencia_ms_por_imagem": round(float(vel.get("inference", 0)), 1),
        "device": device,
        "por_classe": por_classe,
    }
    destino = RESULTADOS / nome
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "metricas_teste.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(destino / "metricas_por_classe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["classe", "precisao", "recall", "mAP50", "mAP50_95"])
        for c, v in por_classe.items():
            w.writerow([c, v["precisao"], v["recall"], v["mAP50"], v["mAP50_95"]])
    for png in ("confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "BoxPR_curve.png"):
        origem = EXECUCOES / f"{nome}_teste" / png
        if origem.exists():
            shutil.copy2(origem, destino / png)
    return resultado


def ap50_uma_classe(preds, gts) -> tuple[float, float, float]:
    """AP50 (interpolação de 101 pontos, como o COCO), precisão e recall para uma classe.
    preds: lista de (id_img, conf, xyxy); gts: dict id_img -> array xyxy."""
    total_gt = sum(len(v) for v in gts.values())
    if total_gt == 0:
        return 0.0, 0.0, 0.0
    preds = sorted(preds, key=lambda p: -p[1])
    usados = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    vp = np.zeros(len(preds))
    for i, (img, _, caixa) in enumerate(preds):
        g = gts.get(img, np.zeros((0, 4)))
        if len(g) == 0:
            continue
        ious = iou(np.array([caixa]), g)[0]
        j = int(ious.argmax())
        if ious[j] >= 0.5 and not usados[img][j]:
            usados[img][j] = True
            vp[i] = 1
    acum_vp = np.cumsum(vp)
    recall = acum_vp / total_gt
    precisao = acum_vp / np.arange(1, len(preds) + 1)
    if len(preds) == 0:
        return 0.0, 0.0, 0.0
    env = np.maximum.accumulate(precisao[::-1])[::-1]
    pontos = np.linspace(0, 1, 101)
    ap = np.mean([env[recall >= t].max() if (recall >= t).any() else 0 for t in pontos])
    return float(ap), float(precisao[-1]), float(recall[-1])


def avaliar_coco(classes: list[str], device: str, imgsz: int) -> dict:
    """Linha de base: YOLO do COCO SEM ajuste. Ele só conhece veículos, então a comparação
    justa é na classe veiculo_na_calcada. Mostra o ganho do ajuste fino — e que ele marca
    também os carros parados corretamente na rua (falsos positivos)."""
    from ultralytics import YOLO

    id_v = classes.index("veiculo_na_calcada")
    PESOS_COCO.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(str(PESOS_COCO / "yolo11n.pt"))
    imagens = listar_imagens(DATASET / "test" / "images")
    preds, gts = [], {}
    for k in range(0, len(imagens), 16):
        lote = imagens[k : k + 16]
        for img, r in zip(lote, modelo.predict([str(p) for p in lote], conf=0.001, imgsz=imgsz,
                                               classes=[1, 2, 3, 5, 7], device=device, verbose=False)):
            cx, cl = ler_rotulo_yolo(DATASET / "test" / "labels" / f"{img.stem}.txt")
            gts[img.name] = cxcywh_para_xyxy(cx[[c == id_v for c in cl]]) if len(cl) else np.zeros((0, 4))
            for caixa, conf in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist()):
                preds.append((img.name, conf, caixa))
    ap, p, rc = ap50_uma_classe(preds, gts)
    resultado = {"nome": "coco_sem_ajuste (só veiculo_na_calcada)", "por_classe": {
        "veiculo_na_calcada": {"mAP50": round(ap, 4), "precisao": round(p, 4), "recall": round(rc, 4)}}}
    (RESULTADOS / "coco_sem_ajuste.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    return resultado


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", action="append", default=[])
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--coco", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    nomes = args.nome or ([p.stem for p in sorted((MODELOS / "pesos").glob("*.pt"))] if args.todos else [])
    if not nomes:
        print("Diga --nome <execucao> ou --todos.")
        return 1
    classes = carregar_classes()
    device = args.device or dispositivo_auto()
    resultados = []
    for n in nomes:
        titulo(n)
        r = avaliar_modelo(n, classes, device, args.imgsz)
        print(json.dumps({k: v for k, v in r.items() if k != "por_classe"}, ensure_ascii=False))
        resultados.append(r)

    linhas = ["| Modelo | mAP50 | mAP50-95 | Precisão | Recall | ms/imagem (" + device + ") |",
              "|---|---|---|---|---|---|"]
    for r in resultados:
        linhas.append(f"| {r['nome']} | {r['mAP50']} | {r['mAP50_95']} | {r['precisao']} | {r['recall']} | "
                      f"{r['inferencia_ms_por_imagem']} |")
    linhas += ["", "Por classe (mAP50 / recall):", "",
               "| Modelo | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for r in resultados:
        cel = [f"{r['por_classe'].get(c, {}).get('mAP50', '-')} / {r['por_classe'].get(c, {}).get('recall', '-')}"
               for c in classes]
        linhas.append(f"| {r['nome']} | " + " | ".join(cel) + " |")
    if args.coco:
        titulo("COCO sem ajuste")
        c = avaliar_coco(classes, device, args.imgsz)["por_classe"]["veiculo_na_calcada"]
        print(c)
        cel = ["-"] * len(classes)
        cel[classes.index("veiculo_na_calcada")] = f"{c['mAP50']} / {c['recall']}"
        linhas.append("| COCO sem ajuste | " + " | ".join(cel) + " |")
    (RESULTADOS / "comparacao.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n" + "\n".join(linhas))
    print(f"\nTabela para o PDF: {RESULTADOS / 'comparacao.md'}")
    if resultados and max(r["mAP50"] for r in resultados) < 0.5:
        print("\nmAP50 abaixo de 0,5: veja o plano B no README.md, seção 6.6 (3 classes, mais dados, mais augmentation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
