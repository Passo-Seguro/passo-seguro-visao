"""Análise de erros no TESTE: separa acertos, falsos positivos, falsos negativos e confusões.

    python scripts/07_analise_erros.py --nome yolo11n_final_640 [--confianca 0.4]

Usa o mesmo limiar de confiança do serviço (0,4), para os erros serem os que o app veria.
Saídas em resultados/<nome>/erros/:
    acertos/  falsos_positivos/  falsos_negativos/  confusao_de_classe/
        imagens com a VERDADE em verde e a PREDIÇÃO na cor da classe
    erros.csv   uma linha por erro, com a coluna `causa` VAZIA para você preencher olhando a
                imagem: sombra, bueiro_tampado, folhas, mancha, reflexo, caixa_imprecisa,
                rotulo_errado, objeto_pequeno, oclusao, outro
    resumo.md   contagem por tipo e por classe (e por causa, depois de preenchida)

Depois de preencher a coluna `causa`, rode de novo com --so-resumo para o resumo contar as causas.
"""

import argparse
import csv
import sys
from collections import Counter

import cv2
import numpy as np

from _comum import DATASET, MODELOS, RESULTADOS, dispositivo_auto, titulo
from visao.caixas import cxcywh_para_xyxy, iou, ler_rotulo_yolo, parear
from visao.classes import CORES, ROTULOS, carregar_classes
from visao.imagens import listar_imagens

TIPOS = ("acertos", "falsos_positivos", "falsos_negativos", "confusao_de_classe")


def desenhar(bgr, gt_xyxy, gt_cls, pred_xyxy, pred_cls, pred_conf, classes):
    h, w = bgr.shape[:2]
    out = bgr.copy()
    esp = max(2, w // 400)
    for (x1, y1, x2, y2), c in zip(gt_xyxy, gt_cls):
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, (0, 200, 0), esp)
        cv2.putText(out, "real: " + ROTULOS[classes[c]], (p1[0], p2[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    for (x1, y1, x2, y2), c, s in zip(pred_xyxy, pred_cls, pred_conf):
        cor = CORES[classes[c]]
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, cor, esp)
        cv2.putText(out, f"{ROTULOS[classes[c]]} {s:.2f}", (p1[0], max(p1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    return out


def resumir(pasta, classes):
    with open(pasta / "erros.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    por_tipo = Counter(l["tipo"] for l in linhas)
    por_classe = Counter((l["tipo"], l["classe_pred"] or l["classe_real"]) for l in linhas)
    causas = Counter(l["causa"] for l in linhas if l["causa"])
    texto = ["# Análise de erros", "", "| Tipo | Quantidade |", "|---|---|"]
    texto += [f"| {t} | {por_tipo.get(t, 0)} |" for t in TIPOS if t != "acertos"]
    texto += ["", "| Tipo | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for t in ("falsos_positivos", "falsos_negativos", "confusao_de_classe"):
        texto.append(f"| {t} | " + " | ".join(str(por_classe.get((t, c), 0)) for c in classes) + " |")
    if causas:
        texto += ["", "## Causas (preenchidas à mão)", "", "| Causa | Quantidade |", "|---|---|"]
        texto += [f"| {c} | {n} |" for c, n in causas.most_common()]
    else:
        texto += ["", "_Preencha a coluna `causa` em erros.csv e rode com --so-resumo._"]
    (pasta / "resumo.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n".join(texto))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--confianca", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--so-resumo", action="store_true")
    args = ap.parse_args()

    classes = carregar_classes()
    pasta = RESULTADOS / args.nome / "erros"
    if args.so_resumo:
        resumir(pasta, classes)
        return 0

    from ultralytics import YOLO

    modelo = YOLO(str(MODELOS / "pesos" / f"{args.nome}.pt"))
    for t in TIPOS:
        (pasta / t).mkdir(parents=True, exist_ok=True)
    imagens = listar_imagens(DATASET / "test" / "images")
    titulo(f"{len(imagens)} imagens de teste, confiança >= {args.confianca}")
    linhas = []
    for k in range(0, len(imagens), 16):
        lote = imagens[k : k + 16]
        res = modelo.predict([str(p) for p in lote], conf=args.confianca, imgsz=args.imgsz,
                             device=dispositivo_auto(), verbose=False)
        for img, r in zip(lote, res):
            cx, gcls = ler_rotulo_yolo(DATASET / "test" / "labels" / f"{img.stem}.txt")
            gxy = cxcywh_para_xyxy(cx) if len(gcls) else np.zeros((0, 4))
            pxy = np.array(r.boxes.xyxyn.tolist()).reshape(-1, 4)
            pcls = [int(c) for c in r.boxes.cls.tolist()]
            pconf = r.boxes.conf.tolist()
            par = parear(pxy, pcls, pconf, gxy, gcls)
            confusos = {p for p, _ in par.confusoes}
            tipos_img = set()
            for p, g in par.verdadeiros_positivos:
                tipos_img.add("acertos")
            for p in par.falsos_positivos:
                tipo = "confusao_de_classe" if p in confusos else "falsos_positivos"
                tipos_img.add(tipo)
                g = next((g for pp, g in par.confusoes if pp == p), None)
                melhor_iou = float(iou(pxy[p:p + 1], gxy).max()) if len(gxy) else 0.0
                linhas.append({"arquivo": img.name, "tipo": tipo, "classe_pred": classes[pcls[p]],
                               "classe_real": classes[gcls[g]] if g is not None else "",
                               "confianca": round(pconf[p], 3), "iou": round(melhor_iou, 3), "causa": ""})
            for g in par.falsos_negativos:
                if any(gg == g for _, gg in par.confusoes):
                    continue  # já contado como confusão
                tipos_img.add("falsos_negativos")
                linhas.append({"arquivo": img.name, "tipo": "falsos_negativos", "classe_pred": "",
                               "classe_real": classes[gcls[g]], "confianca": "", "iou": "", "causa": ""})
            if not tipos_img:
                continue
            desenho = desenhar(cv2.imread(str(img)), gxy, gcls, pxy, pcls, pconf, classes)
            for t in tipos_img:
                cv2.imwrite(str(pasta / t / img.name), desenho)

    with open(pasta / "erros.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "tipo", "classe_pred", "classe_real", "confianca", "iou", "causa"])
        w.writeheader()
        w.writerows(linhas)
    resumir(pasta, classes)
    print(f"\nImagens em {pasta}. Escolha 3 a 5 falsos positivos típicos (sombra, bueiro, folhas) para o PDF e o vídeo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
