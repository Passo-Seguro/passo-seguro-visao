"""Conversões de caixa, IoU, NMS e pareamento predição × verdade.

Formatos usados no projeto:
- YOLO (rótulo .txt): classe cx cy w h, normalizados 0..1, centro da caixa.
- Contrato do Passo Seguro (`Caixa`): x, y, largura, altura normalizados 0..1, com x e y no
  CANTO SUPERIOR ESQUERDO.
- Interno (xyxy): x1, y1, x2, y2 em pixels ou normalizados.
"""

from dataclasses import dataclass

import numpy as np


def cxcywh_para_xyxy(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    x1 = c[..., 0] - c[..., 2] / 2
    y1 = c[..., 1] - c[..., 3] / 2
    x2 = c[..., 0] + c[..., 2] / 2
    y2 = c[..., 1] + c[..., 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy_para_contrato(xyxy, largura_img: float, altura_img: float) -> dict:
    """xyxy em pixels -> {x, y, largura, altura} normalizado e preso entre 0 e 1."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1 = min(max(x1 / largura_img, 0.0), 1.0)
    x2 = min(max(x2 / largura_img, 0.0), 1.0)
    y1 = min(max(y1 / altura_img, 0.0), 1.0)
    y2 = min(max(y2 / altura_img, 0.0), 1.0)
    return {
        "x": round(x1, 4),
        "y": round(y1, 4),
        "largura": round(max(x2 - x1, 0.0), 4),
        "altura": round(max(y2 - y1, 0.0), 4),
    }


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cada caixa de `a` (N,4) e de `b` (M,4), em xyxy. Devolve (N, M)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(caixas: np.ndarray, scores: np.ndarray, limiar_iou: float) -> list[int]:
    """Non-maximum suppression simples (por classe, quem chama separa as classes)."""
    ordem = np.argsort(-scores)
    manter: list[int] = []
    while len(ordem):
        i = int(ordem[0])
        manter.append(i)
        if len(ordem) == 1:
            break
        ious = iou(caixas[i : i + 1], caixas[ordem[1:]])[0]
        ordem = ordem[1:][ious < limiar_iou]
    return manter


@dataclass
class Pareamento:
    verdadeiros_positivos: list[tuple[int, int]]  # (indice_pred, indice_gt)
    falsos_positivos: list[int]  # indice_pred
    falsos_negativos: list[int]  # indice_gt
    confusoes: list[tuple[int, int]]  # pred com IoU bom mas classe errada (indice_pred, indice_gt)


def parear(pred_xyxy, pred_cls, pred_conf, gt_xyxy, gt_cls, limiar_iou: float = 0.5) -> Pareamento:
    """Pareia predições com a verdade (guloso por confiança), como na avaliação de mAP50."""
    pred_xyxy = np.asarray(pred_xyxy, dtype=np.float32).reshape(-1, 4)
    gt_xyxy = np.asarray(gt_xyxy, dtype=np.float32).reshape(-1, 4)
    pred_cls = list(pred_cls)
    gt_cls = list(gt_cls)
    ordem = np.argsort(-np.asarray(pred_conf, dtype=np.float32)) if len(pred_cls) else []
    matriz = iou(pred_xyxy, gt_xyxy)
    usados: set[int] = set()
    vp, fp, conf = [], [], []
    for p in ordem:
        p = int(p)
        melhor, melhor_iou = -1, limiar_iou
        for g in range(len(gt_cls)):
            if g in usados or gt_cls[g] != pred_cls[p]:
                continue
            if matriz[p, g] >= melhor_iou:
                melhor, melhor_iou = g, matriz[p, g]
        if melhor >= 0:
            usados.add(melhor)
            vp.append((p, melhor))
            continue
        # sem par da mesma classe: confusão de classe se encostar numa caixa de outra classe
        outras = [g for g in range(len(gt_cls)) if g not in usados and matriz[p, g] >= limiar_iou]
        if outras:
            conf.append((p, max(outras, key=lambda g: matriz[p, g])))
        fp.append(p)
    fn = [g for g in range(len(gt_cls)) if g not in usados]
    return Pareamento(vp, fp, fn, conf)


def ler_rotulo_yolo(caminho) -> tuple[np.ndarray, list[int]]:
    """Lê um .txt YOLO (caixas). Devolve (cxcywh normalizado (N,4), classes)."""
    caixas, classes = [], []
    try:
        texto = open(caminho, encoding="utf-8").read()
    except FileNotFoundError:
        return np.zeros((0, 4), dtype=np.float32), []
    for linha in texto.splitlines():
        partes = linha.split()
        if len(partes) < 5:
            continue
        if len(partes) > 5:
            # polígono (segmentação, comum no Roboflow): vira a caixa que o envolve
            pts = np.array(partes[1:], dtype=np.float32).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            caixas.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        else:
            caixas.append([float(v) for v in partes[1:5]])
        classes.append(int(float(partes[0])))
    return np.array(caixas, dtype=np.float32).reshape(-1, 4), classes
