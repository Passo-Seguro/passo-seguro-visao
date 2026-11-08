"""Detector YOLO exportado para ONNX, sem depender do Ultralytics (só numpy + ONNX Runtime).

A saída do YOLOv8/YOLO11 exportado tem forma (1, 4 + nc, N): para cada uma das N âncoras,
cx, cy, w, h em pixels da entrada (ex.: 640×640) e a pontuação de cada classe (já com
sigmoide). Aqui fazemos o letterbox, o filtro por confiança, o NMS por classe e a conversão
para o formato do contrato (canto superior esquerdo, normalizado 0..1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visao.caixas import cxcywh_para_xyxy, nms, xyxy_para_contrato


@dataclass
class Deteccao:
    classe: str
    confianca: float
    caixa: dict  # {x, y, largura, altura} normalizado, canto superior esquerdo
    xyxy_px: tuple[float, float, float, float]  # para desenhar na tela

    def para_contrato(self) -> dict:
        return {"classe": self.classe, "confianca": round(self.confianca, 4), "caixa": self.caixa}


def letterbox(rgb: np.ndarray, tamanho: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Redimensiona mantendo a proporção e completa com cinza (114), como no treino do YOLO."""
    from PIL import Image

    h, w = rgb.shape[:2]
    escala = min(tamanho / h, tamanho / w)
    nw, nh = int(round(w * escala)), int(round(h * escala))
    redim = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    tela = np.full((tamanho, tamanho, 3), 114, dtype=np.uint8)
    dx, dy = (tamanho - nw) / 2, (tamanho - nh) / 2
    top, left = int(round(dy - 0.1)), int(round(dx - 0.1))
    tela[top : top + nh, left : left + nw] = redim
    return tela, escala, (left, top)


class DetectorOnnx:
    def __init__(
        self,
        caminho_modelo: str | Path,
        classes: list[str],
        limiar_confianca: float = 0.4,
        limiar_iou: float = 0.45,
        max_deteccoes: int = 50,
        providers: list[str] | None = None,
        sessao=None,
    ):
        self.classes = classes
        self.limiar_confianca = limiar_confianca
        self.limiar_iou = limiar_iou
        self.max_deteccoes = max_deteccoes
        if sessao is None:
            import onnxruntime as ort

            opcoes = ort.SessionOptions()
            opcoes.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessao = ort.InferenceSession(
                str(caminho_modelo), sess_options=opcoes, providers=providers or ["CPUExecutionProvider"]
            )
        self.sessao = sessao
        entrada = sessao.get_inputs()[0]
        self.nome_entrada = entrada.name
        forma = entrada.shape  # [1, 3, 640, 640]
        self.tamanho = int(forma[2]) if isinstance(forma[2], int) else 640

    # ------------------------------------------------------------------ pré e pós
    def preprocessar(self, rgb: np.ndarray):
        tela, escala, pad = letterbox(rgb, self.tamanho)
        tensor = tela.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(tensor), escala, pad

    def posprocessar(self, saida: np.ndarray, escala: float, pad, largura: int, altura: int) -> list[Deteccao]:
        pred = np.asarray(saida)[0]  # (4+nc, N)
        if pred.shape[0] != 4 + len(self.classes):
            pred = pred.T  # alguns exports saem (N, 4+nc)
        if pred.shape[0] != 4 + len(self.classes):
            raise ValueError(
                f"saída do modelo tem {pred.shape[0]} linhas; esperava {4 + len(self.classes)} "
                f"(4 coordenadas + {len(self.classes)} classes). O modelo foi treinado com as classes certas?"
            )
        caixas = pred[:4].T
        scores = pred[4:].T
        cls = scores.argmax(axis=1)
        conf = scores[np.arange(len(cls)), cls]
        ok = conf >= self.limiar_confianca
        caixas, cls, conf = caixas[ok], cls[ok], conf[ok]
        if len(conf) == 0:
            return []
        xyxy = cxcywh_para_xyxy(caixas)
        # desfaz o letterbox: volta para pixels da imagem original
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / escala
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / escala
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, largura)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, altura)

        resultado: list[Deteccao] = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            for i in nms(xyxy[idx], conf[idx], self.limiar_iou):
                j = idx[i]
                caixa = xyxy_para_contrato(xyxy[j], largura, altura)
                if caixa["largura"] <= 0 or caixa["altura"] <= 0:
                    continue
                resultado.append(
                    Deteccao(
                        classe=self.classes[int(c)],
                        confianca=float(conf[j]),
                        caixa=caixa,
                        xyxy_px=tuple(float(v) for v in xyxy[j]),
                    )
                )
        resultado.sort(key=lambda d: d.confianca, reverse=True)
        return resultado[: self.max_deteccoes]

    # ------------------------------------------------------------------ uso
    def detectar(self, rgb: np.ndarray) -> list[Deteccao]:
        """`rgb`: imagem (H, W, 3) uint8 em RGB. Devolve da maior para a menor confiança."""
        altura, largura = rgb.shape[:2]
        tensor, escala, pad = self.preprocessar(rgb)
        saida = self.sessao.run(None, {self.nome_entrada: tensor})[0]
        return self.posprocessar(saida, escala, pad, largura, altura)


def carregar_metadados(caminho_modelo: str | Path) -> dict:
    """Lê o modelos/<nome>.json gravado pelo 08_exportar_onnx.py (classes, imgsz, métricas)."""
    meta = Path(caminho_modelo).with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}
