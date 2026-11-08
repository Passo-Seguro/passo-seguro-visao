"""Detecção em tempo real com webcam ou vídeo gravado, mostrando caixas, classes e FPS.

    python scripts/10_tempo_real.py --fonte 0                          # webcam
    python scripts/10_tempo_real.py --fonte dados/videos/caminhada.mp4       # vídeo da caminhada
    python scripts/10_tempo_real.py --fonte dados/videos/caminhada.mp4 --salvar resultados/demo.mp4
    python scripts/10_tempo_real.py --fonte 0 --pt yolo11n_final_640   # usa o .pt na GPU

Por padrão usa modelos/detector.onnx (o MESMO modelo do serviço), em CPU, ou em GPU se o
onnxruntime-gpu estiver instalado. Teclas: q sai, p pausa, s salva o quadro em resultados/quadros/.
Para o vídeo da entrega, grave com --salvar: o arquivo sai com as caixas e o FPS desenhados.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from _comum import MODELOS, RESULTADOS, dispositivo_auto
from visao.classes import CORES, ROTULOS, carregar_classes
from visao.detector_onnx import DetectorOnnx


class DetectorPt:
    """Mesma interface do DetectorOnnx, mas com o .pt do Ultralytics (GPU)."""

    def __init__(self, nome, classes, conf):
        from ultralytics import YOLO

        self.m = YOLO(str(MODELOS / "pesos" / f"{nome}.pt"))
        self.classes, self.conf, self.device = classes, conf, dispositivo_auto()

    def detectar(self, rgb):
        r = self.m.predict(rgb, conf=self.conf, device=self.device, verbose=False)[0]
        saida = []
        for xyxy, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            saida.append(type("D", (), {"classe": self.classes[int(c)], "confianca": s, "xyxy_px": xyxy}))
        return saida


def desenhar(quadro, deteccoes, fps, ms):
    h, w = quadro.shape[:2]
    esp = max(2, w // 500)
    escala = max(0.5, w / 1600)
    for d in deteccoes:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy_px)
        cor = CORES[d.classe]
        cv2.rectangle(quadro, (x1, y1), (x2, y2), cor, esp)
        texto = f"{ROTULOS[d.classe]} {d.confianca:.2f}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
        cv2.rectangle(quadro, (x1, max(y1 - th - 10, 0)), (x1 + tw + 8, max(y1, th + 10)), cor, -1)
        cv2.putText(quadro, texto, (x1 + 4, max(y1 - 6, th + 4)), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 2)
    # painel do canto: FPS, latência e contagem por classe
    contagem = Counter(d.classe for d in deteccoes)
    linhas = [f"FPS: {fps:.1f}   modelo: {ms:.0f} ms", "Passo Seguro - deteccao em tempo real"]
    linhas += [f"{ROTULOS[c]}: {n}" for c, n in contagem.items()]
    fonte = max(0.35, w / 2400)
    alt = int(32 * fonte * 1.3)
    largura_painel = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fonte, 1)[0][0] for l in linhas) + 16
    painel = quadro.copy()
    cv2.rectangle(painel, (0, 0), (largura_painel, alt * len(linhas) + 8), (0, 0, 0), -1)
    quadro = cv2.addWeighted(painel, 0.55, quadro, 0.45, 0)  # semitransparente: não esconde as caixas
    for i, l in enumerate(linhas):
        cv2.putText(quadro, l, (8, alt * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fonte, (255, 255, 255), 1, cv2.LINE_AA)
    return quadro


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default="0", help="índice da webcam (0) ou caminho do vídeo")
    ap.add_argument("--modelo", default=str(MODELOS / "detector.onnx"))
    ap.add_argument("--pt", default=None, help="nome do .pt em modelos/pesos (usa GPU via Ultralytics)")
    ap.add_argument("--confianca", type=float, default=0.4)
    ap.add_argument("--salvar", default=None, help="grava o vídeo anotado (.mp4)")
    ap.add_argument("--largura-max", type=int, default=1280, help="reduz quadros maiores (FPS melhor)")
    ap.add_argument("--sem-janela", action="store_true", help="só grava, sem mostrar (servidor)")
    args = ap.parse_args()

    classes = carregar_classes()
    if args.pt:
        det = DetectorPt(args.pt, classes, args.confianca)
        print(f"Usando {args.pt}.pt em device={det.device}")
    else:
        import onnxruntime as ort

        provedores = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        det = DetectorOnnx(args.modelo, classes, limiar_confianca=args.confianca, providers=provedores)
        print(f"Usando {Path(args.modelo).name} com {provedores[0]}")

    fonte = int(args.fonte) if args.fonte.isdigit() else args.fonte
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print(f"Não consegui abrir a fonte {args.fonte}.")
        return 1
    gravador = None
    fps_media, pausado = 0.0, False
    (RESULTADOS / "quadros").mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        if not pausado:
            ok, quadro = cap.read()
            if not ok:
                break
            h, w = quadro.shape[:2]
            if w > args.largura_max:
                quadro = cv2.resize(quadro, (args.largura_max, int(h * args.largura_max / w)))
            t0 = time.perf_counter()
            deteccoes = det.detectar(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
            dt = time.perf_counter() - t0
            fps_media = 1 / dt if fps_media == 0 else 0.9 * fps_media + 0.1 / max(dt, 1e-6)
            quadro = desenhar(quadro, deteccoes, fps_media, dt * 1000)
            if args.salvar:
                if gravador is None:
                    Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
                    fps_fonte = cap.get(cv2.CAP_PROP_FPS) or 30
                    gravador = cv2.VideoWriter(args.salvar, cv2.VideoWriter_fourcc(*"mp4v"), fps_fonte,
                                               (quadro.shape[1], quadro.shape[0]))
                gravador.write(quadro)
            n += 1
        if args.sem_janela:
            continue
        cv2.imshow("Passo Seguro - q sai, p pausa, s salva quadro", quadro)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("p"):
            pausado = not pausado
        if tecla == ord("s"):
            destino = RESULTADOS / "quadros" / f"quadro_{n:05d}.jpg"
            cv2.imwrite(str(destino), quadro)
            print(f"salvo {destino}")
    cap.release()
    if gravador:
        gravador.release()
        print(f"Vídeo anotado: {args.salvar}")
    cv2.destroyAllWindows()
    print(f"{n} quadros, FPS médio final {fps_media:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
