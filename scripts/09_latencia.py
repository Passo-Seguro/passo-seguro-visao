"""Mede a latência do detector ONNX em CPU (a nuvem não tem GPU), como o serviço vai rodar.

    python scripts/09_latencia.py [--modelo modelos/detector.onnx] [--repeticoes 100] [--threads 2]

`--threads` simula a quantidade de vCPUs do container (ex.: 2 no Container App). Grava
resultados/latencia_cpu.md com p50, p95 e média, separando pré-processamento, modelo e
pós-processamento. A API espera no máximo 5 s pela visão; o alvo é p95 bem abaixo de 1 s.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

from _comum import DATASET, MODELOS, RESULTADOS, titulo
from visao.classes import carregar_classes
from visao.detector_onnx import DetectorOnnx
from visao.imagens import abrir_rgb, listar_imagens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(MODELOS / "detector.onnx"))
    ap.add_argument("--repeticoes", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="0 = todas as do PC")
    args = ap.parse_args()

    import onnxruntime as ort

    opcoes = ort.SessionOptions()
    if args.threads:
        opcoes.intra_op_num_threads = args.threads
    sessao = ort.InferenceSession(args.modelo, sess_options=opcoes, providers=["CPUExecutionProvider"])
    det = DetectorOnnx(args.modelo, carregar_classes(), sessao=sessao)

    imagens = listar_imagens(DATASET / "test" / "images")[:20]
    if imagens:
        fotos = [abrir_rgb(p) for p in imagens]
    else:
        print("Sem imagens de teste; usando imagens sintéticas 1600×1200.")
        fotos = [np.random.randint(0, 255, (1200, 1600, 3), np.uint8) for _ in range(5)]

    for f in fotos[:3]:
        det.detectar(f)  # aquecimento
    pre, inf, pos, total = [], [], [], []
    for i in range(args.repeticoes):
        f = fotos[i % len(fotos)]
        t0 = time.perf_counter()
        tensor, escala, pad = det.preprocessar(f)
        t1 = time.perf_counter()
        saida = sessao.run(None, {det.nome_entrada: tensor})[0]
        t2 = time.perf_counter()
        det.posprocessar(saida, escala, pad, f.shape[1], f.shape[0])
        t3 = time.perf_counter()
        pre.append(t1 - t0); inf.append(t2 - t1); pos.append(t3 - t2); total.append(t3 - t0)

    def ms(v, q):
        return f"{np.percentile(np.array(v) * 1000, q):.1f}"

    cpu = platform.processor() or platform.machine()
    linhas = [
        f"Latência em CPU — {Path(args.modelo).name}, {args.repeticoes} execuções, "
        f"threads={args.threads or 'todas'}, CPU: {cpu}",
        "",
        "| Etapa | p50 (ms) | p95 (ms) | média (ms) |",
        "|---|---|---|---|",
    ]
    for nome, v in (("pré-processamento", pre), ("modelo", inf), ("pós-processamento", pos), ("total", total)):
        linhas.append(f"| {nome} | {ms(v, 50)} | {ms(v, 95)} | {np.mean(v) * 1000:.1f} |")
    linhas.append(f"\nVazão aproximada: {1 / np.mean(total):.1f} imagens/s por réplica.")
    titulo("Resultado")
    print("\n".join(linhas))
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "latencia_cpu.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
