import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


def criar_onnx_falso(caminho: Path, caixas: list[tuple], nc: int = 4, n: int = 8400, tamanho: int = 640) -> Path:
    """ONNX minúsculo com a MESMA interface do YOLO exportado: entrada (1,3,640,640) e saída
    (1, 4+nc, N). A saída é fixa: `caixas` = [(cx, cy, w, h, classe, score)] em pixels da entrada."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    saida = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(caixas):
        saida[0, :4, i] = [cx, cy, w, h]
        saida[0, 4 + c, i] = s
    const = numpy_helper.from_array(saida, "fixo")
    zero = numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")
    nos = [
        helper.make_node("ReduceMean", ["images"], ["media"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["media", "zero"], ["nulo"]),
        helper.make_node("Add", ["fixo", "nulo"], ["output0"]),
    ]
    grafo = helper.make_graph(
        nos, "yolo_falso",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, tamanho, tamanho])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + nc, n])],
        initializer=[const, zero],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 13)])
    modelo.ir_version = 8
    onnx.save(modelo, caminho)
    return caminho


@pytest.fixture
def onnx_falso(tmp_path):
    # um buraco no centro (alta confiança), um obstáculo de baixa confiança (abaixo de 0,4)
    # e dois piso_danificado quase sobrepostos (o NMS deve deixar um só)
    return criar_onnx_falso(tmp_path / "detector.onnx", [
        (320, 320, 100, 50, 0, 0.91),
        (100, 100, 40, 40, 2, 0.30),
        (500, 400, 80, 80, 1, 0.70),
        (502, 401, 80, 80, 1, 0.65),
    ])


def jpeg(largura=800, altura=600, cor=(120, 120, 120)) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (largura, altura), cor).save(buf, "JPEG")
    return buf.getvalue()
