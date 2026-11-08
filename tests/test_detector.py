import numpy as np

from visao.classes import carregar_classes
from visao.detector_onnx import DetectorOnnx


def test_detector_onnx_desfaz_letterbox(onnx_falso):
    det = DetectorOnnx(onnx_falso, carregar_classes(), limiar_confianca=0.4)
    # imagem 1280×960: escala 0,5 e faixa de 80 px em cima e embaixo no letterbox 640×640
    resultado = det.detectar(np.zeros((960, 1280, 3), np.uint8))
    assert [d.classe for d in resultado] == ["buraco", "piso_danificado"]  # obstáculo 0,30 some; NMS junta os pisos
    b = resultado[0]
    # na entrada: centro (320,320), 100×50 -> na imagem: centro (640, 480), 200×100
    assert np.allclose(b.xyxy_px, (540, 430, 740, 530), atol=1)
    assert b.caixa == {"x": round(540 / 1280, 4), "y": round(430 / 960, 4),
                       "largura": round(200 / 1280, 4), "altura": round(100 / 960, 4)}
    assert resultado[0].confianca > resultado[1].confianca
