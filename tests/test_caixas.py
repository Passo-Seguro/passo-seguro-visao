import numpy as np

from visao.caixas import cxcywh_para_xyxy, iou, nms, parear, xyxy_para_contrato


def test_centro_para_canto_superior_esquerdo():
    xyxy = cxcywh_para_xyxy(np.array([[0.5, 0.5, 0.2, 0.4]]))[0]
    assert np.allclose(xyxy, [0.4, 0.3, 0.6, 0.7])


def test_contrato_normaliza_e_prende_entre_0_e_1():
    c = xyxy_para_contrato([-10, 50, 210, 150], 200, 100)
    assert c == {"x": 0.0, "y": 0.5, "largura": 1.0, "altura": 0.5}


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(iou(a, b), [[50 / 150, 0]])


def test_nms_tira_sobreposta():
    caixas = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(caixas, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_parear_classifica_erros():
    gt = np.array([[0, 0, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]])
    pred = np.array([[0, 0, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9], [0.1, 0.8, 0.2, 0.9]])
    p = parear(pred, [0, 2, 1], [0.9, 0.8, 0.7], gt, [0, 1])
    assert p.verdadeiros_positivos == [(0, 0)]
    assert sorted(p.falsos_positivos) == [1, 2]
    assert p.confusoes == [(1, 1)]  # previu obstáculo onde era piso danificado
    assert p.falsos_negativos == [1]
