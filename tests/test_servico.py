"""O serviço tem de cumprir o contrato `POST /deteccoes` do openapi.yaml do repositório principal."""

import jsonschema
import pytest
from fastapi.testclient import TestClient

from servico.app import criar_app
from servico.config import Config
from tests.conftest import jpeg

# Recorte do schema `Deteccao` do openapi.yaml (rc2)
SCHEMA_LISTA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["classe", "confianca", "caixa"],
        "additionalProperties": False,
        "properties": {
            "classe": {"enum": ["buraco", "piso_danificado", "obstaculo", "veiculo_na_calcada"]},
            "confianca": {"type": "number", "minimum": 0, "maximum": 1},
            "caixa": {
                "type": "object",
                "required": ["x", "y", "largura", "altura"],
                "properties": {k: {"type": "number", "minimum": 0, "maximum": 1}
                               for k in ("x", "y", "largura", "altura")},
            },
        },
    },
}
SCHEMA_ERRO = {"type": "object", "required": ["codigo", "mensagem"]}
TOKEN = "token-de-teste"


@pytest.fixture
def cliente(onnx_falso):
    app = criar_app(Config(modelo_path=str(onnx_falso), token=TOKEN))
    with TestClient(app) as c:
        yield c


def post(cliente, dados=None, tipo="image/jpeg", token=TOKEN):
    cab = {"Authorization": f"Bearer {token}"} if token else {}
    arquivos = {"imagem": ("foto.jpg", dados if dados is not None else jpeg(), tipo)}
    return cliente.post("/deteccoes", files=arquivos, headers=cab)


def test_200_no_formato_do_contrato(cliente):
    r = post(cliente)
    assert r.status_code == 200
    corpo = r.json()
    jsonschema.validate(corpo, SCHEMA_LISTA)
    assert [d["classe"] for d in corpo] == ["buraco", "piso_danificado"]
    confs = [d["confianca"] for d in corpo]
    assert confs == sorted(confs, reverse=True)


def test_png_tambem(cliente):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (300, 300)).save(buf, "PNG")
    assert post(cliente, buf.getvalue(), "image/png").status_code == 200


def test_401_sem_token_ou_token_errado(cliente):
    for token in (None, "errado"):
        r = post(cliente, token=token)
        assert r.status_code == 401
        assert r.json()["codigo"] == "nao_autenticado"


def test_413_acima_de_10_mb(cliente):
    r = post(cliente, b"\xff" * (10 * 1024 * 1024 + 1))
    assert r.status_code == 413 and r.json()["codigo"] == "arquivo_grande"


def test_422_imagem_ilegivel(cliente):
    r = post(cliente, b"isto nao e uma imagem")
    assert r.status_code == 422 and r.json()["codigo"] == "dados_invalidos"
    jsonschema.validate(r.json(), SCHEMA_ERRO)


def test_422_tipo_nao_aceito(cliente):
    r = post(cliente, b"GIF89a", "image/gif")
    assert r.status_code == 422


def test_400_sem_o_campo_imagem(cliente):
    r = cliente.post("/deteccoes", headers={"Authorization": f"Bearer {TOKEN}"}, data={"outro": "x"})
    assert r.status_code == 400 and r.json()["codigo"] == "requisicao_invalida"


def test_503_sem_modelo(tmp_path):
    app = criar_app(Config(modelo_path=str(tmp_path / "nao_existe.onnx"), token=TOKEN))
    with TestClient(app) as c:
        assert c.get("/saude").status_code == 503
        r = post(c)
        assert r.status_code == 503 and r.json()["codigo"] == "visao_indisponivel"


def test_saude_e_metricas(cliente):
    post(cliente)
    s = cliente.get("/saude").json()
    assert s["status"] == "ok" and s["limiar_confianca"] == 0.4
    m = cliente.get("/metrics").text
    assert 'visao_requisicoes_total{rota="/deteccoes",status="200"} 1' in m
    assert 'visao_deteccoes_total{classe="buraco"} 1' in m


def test_lista_vazia_quando_nada_passa_do_limiar(tmp_path):
    from tests.conftest import criar_onnx_falso

    modelo = criar_onnx_falso(tmp_path / "vazio.onnx", [(320, 320, 50, 50, 0, 0.2)])
    with TestClient(criar_app(Config(modelo_path=str(modelo), token=TOKEN))) as c:
        r = post(c)
        assert r.status_code == 200 and r.json() == []


def test_json_declara_charset_utf8(cliente):
    # sem o charset, o Invoke-RestMethod do Windows PowerShell 5.1 lê o corpo como latin-1
    for r in [post(cliente), post(cliente, token="errado"), cliente.get("/saude")]:
        assert r.headers["content-type"] == "application/json; charset=utf-8", r.request.url
