import importlib.util
import sys

from tests.conftest import RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
_spec = importlib.util.spec_from_file_location("baixar_publicos", RAIZ / "scripts" / "01a_baixar_publicos.py")
baixar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baixar)
importar = importlib.import_module("01_importar_publicos")


def test_copias_da_mesma_foto_ficam_no_mesmo_grupo_e_fotos_vizinhas_nao():
    a = importar.grupo_da_imagem(baixar.nome_novo("images218_jpg.rf.34bb.jpg"))
    b = importar.grupo_da_imagem(baixar.nome_novo("images218_jpg.rf.8c94.jpg"))
    c = importar.grupo_da_imagem(baixar.nome_novo("images219_jpg.rf.1111.jpg"))
    assert a == b != c
    # sem o nome novo, o 01 juntaria images218 e images219 no grupo "images"
    assert importar.grupo_da_imagem("images218_jpg.rf.34bb.jpg") == importar.grupo_da_imagem("images219_jpg.rf.1.jpg")


def test_rotulo_valido_aceita_caixa_e_poligono():
    assert baixar.rotulo_valido(["0 0.5 0.5 0.2 0.1"], nc=1)
    assert baixar.rotulo_valido(["0 0.1 0.1 0.2 0.1 0.2 0.3"], nc=1)
    assert baixar.rotulo_valido([], nc=1)  # imagem sem caixa = fundo


def test_rotulo_invalido_descarta_a_imagem():
    assert not baixar.rotulo_valido(["Dear Hiring Manager,"], nc=1)  # o .txt do rupesh002
    assert not baixar.rotulo_valido(["91 0.5 0.5 0.2 0.1"], nc=1)    # classe fora do data.yaml
    assert not baixar.rotulo_valido(["0 0.5 1.7 0.2 0.1"], nc=1)     # coordenada fora de 0..1
