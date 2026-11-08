"""A ordem das classes tem de ser a do contrato (enum Categoria do openapi.yaml)."""

from visao.classes import CORES, ROTULOS, carregar_classes

ENUM_CONTRATO = ["buraco", "piso_danificado", "obstaculo", "veiculo_na_calcada"]


def test_ordem_igual_ao_contrato():
    assert carregar_classes() == ENUM_CONTRATO


def test_toda_classe_tem_rotulo_e_cor():
    for c in carregar_classes():
        assert c in ROTULOS and c in CORES
