"""Validação de ponta a ponta do serviço publicado (local, Docker ou Railway), pelo HTTP, como a API chama.

    $env:VISAO_URL="https://visao-production-e222.up.railway.app"; $env:VISAO_TOKEN="<token>"
    python scripts/11_testar_servico.py --saida docs/execucao/servico_railway_e2e.txt

Confere:
1. contrato: saúde, Swagger, 401, 400, 413, 422, schema da resposta e charset;
2. detecção em fotos do conjunto de TESTE (dados/dataset/test): as classes que o serviço devolve
   contra as classes do gabarito de cada foto;
3. consistência: a resposta do serviço é a MESMA do modelo ONNX local (modelos/detector.onnx) na
   mesma foto, o que prova que o modelo no ar é o treinado;
4. tempo de resposta e /metrics.
O token nunca é impresso.
"""

import argparse
import io
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone

import httpx
import jsonschema
from PIL import Image

from _comum import DATASET, RAIZ
from visao.classes import carregar_classes
from visao.detector_onnx import DetectorOnnx
from visao.imagens import abrir_rgb

SCHEMA_LISTA = {  # recorte do schema `Deteccao` do openapi.yaml (o mesmo de tests/test_servico.py)
    "type": "array",
    "items": {
        "type": "object", "required": ["classe", "confianca", "caixa"], "additionalProperties": False,
        "properties": {
            "classe": {"enum": ["buraco", "piso_danificado", "obstaculo", "veiculo_na_calcada"]},
            "confianca": {"type": "number", "minimum": 0, "maximum": 1},
            "caixa": {"type": "object", "required": ["x", "y", "largura", "altura"],
                      "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in ("x", "y", "largura", "altura")}},
        },
    },
}


class Relatorio:
    def __init__(self):
        self.linhas, self.falhas = [], 0

    def __call__(self, texto=""):
        print(texto, flush=True)
        self.linhas.append(texto)

    def checar(self, nome, ok, detalhe=""):
        self.falhas += not ok
        self(f"[{'OK' if ok else 'FALHOU'}] {nome}" + (f"  ({detalhe})" if detalhe else ""))


def jpeg(cor=(128, 128, 128), tamanho=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", tamanho, cor).save(buf, "JPEG")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("VISAO_URL", "http://localhost:8001"))
    ap.add_argument("--token", default=os.getenv("VISAO_TOKEN", ""))
    ap.add_argument("--fotos", type=int, default=30, help="fotos do conjunto de teste (sorteadas com semente fixa)")
    ap.add_argument("--saida")
    args = ap.parse_args()
    url, auth = args.url.rstrip("/"), {"Authorization": f"Bearer {args.token}"}
    cli = httpx.Client(timeout=30)
    out = Relatorio()

    def detectar(dados, nome="foto.jpg", tipo="image/jpeg", cab=auth):
        return cli.post(f"{url}/deteccoes", headers=cab, files={"imagem": (nome, dados, tipo)})

    out(f"# Validação de ponta a ponta do serviço de visão: {url}")
    out(f"# {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · token omitido")

    out("\n## 1. Contrato")
    r = cli.get(f"{url}/saude")
    saude = r.json()
    out.checar("GET /saude → 200 com o modelo carregado", r.status_code == 200 and saude.get("status") == "ok",
               f"modelo {saude.get('modelo')}, mAP50 de teste {saude.get('metricas_teste', {}).get('mAP50')}")
    out.checar("GET /docs (Swagger) → 200", cli.get(f"{url}/docs").status_code == 200)
    cinza = jpeg()
    r = detectar(cinza, cab={})
    out.checar("sem token → 401 nao_autenticado", r.status_code == 401 and r.json().get("codigo") == "nao_autenticado")
    out.checar("token errado → 401", detectar(cinza, cab={"Authorization": "Bearer errado"}).status_code == 401)
    r = cli.post(f"{url}/deteccoes", headers=auth, data={"outro": "x"})
    out.checar("sem o campo imagem → 400 requisicao_invalida", r.status_code == 400 and r.json().get("codigo") == "requisicao_invalida")
    r = detectar(b"isto nao e imagem", nome="a.txt", tipo="text/plain")
    out.checar("arquivo de texto → 422 dados_invalidos", r.status_code == 422 and r.json().get("codigo") == "dados_invalidos")
    r = detectar(b"\xff\xd8\xff" + b"0" * 100)
    out.checar("JPEG corrompido → 422", r.status_code == 422)
    r = detectar(b"\xff\xd8\xff" + b"0" * (10 * 1024 * 1024 + 10))
    out.checar("arquivo acima de 10 MB → 413 arquivo_grande", r.status_code == 413 and r.json().get("codigo") == "arquivo_grande")
    r = detectar(cinza)
    out.checar("imagem lisa, sem calçada → 200 e lista vazia", r.status_code == 200 and r.json() == [], r.text[:80])
    out.checar("JSON com charset=utf-8", r.headers.get("content-type") == "application/json; charset=utf-8",
               r.headers.get("content-type"))

    out("\n## 2. Fotos do conjunto de teste (nunca usadas no treino)")
    classes = carregar_classes()
    local = DetectorOnnx(RAIZ / "modelos" / "detector.onnx", classes, saude.get("limiar_confianca", 0.4))
    fotos = sorted((DATASET / "test" / "images").glob("*.jpg"))
    random.Random(42).shuffle(fotos)
    fotos = fotos[: args.fotos]
    tempos, iguais, acertos, com_problema, schema_ok = [], 0, 0, 0, 0
    for f in fotos:
        dados = f.read_bytes()
        inicio = time.perf_counter()
        r = detectar(dados, nome=f.name)
        tempos.append(time.perf_counter() - inicio)
        lista = r.json()
        try:
            jsonschema.validate(lista, SCHEMA_LISTA)
            schema_ok += 1
        except jsonschema.ValidationError:
            pass
        rotulo = DATASET / "test" / "labels" / f"{f.stem}.txt"
        gabarito = {classes[int(l.split()[0])] for l in rotulo.read_text().splitlines() if l.strip()} if rotulo.exists() else set()
        servidas = {d["classe"] for d in lista}
        locais = [d.para_contrato() for d in local.detectar(abrir_rgb(dados))]
        igual = len(lista) == len(locais) and all(
            a["classe"] == b["classe"] and abs(a["confianca"] - b["confianca"]) < 0.01 for a, b in zip(lista, locais))
        iguais += igual
        if gabarito:
            com_problema += 1
            acertos += bool(servidas & gabarito)
        out(f"  {r.status_code} {tempos[-1]:5.2f}s  gabarito={sorted(gabarito)!s:32} serviço={len(lista):2d} det {sorted(servidas)!s:32} "
            f"{'= local' if igual else '≠ LOCAL'}  {f.name[:40]}")
    out.checar(f"{len(fotos)} respostas 200 no schema do contrato", schema_ok == len(fotos), f"{schema_ok}/{len(fotos)}")
    out.checar("resposta do serviço igual à do modelo ONNX local (mesmo modelo no ar)", iguais == len(fotos),
               f"{iguais}/{len(fotos)}")
    out(f"fotos com problema no gabarito em que o serviço achou ao menos uma classe certa: {acertos}/{com_problema}")

    out("\n## 3. Tempo e métricas")
    o = sorted(tempos)
    out(f"tempo por foto (rede + modelo): mediana {statistics.median(o):.2f} s · máximo {o[-1]:.2f} s")
    out.checar("todas abaixo dos 5 s de timeout da API principal", o[-1] < 5, f"máximo {o[-1]:.2f} s")
    r = cli.get(f"{url}/metrics")
    out.checar("GET /metrics → 200 (Prometheus)", r.status_code == 200)
    out("\n".join(l for l in r.text.splitlines() if l and not l.startswith("#"))[:1200])

    out("\n## Resumo")
    out(f"checagens que falharam: {out.falhas}")
    if args.saida:
        texto = "\n".join(out.linhas) + "\n"
        assert not args.token or args.token not in texto
        open(args.saida, "w", encoding="utf-8").write(texto)
    return 1 if out.falhas else 0


if __name__ == "__main__":
    sys.exit(main())
