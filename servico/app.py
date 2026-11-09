"""Microsserviço de visão do Passo Seguro: `POST /deteccoes`.

Contrato: openapi.yaml do repositório principal (passo-seguro), operação `detectarProblemas`.
- multipart/form-data com o campo `imagem` (JPEG ou PNG, até 10 MB);
- 200 com a lista de `{classe, confianca, caixa}`, da maior para a menor confiança
  (lista vazia = nada detectado); `caixa` = {x, y, largura, altura} normalizados 0..1, x e y no
  canto superior esquerdo;
- erros no formato único `{codigo, mensagem, detalhes?}`: 400, 401, 413, 422 e 503.

Rodar localmente:
    uvicorn servico.app:app --port 8001
"""

from __future__ import annotations

import hmac
import json
import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse as _JSONResponse, PlainTextResponse
from PIL import UnidentifiedImageError

from servico.config import Config
from visao.classes import carregar_classes
from visao.detector_onnx import DetectorOnnx, carregar_metadados
from visao.imagens import abrir_rgb


class JSONResponse(_JSONResponse):
    # sem o charset, o Invoke-RestMethod do Windows PowerShell 5.1 lê o corpo como latin-1 ("calÃ§ada")
    media_type = "application/json; charset=utf-8"

log = logging.getLogger("passo_seguro.visao")
TIPOS_ACEITOS = {"image/jpeg", "image/jpg", "image/png"}


class Metricas:
    """Contadores mínimos no formato do Prometheus (sem dependência extra)."""

    def __init__(self):
        self.requisicoes: dict[tuple[str, int], int] = {}
        self.latencias: list[float] = []
        self.deteccoes: dict[str, int] = {}

    def registrar(self, status: int, segundos: float, classes: list[str]):
        self.requisicoes[("/deteccoes", status)] = self.requisicoes.get(("/deteccoes", status), 0) + 1
        self.latencias.append(segundos)
        self.latencias = self.latencias[-2000:]
        for c in classes:
            self.deteccoes[c] = self.deteccoes.get(c, 0) + 1

    def texto(self) -> str:
        linhas = ["# TYPE visao_requisicoes_total counter"]
        for (rota, status), n in sorted(self.requisicoes.items()):
            linhas.append(f'visao_requisicoes_total{{rota="{rota}",status="{status}"}} {n}')
        linhas.append("# TYPE visao_deteccoes_total counter")
        for c, n in sorted(self.deteccoes.items()):
            linhas.append(f'visao_deteccoes_total{{classe="{c}"}} {n}')
        if self.latencias:
            ordenadas = sorted(self.latencias)
            for q in (0.5, 0.95, 0.99):
                v = ordenadas[min(int(q * len(ordenadas)), len(ordenadas) - 1)]
                linhas.append(f'visao_latencia_segundos{{quantil="{q}"}} {v:.4f}')
        return "\n".join(linhas) + "\n"


def erro(status: int, codigo: str, mensagem: str, detalhes: list | None = None) -> JSONResponse:
    corpo = {"codigo": codigo, "mensagem": mensagem}
    if detalhes:
        corpo["detalhes"] = detalhes
    return JSONResponse(status_code=status, content=corpo)


def criar_app(config: Config | None = None, detector=None) -> FastAPI:
    config = config or Config.do_ambiente()
    estado: dict = {"detector": detector, "meta": {}, "erro_modelo": None}
    metricas = Metricas()

    @asynccontextmanager
    async def ciclo(app: FastAPI):
        # stdout: o Railway (e outros) marca tudo o que sai no stderr como erro
        logging.basicConfig(level=config.nivel_log, format="%(message)s", stream=sys.stdout)
        if estado["detector"] is None:
            try:
                import onnxruntime as ort

                opcoes = ort.SessionOptions()
                if config.onnx_threads:
                    opcoes.intra_op_num_threads = config.onnx_threads
                sessao = ort.InferenceSession(config.modelo_path, sess_options=opcoes,
                                              providers=["CPUExecutionProvider"])
                estado["detector"] = DetectorOnnx(config.modelo_path, carregar_classes(),
                                                  config.limiar_confianca, config.limiar_iou, sessao=sessao)
                estado["meta"] = carregar_metadados(config.modelo_path)
                log.info(json.dumps({"evento": "modelo_carregado", "modelo": estado["meta"].get("nome")}))
            except Exception as e:  # o serviço sobe mesmo assim e responde 503
                estado["erro_modelo"] = type(e).__name__
                log.error(json.dumps({"evento": "modelo_nao_carregou", "tipo_erro": type(e).__name__}))
        if not config.token:
            log.warning(json.dumps({"evento": "sem_token", "aviso": "VISAO_TOKEN vazio: /deteccoes sem autenticação"}))
        yield

    app = FastAPI(title="Passo Seguro — Visão", version="1.0.0", lifespan=ciclo,
                  docs_url="/docs", redoc_url=None, default_response_class=JSONResponse)

    @app.exception_handler(RequestValidationError)
    async def _validacao(request: Request, exc: RequestValidationError):
        detalhes = [{"campo": str(e["loc"][-1]), "problema": e["msg"]} for e in exc.errors()]
        return erro(400, "requisicao_invalida", "A requisição está malformada.", detalhes)

    def autorizado(request: Request) -> bool:
        if not config.token:
            return True
        cab = request.headers.get("authorization", "")
        if not cab.lower().startswith("bearer "):
            return False
        return hmac.compare_digest(cab[7:].strip().encode(), config.token.encode())

    @app.get("/saude")
    async def saude():
        ok = estado["detector"] is not None
        corpo = {"status": "ok" if ok else "sem_modelo", "modelo": estado["meta"].get("nome"),
                 "classes": carregar_classes(), "limiar_confianca": config.limiar_confianca,
                 "metricas_teste": estado["meta"].get("metricas_teste")}
        if not ok:
            corpo["erro"] = estado["erro_modelo"]
        return JSONResponse(status_code=200 if ok else 503, content=corpo)

    @app.get("/metrics")
    async def metrics(request: Request):
        if config.metrics_token and request.headers.get("authorization", "") != f"Bearer {config.metrics_token}":
            return erro(401, "nao_autenticado", "Token ausente ou inválido.")
        return PlainTextResponse(metricas.texto())

    @app.post("/deteccoes")
    async def deteccoes(request: Request, imagem: UploadFile):
        inicio = time.perf_counter()
        if not autorizado(request):
            metricas.registrar(401, time.perf_counter() - inicio, [])
            return erro(401, "nao_autenticado", "Token ausente ou inválido.")
        dados = await imagem.read(config.max_bytes + 1)
        if len(dados) > config.max_bytes:
            metricas.registrar(413, time.perf_counter() - inicio, [])
            return erro(413, "arquivo_grande", "O arquivo enviado passa do tamanho máximo de 10 MB.")
        tipo = (imagem.content_type or "").lower()
        if tipo and tipo not in TIPOS_ACEITOS and tipo != "application/octet-stream":
            metricas.registrar(422, time.perf_counter() - inicio, [])
            return erro(422, "dados_invalidos", "A imagem deve ser JPEG ou PNG.",
                        [{"campo": "imagem", "problema": f"tipo {tipo} não aceito"}])
        if not dados:
            metricas.registrar(422, time.perf_counter() - inicio, [])
            return erro(422, "dados_invalidos", "A imagem está vazia.", [{"campo": "imagem", "problema": "vazia"}])
        if estado["detector"] is None:
            metricas.registrar(503, time.perf_counter() - inicio, [])
            return erro(503, "visao_indisponivel", "O serviço de visão está indisponível no momento.")
        try:
            rgb = abrir_rgb(dados)
        except (UnidentifiedImageError, OSError, ValueError):
            metricas.registrar(422, time.perf_counter() - inicio, [])
            return erro(422, "dados_invalidos", "Não foi possível ler a imagem.",
                        [{"campo": "imagem", "problema": "imagem ilegível"}])
        lista = estado["detector"].detectar(rgb)
        corpo = [d.para_contrato() for d in lista]
        duracao = time.perf_counter() - inicio
        metricas.registrar(200, duracao, [d["classe"] for d in corpo])
        # log sem imagem e sem nada do cidadão: só tamanho, quantidade e tempo
        log.info(json.dumps({"evento": "deteccao", "bytes": len(dados), "largura": rgb.shape[1],
                             "altura": rgb.shape[0], "n": len(corpo), "ms": round(duracao * 1000, 1)}))
        return corpo

    return app


app = criar_app()
