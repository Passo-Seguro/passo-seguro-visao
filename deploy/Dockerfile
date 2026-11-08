# Serviço POST /deteccoes — ONNX Runtime em CPU, sem PyTorch e sem Ultralytics na imagem.
#   docker build -f deploy/Dockerfile -t ghcr.io/passo-seguro/passo-seguro-visao:local .
#   docker run --rm -p 8001:8000 -e VISAO_TOKEN=troque ghcr.io/passo-seguro/passo-seguro-visao:local
# O modelo (modelos/detector.onnx) entra na imagem; nenhum segredo entra (VISAO_TOKEN vem do ambiente).

FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements/servico.txt requirements.txt
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODELO_PATH=/app/modelos/detector.onnx
RUN useradd --create-home --uid 10001 visao
COPY --from=deps /install /usr/local
WORKDIR /app
COPY visao/ visao/
COPY servico/ servico/
COPY config/classes.yaml config/classes.yaml
COPY modelos/detector.onnx modelos/detector.json* modelos/
USER visao
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/saude', timeout=2).status==200 else 1)"
CMD ["uvicorn", "servico.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
