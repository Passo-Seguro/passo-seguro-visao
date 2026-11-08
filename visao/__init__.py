"""Pacote compartilhado do detector de problemas de calçada do Passo Seguro.

Usado pelos scripts de treino e avaliação, pelo serviço `POST /deteccoes` e pelo script
de tempo real. Nada aqui depende do Ultralytics: o serviço roda só com ONNX Runtime.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
