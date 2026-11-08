"""Configuração do serviço por variáveis de ambiente (nenhum segredo no código ou na imagem)."""

import os
from dataclasses import dataclass

from visao import RAIZ


@dataclass(frozen=True)
class Config:
    modelo_path: str = str(RAIZ / "modelos" / "detector.onnx")
    token: str = ""  # VISAO_TOKEN: o mesmo valor configurado na API (VISAO_TOKEN de lá)
    metrics_token: str = ""
    limiar_confianca: float = 0.4
    limiar_iou: float = 0.45
    max_bytes: int = 10 * 1024 * 1024
    onnx_threads: int = 0
    nivel_log: str = "INFO"

    @classmethod
    def do_ambiente(cls) -> "Config":
        e = os.environ
        return cls(
            modelo_path=e.get("MODELO_PATH", cls.modelo_path),
            token=e.get("VISAO_TOKEN", ""),
            metrics_token=e.get("METRICS_TOKEN", ""),
            limiar_confianca=float(e.get("LIMIAR_CONFIANCA", cls.limiar_confianca)),
            limiar_iou=float(e.get("LIMIAR_IOU", cls.limiar_iou)),
            max_bytes=int(e.get("MAX_BYTES", cls.max_bytes)),
            onnx_threads=int(e.get("ONNX_THREADS", 0)),
            nivel_log=e.get("NIVEL_LOG", "INFO"),
        )
