"""Confere o ambiente: Python, PyTorch, GPU (CUDA), VRAM e sugere o batch de treino.

    python scripts/00_verificar_ambiente.py
"""

import platform
import shutil
import sys

from _comum import titulo


def main() -> int:
    titulo("Sistema")
    print(f"Python {sys.version.split()[0]} em {platform.system()} {platform.release()}")
    if sys.version_info < (3, 10):
        print("ERRO: use Python 3.10 a 3.12.")
        return 1

    titulo("PyTorch")
    try:
        import torch
    except ImportError:
        print("PyTorch não instalado. Veja o passo 1 do PASSO_A_PASSO.md.")
        return 1
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {nome} ({vram:.1f} GB de VRAM), CUDA {torch.version.cuda}")
        if vram < 4:
            sug = "nano com batch 8 (small provavelmente não cabe)"
        elif vram < 8:
            sug = "nano com batch 16, small com batch 8"
        else:
            sug = "nano com batch 32, small com batch 16 (ou batch -1 = automático)"
        print(f"Sugestão de treino: {sug}")
    elif "+cpu" in torch.__version__ or torch.version.cuda is None:
        print("PyTorch SEM suporte a CUDA (versão só-CPU).")
        print("Se você tem placa NVIDIA, reinstale com o índice CUDA (passo 1 do PASSO_A_PASSO.md):")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("Sem GPU o treino funciona, mas o de 100 épocas pode levar horas: use --modelo n e --imgsz 512.")
    else:
        print("PyTorch tem CUDA, mas nenhuma GPU foi encontrada. Confira o driver da NVIDIA (nvidia-smi).")

    titulo("Pacotes")
    for pacote in ("ultralytics", "onnx", "onnxruntime", "cv2", "PIL", "yaml", "pandas"):
        try:
            mod = __import__(pacote)
            print(f"ok  {pacote} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"FALTA {pacote}  ->  pip install -r requirements.txt")

    titulo("Ferramentas")
    print(f"nvidia-smi: {'encontrado' if shutil.which('nvidia-smi') else 'não encontrado'}")
    print(f"docker: {'encontrado' if shutil.which('docker') else 'não encontrado (só precisa para o container)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
