"""Baixa os datasets públicos que NÃO exigem login e os deixa no formato que o 01 espera.

Cada um vai para dados/brutos/publicos/<pasta>/{train,valid,test}/{images,labels} + data.yaml:
- crack-seg (Ultralytics, origem Roboflow, CC BY 4.0): polígonos de rachadura -> piso_danificado
- manot/pothole-segmentation (Hugging Face, Roboflow, CC BY 4.0): COCO -> YOLO
- keremberke/pothole-segmentation (Hugging Face, Roboflow, CC BY 4.0): COCO -> YOLO
- rupesh002/pothole-detection-dataset (Hugging Face, MIT): já em YOLO

Os nomes viram <foto de origem>x.rf.<hash>.jpg: o 01 agrupa pelo que vem antes de ".rf.", então
cópias da mesma foto ficam no mesmo split e fotos diferentes não se juntam por terem números
seguidos. Imagens repetidas entre datasets (dHash igual) são descartadas, para não vazar do
treino para o teste.

    python scripts/01a_baixar_publicos.py
    python scripts/01_importar_publicos.py
"""

import io
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

from _comum import BRUTOS, titulo

PUBLICOS = BRUTOS / "publicos"
ZIPS = PUBLICOS / "_zips"
HF = "https://huggingface.co/datasets"
SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}

FONTES = {
    "crack-seg": {"tipo": "yolo_ultralytics",
                  "zips": {"crack-seg.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip"},
                  "nomes": ["crack"]},
    "manot-pothole": {"tipo": "coco",
                      "zips": {f"manot_{s}.zip": f"{HF}/manot/pothole-segmentation/resolve/main/data/{s}.zip"
                               for s in ("train", "valid", "test")}},
    "keremberke-pothole": {"tipo": "coco",
                           "zips": {f"kerem_{s}.zip": f"{HF}/keremberke/pothole-segmentation/resolve/main/data/{s}.zip"
                                    for s in ("train", "valid", "test")}},
    "rupesh-pothole": {"tipo": "yolo_roboflow",
                       "zips": {"rupesh.zip": f"{HF}/rupesh002/pothole-detection-dataset/resolve/main/pothole_dataset.zip"}},
}


def baixar(nome: str, url: str) -> Path:
    destino = ZIPS / nome
    if not destino.exists():
        print(f"baixando {nome}")
        ZIPS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, destino)
    return destino


def nome_novo(original: str) -> str:
    """images218_jpg.rf.34bb.jpg -> images218x.rf.34bb.jpg"""
    origem, _, resto = Path(original).name.partition(".rf.")
    origem = re.sub(r"_(jpg|jpeg|png)$", "", origem, flags=re.I)
    hash_ = Path(resto).stem if resto else "0"
    return f"{origem}x.rf.{hash_}.jpg"


def dhash(dados: bytes) -> int:
    img = Image.open(io.BytesIO(dados)).convert("L").resize((9, 8))
    px = list(img.tobytes())
    return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])


def gravar(pasta: Path, split: str, original: str, dados: bytes, linhas: list[str], vistos: set) -> bool:
    h = dhash(dados)
    if h in vistos:
        return False
    vistos.add(h)
    nome = nome_novo(original)
    img = Image.open(io.BytesIO(dados)).convert("RGB")
    (pasta / split / "images").mkdir(parents=True, exist_ok=True)
    (pasta / split / "labels").mkdir(parents=True, exist_ok=True)
    img.save(pasta / split / "images" / nome, "JPEG", quality=95)
    (pasta / split / "labels" / (Path(nome).stem + ".txt")).write_text("\n".join(linhas), encoding="utf-8")
    return True


def rotulo_valido(linhas: list[str], nc: int) -> bool:
    """Caixa (5 números) ou polígono (classe + pares), classe < nc e coordenadas em 0..1.
    O rupesh002 tem um .txt de rótulo que é uma carta de apresentação: a imagem sai inteira."""
    for linha in linhas:
        p = linha.split()
        try:
            c, xs = int(p[0]), [float(v) for v in p[1:]]
        except (ValueError, IndexError):
            return False
        if not 0 <= c < nc or len(xs) < 4 or len(xs) % 2 or not all(-0.01 <= v <= 1.01 for v in xs):
            return False
    return True


def de_yolo(zips: list[Path], pasta: Path, vistos: set, nc: int) -> tuple[list[str], int, int]:
    """Ultralytics (images/<split>/x.jpg) ou Roboflow (<split>/images/x.jpg), rótulos ao lado."""
    nomes, gravadas, repetidas = None, 0, 0
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            arquivos = set(zf.namelist())
            for a in sorted(arquivos):
                if a.endswith("data.yaml"):
                    import yaml

                    nomes = [str(n) for n in yaml.safe_load(zf.read(a))["names"]]
                if not a.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                partes = a.split("/")
                i = partes.index("images")
                depois = partes[i + 1] if i + 1 < len(partes) - 1 else None  # images/<split>/x.jpg
                split = SPLITS[depois] if depois in SPLITS else SPLITS[partes[i - 1]]  # <split>/images/x.jpg
                rot = "/".join(partes[:i] + ["labels"] + partes[i + 1:])
                rot = str(Path(rot).with_suffix(".txt")).replace("\\", "/")
                linhas = zf.read(rot).decode(errors="replace").split("\n") if rot in arquivos else []
                linhas = [x for x in linhas if x.strip()]
                if not rotulo_valido(linhas, nc):
                    print(f"  rótulo inválido, imagem descartada: {a}")
                    continue
                if gravar(pasta, split, partes[-1], zf.read(a), linhas, vistos):
                    gravadas += 1
                else:
                    repetidas += 1
    return nomes, gravadas, repetidas


def de_coco(zips: list[Path], pasta: Path, vistos: set) -> tuple[list[str], int, int]:
    nomes, gravadas, repetidas = None, 0, 0
    for z in zips:
        split = SPLITS[z.stem.split("_")[-1]]
        with zipfile.ZipFile(z) as zf:
            coco = json.loads(zf.read(next(n for n in zf.namelist() if n.endswith(".json"))))
            cats = sorted(coco["categories"], key=lambda c: c["id"])
            nomes = nomes or [c["name"] for c in cats]
            idx = {c["id"]: i for i, c in enumerate(cats)}
            por_img = {}
            for an in coco["annotations"]:
                por_img.setdefault(an["image_id"], []).append(an)
            caminhos = {Path(n).name: n for n in zf.namelist()}
            for im in coco["images"]:
                W, H = im["width"], im["height"]
                linhas = []
                for an in por_img.get(im["id"], []):
                    x, y, w, h = an["bbox"]
                    if w <= 0 or h <= 0:
                        continue
                    linhas.append(f"{idx[an['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                if gravar(pasta, split, im["file_name"], zf.read(caminhos[Path(im["file_name"]).name]), linhas, vistos):
                    gravadas += 1
                else:
                    repetidas += 1
    return nomes, gravadas, repetidas


def main() -> int:
    vistos: set[int] = set()
    for pasta_nome, f in FONTES.items():
        titulo(pasta_nome)
        zips = [baixar(n, u) for n, u in f["zips"].items()]
        pasta = PUBLICOS / pasta_nome
        if pasta.exists():
            shutil.rmtree(pasta)
        if f["tipo"] == "coco":
            nomes, gravadas, repetidas = de_coco(zips, pasta, vistos)
        else:
            nomes, gravadas, repetidas = de_yolo(zips, pasta, vistos, nc=len(f.get("nomes", ["pothole"])))
        nomes = f.get("nomes") or nomes
        (pasta / "data.yaml").write_text(
            "train: train/images\nval: valid/images\ntest: test/images\n"
            f"nc: {len(nomes)}\nnames: {json.dumps(nomes, ensure_ascii=False)}\n", encoding="utf-8")
        print(f"{gravadas} imagens ({repetidas} repetidas descartadas), classes: {nomes}")
    print("\nPróximo: confira config/datasets.yaml e rode python scripts/01_importar_publicos.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
