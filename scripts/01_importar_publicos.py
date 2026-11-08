"""Importa os datasets públicos (Roboflow, formato YOLO) e remapeia para as 4 classes.

Lê config/datasets.yaml, e para cada dataset ativo em dados/brutos/publicos/<pasta>/:
- troca o id de classe do dataset pelo id do contrato (ou descarta a caixa);
- converte polígonos em caixas;
- grava em dados/intermediario/publicos/<pasta>/{images,labels}, com o "grupo" de cada imagem
  (para a divisão 70/20/10 não misturar quadros do mesmo vídeo ou do mesmo lugar);
- escreve resultados/tabela_datasets.md (nome, url, licença, imagens, caixas por classe) para o PDF.

    python scripts/01_importar_publicos.py
"""

import csv
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

from _comum import BRUTOS, INTERMEDIARIO, RAIZ, RESULTADOS, titulo
from visao.caixas import ler_rotulo_yolo
from visao.classes import carregar_classes
from visao.imagens import EXTENSOES

PUBLICOS_BRUTOS = BRUTOS / "publicos"
PUBLICOS_SAIDA = INTERMEDIARIO / "publicos"


def nomes_do_dataset(pasta: Path) -> list[str]:
    data = yaml.safe_load((pasta / "data.yaml").read_text(encoding="utf-8"))
    nomes = data["names"]
    if isinstance(nomes, dict):  # {0: 'pothole', 1: ...}
        nomes = [nomes[k] for k in sorted(nomes)]
    return [str(n).strip().lower() for n in nomes]


def grupo_da_imagem(nome_arquivo: str) -> str:
    """Roboflow nomeia como <original>_jpg.rf.<hash>.jpg; quadros de vídeo costumam ser
    <video>_frame123 ou <video>-0042. Agrupa pelo prefixo para manter juntos no split."""
    base = nome_arquivo.split(".rf.")[0]
    base = re.sub(r"_(jpg|jpeg|png)$", "", base, flags=re.I)
    base = re.sub(r"([_\- ]?(frame|img|image)?[_\- ]?\d{2,})$", "", base, flags=re.I)
    return base or nome_arquivo


def importar(cfg: dict, classes: list[str], max_imagens: int, sementes: int = 42) -> dict:
    pasta = PUBLICOS_BRUTOS / cfg["pasta"]
    if not (pasta / "data.yaml").exists():
        raise FileNotFoundError(f"{pasta}/data.yaml não existe. Baixou e descompactou o dataset?")
    nomes = nomes_do_dataset(pasta)
    remap = {str(k).strip().lower(): v for k, v in (cfg.get("remapear") or {}).items()}
    faltando = [n for n in nomes if n not in remap]
    if faltando:
        raise ValueError(
            f"[{cfg['pasta']}] classes sem remapeamento em config/datasets.yaml: {faltando}. "
            "Ligue cada uma a uma das 4 classes ou a 'descartar'."
        )
    for destino in remap.values():
        if destino != "descartar" and destino not in classes:
            raise ValueError(f"[{cfg['pasta']}] destino '{destino}' não é classe do contrato {classes}")
    id_novo = {i: (None if remap[n] == "descartar" else classes.index(remap[n])) for i, n in enumerate(nomes)}

    imagens = [p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES and "images" in p.parts]
    random.Random(sementes).shuffle(imagens)
    imagens = imagens[:max_imagens]

    saida = PUBLICOS_SAIDA / cfg["pasta"]
    if saida.exists():
        shutil.rmtree(saida)
    (saida / "images").mkdir(parents=True)
    (saida / "labels").mkdir(parents=True)
    contagem = Counter()
    linhas_grupo = []
    sem_caixa = 0
    for img in imagens:
        rotulo = img.parent.parent / "labels" / (img.stem + ".txt")
        caixas, cls = ler_rotulo_yolo(rotulo)
        linhas = []
        for c, (cx, cy, w, h) in zip(cls, caixas):
            novo = id_novo.get(c)
            if novo is None:
                continue
            linhas.append(f"{novo} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            contagem[classes[novo]] += 1
        if not linhas:
            sem_caixa += 1  # imagem sem nenhuma das 4 classes: entra como "fundo" (negativo)
        nome = f"{cfg['pasta']}__{img.name}"
        shutil.copy2(img, saida / "images" / nome)
        (saida / "labels" / (Path(nome).stem + ".txt")).write_text("\n".join(linhas), encoding="utf-8")
        linhas_grupo.append({"arquivo": nome, "grupo": f"{cfg['pasta']}::{grupo_da_imagem(img.name)}"})

    with open(saida / "grupos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "grupo"])
        w.writeheader()
        w.writerows(linhas_grupo)
    return {"imagens": len(imagens), "sem_caixa": sem_caixa, "caixas": contagem}


def main() -> int:
    classes = carregar_classes()
    cfg_geral = yaml.safe_load((RAIZ / "config" / "datasets.yaml").read_text(encoding="utf-8"))
    ativos = [d for d in cfg_geral["datasets"] if d.get("ativo")]
    if not ativos:
        print("Nenhum dataset com `ativo: true` em config/datasets.yaml.")
        return 1
    RESULTADOS.mkdir(exist_ok=True)
    linhas = [
        "| Dataset | URL | Licença | Imagens | Sem caixa (fundo) | " + " | ".join(classes) + " |",
        "|---|---|---|---|---|" + "---|" * len(classes),
    ]
    for d in ativos:
        titulo(d["nome"])
        r = importar(d, classes, cfg_geral.get("max_imagens_por_dataset", 1500))
        print(f"{r['imagens']} imagens, {r['sem_caixa']} sem caixa, caixas: {dict(r['caixas'])}")
        linhas.append(
            f"| {d['nome']} | {d['url']} | {d['licenca']} | {r['imagens']} | {r['sem_caixa']} | "
            + " | ".join(str(r["caixas"].get(c, 0)) for c in classes)
            + " |"
        )
    (RESULTADOS / "tabela_datasets.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"\nTabela para o PDF: {RESULTADOS / 'tabela_datasets.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
