"""Monta o dataset final em dados/dataset/ com divisão 70/20/10 POR LOCAL, não por foto.

Fontes:
- Fotos próprias anotadas: dados/anotado/proprias/ (export YOLOv8 do Roboflow ou YOLO 1.1 do
  CVAT, qualquer estrutura com images/ e labels/ lado a lado). O trecho vem do prefixo do nome
  do arquivo, `<trecho>__...`, colocado pelo 02_preparar_fotos_proprias.py. A divisão que o
  Roboflow fez é IGNORADA: refazemos aqui por trecho.
- Públicos: dados/intermediario/publicos/<pasta>/ (do 01_importar_publicos.py), agrupados por
  sequência de origem (grupos.csv).

Regras:
- Fotos do mesmo trecho (ou da mesma sequência de um dataset público) ficam todas no mesmo
  grupo: treino, validação ou teste. Senão a validação fica otimista.
- Próprias e públicas são divididas separadamente, para o TESTE sempre ter fotos de São Paulo.
- `--razao-max` equilibra as classes no TREINO: enquanto a classe mais comum tiver mais que
  razao × a menos comum, tira imagens PÚBLICAS de treino que só têm a classe dominante.
  As fotos próprias nunca são descartadas.

    python scripts/04_montar_dataset.py [--razao-max 3] [--semente 42]
"""

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from _comum import ANOTADO, DATASET, INTERMEDIARIO, RESULTADOS, titulo
from visao.caixas import ler_rotulo_yolo
from visao.classes import carregar_classes
from visao.imagens import EXTENSOES

PROPORCOES = {"train": 0.7, "val": 0.2, "test": 0.1}


def achar_rotulo(img: Path) -> Path | None:
    for cand in (img.parent.parent / "labels" / f"{img.stem}.txt", img.with_suffix(".txt")):
        if cand.exists():
            return cand
    return None


def coletar_proprias() -> list[dict]:
    itens = []
    base = ANOTADO / "proprias"
    for img in sorted(base.rglob("*")):
        if img.suffix.lower() not in EXTENSOES:
            continue
        if "__" not in img.name:
            print(f"AVISO: {img.name} sem prefixo '<trecho>__' — ignorada (passou pelo 02?)")
            continue
        rotulo = achar_rotulo(img)
        if rotulo is None:
            print(f"AVISO: {img.name} sem arquivo de rótulo — ignorada")
            continue
        itens.append({"img": img, "rotulo": rotulo, "grupo": "sp::" + img.name.split("__")[0], "origem": "propria"})
    # o Roboflow pode exportar a mesma foto em mais de um split: fica uma só
    vistos, unicos = set(), []
    for it in itens:
        chave = it["img"].name.split(".rf.")[0]
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(it)
    return unicos


def coletar_publicos() -> list[dict]:
    itens = []
    for pasta in sorted((INTERMEDIARIO / "publicos").glob("*")):
        grupos = {}
        arq = pasta / "grupos.csv"
        if arq.exists():
            with open(arq, encoding="utf-8") as f:
                grupos = {r["arquivo"]: r["grupo"] for r in csv.DictReader(f)}
        for img in sorted((pasta / "images").glob("*")):
            itens.append({
                "img": img,
                "rotulo": pasta / "labels" / f"{img.stem}.txt",
                "grupo": grupos.get(img.name, f"{pasta.name}::{img.stem}"),
                "origem": pasta.name,
            })
    return itens


def dividir_por_grupo(itens: list[dict], semente: int) -> None:
    """Atribui `split` a cada item: grupos inteiros, perseguindo 70/20/10 em número de imagens."""
    por_grupo = defaultdict(list)
    for it in itens:
        por_grupo[it["grupo"]].append(it)
    grupos = list(por_grupo)
    random.Random(semente).shuffle(grupos)
    grupos.sort(key=lambda g: -len(por_grupo[g]))  # maiores primeiro: encaixe guloso
    total = len(itens)
    contagem = Counter()
    for g in grupos:
        # vai para o split mais longe da sua meta (em proporção)
        falta = {s: PROPORCOES[s] * total - contagem[s] for s in PROPORCOES}
        if len(grupos) >= 3:
            for s in ("test", "val"):  # garante ao menos um grupo em val e test
                if contagem[s] == 0 and falta[s] > 0:
                    falta[s] = float("inf")
        destino = max(falta, key=falta.get)
        for it in por_grupo[g]:
            it["split"] = destino
        contagem[destino] += len(por_grupo[g])


def classes_do_item(it: dict) -> list[int]:
    return ler_rotulo_yolo(it["rotulo"])[1]


def equilibrar(itens: list[dict], classes: list[str], razao_max: float, semente: int) -> int:
    treino = [it for it in itens if it["split"] == "train"]
    contagem = Counter(c for it in treino for c in classes_do_item(it))
    if not contagem:
        return 0
    removidos = 0
    candidatos = [it for it in treino if it["origem"] != "propria"]
    random.Random(semente).shuffle(candidatos)
    for it in candidatos:
        presentes = [c for c in range(len(classes)) if contagem.get(c, 0) > 0]
        if len(presentes) < 2:
            break
        dominante = max(presentes, key=lambda c: contagem[c])
        menor = min(presentes, key=lambda c: contagem[c])
        if contagem[dominante] <= razao_max * contagem[menor]:
            break
        cls = classes_do_item(it)
        if cls and set(cls) == {dominante}:
            it["split"] = None
            for c in cls:
                contagem[c] -= 1
            removidos += 1
    return removidos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--razao-max", type=float, default=3.0)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--max-fundo", type=float, default=0.1, help="fração máx. de imagens sem caixa no treino")
    args = ap.parse_args()

    classes = carregar_classes()
    proprias, publicos = coletar_proprias(), coletar_publicos()
    if not proprias and not publicos:
        print("Nada para montar. Rode 01 (públicos) e anote as fotos próprias em dados/anotado/proprias/.")
        return 1
    if not proprias:
        print("AVISO: sem fotos próprias anotadas. O teste não vai provar nada sobre São Paulo.")
    dividir_por_grupo(proprias, args.semente)
    dividir_por_grupo(publicos, args.semente)
    itens = proprias + publicos

    removidos = equilibrar(itens, classes, args.razao_max, args.semente)
    # limita imagens de fundo (sem caixa) no treino
    fundo = [it for it in itens if it["split"] == "train" and not classes_do_item(it)]
    limite = int(args.max_fundo * sum(1 for it in itens if it["split"] == "train"))
    for it in fundo[limite:]:
        if it["origem"] != "propria":
            it["split"] = None

    if DATASET.exists():
        shutil.rmtree(DATASET)
    estat = defaultdict(Counter)
    grupos_por_split = defaultdict(set)
    for it in itens:
        s = it["split"]
        if s is None:
            continue
        (DATASET / s / "images").mkdir(parents=True, exist_ok=True)
        (DATASET / s / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(it["img"], DATASET / s / "images" / it["img"].name)
        shutil.copy2(it["rotulo"], DATASET / s / "labels" / f"{it['img'].stem}.txt")
        estat[s]["imagens"] += 1
        estat[s]["proprias" if it["origem"] == "propria" else "publicas"] += 1
        for c in classes_do_item(it):
            estat[s][classes[c]] += 1
        if it["origem"] == "propria":
            grupos_por_split[s].add(it["grupo"].split("::")[1])

    (DATASET / "data.yaml").write_text(
        yaml.safe_dump({"path": str(DATASET.resolve()), "train": "train/images", "val": "val/images",
                        "test": "test/images", "names": {i: n for i, n in enumerate(classes)}},
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    titulo("Dataset montado")
    cab = ["split", "imagens", "proprias", "publicas", *classes, "trechos de SP"]
    linhas = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for s in ("train", "val", "test"):
        e = estat[s]
        linhas.append("| " + " | ".join([s, *(str(e.get(k, 0)) for k in cab[1:-1]),
                                         ", ".join(sorted(grupos_por_split[s])) or "-"]) + " |")
    tabela = "\n".join(linhas)
    print(tabela)
    print(f"\nImagens públicas tiradas do treino para equilibrar classes: {removidos}")
    vazias = [c for c in classes if estat["train"].get(c, 0) == 0]
    if vazias:
        print(f"ATENÇÃO: classes sem nenhuma caixa no treino: {vazias}")
    poucas = [c for c in classes if 0 < estat["val"].get(c, 0) < 20]
    if poucas:
        print(f"AVISO: menos de 20 caixas na validação para {poucas}; a métrica dessas classes vai oscilar.")
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "tabela_split.md").write_text(
        tabela + f"\n\nImagens públicas removidas do treino para equilíbrio: {removidos}\n", encoding="utf-8")
    print(f"\ndata.yaml: {DATASET / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
