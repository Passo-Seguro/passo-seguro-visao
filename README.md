# Passo Seguro — Applied Computer Vision

Detector de problemas em calçadas de São Paulo. Um YOLO ajustado com fotos de calçadas
paulistanas classifica cada relato do app e roda em tempo real num vídeo de caminhada.

Ficha Técnica: [README do repositório principal](https://github.com/Passo-Seguro/passo-seguro#ficha-técnica)

## Integrantes
| Nome completo | RM |
|---|---|
| Guilherme Morais | 551981 |
| Júlia Marques | 98680 |
| Matheus Gusmão | 550826 |

## Objetivo

Treinar um detector de objetos para identificar, em fotos e em vídeo em tempo real, quatro
problemas de calçada em São Paulo (buraco, piso danificado, obstáculo e veículo estacionado) e
entregar a classe e a posição ao Passo Seguro para triagem e priorização.

## Resultados

> Preenchido a partir de `resultados/comparacao.md` depois do treino.

| Modelo | mAP50 | mAP50-95 | Precisão | Recall | Latência CPU p95 |
|---|---|---|---|---|---|
| YOLO11n ajustado | – | – | – | – | – |
| YOLO11s ajustado | – | – | – | – | – |

Vídeo da solução: _link do YouTube (não listado)_ · Relatório: [`docs/relatorio.md`](docs/relatorio.md)

## Como reproduzir

O passo a passo completo, treinando na GPU do seu PC, está em **[PASSO_A_PASSO.md](PASSO_A_PASSO.md)**.
Resumo:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -r requirements-dev.txt
python scripts/00_verificar_ambiente.py
python scripts/01_importar_publicos.py        # datasets do Roboflow -> 4 classes
python scripts/02_preparar_fotos_proprias.py  # LGPD: borra rostos e placas, tira GPS
python scripts/03_pre_anotar.py               # pré-anota veículos para o Roboflow/CVAT
python scripts/04_montar_dataset.py           # 70/20/10 por local
python scripts/05_treinar.py --preset final --modelo n
python scripts/06_avaliar.py --todos --coco
python scripts/07_analise_erros.py --nome yolo11n_final_640
python scripts/08_exportar_onnx.py --nome yolo11n_final_640
python scripts/10_tempo_real.py --fonte videos/caminhada.mp4 --salvar resultados/demo.mp4
```

## Serviço `POST /deteccoes`

O contrato é o do [`openapi.yaml`](https://github.com/Passo-Seguro/passo-seguro/blob/main/openapi.yaml)
do repositório principal: multipart com o campo `imagem`, resposta `[{classe, confianca, caixa}]`
da maior para a menor confiança, `caixa` normalizada com canto superior esquerdo, limiar de
confiança 0,4, Bearer token (`VISAO_TOKEN`) e erros `{codigo, mensagem}`.

```bash
docker build -t ghcr.io/passo-seguro/passo-seguro-visao:local .
docker run --rm -p 8001:8000 -e VISAO_TOKEN=troque ghcr.io/passo-seguro/passo-seguro-visao:local
```

| Variável | Padrão | Para quê |
|---|---|---|
| `VISAO_TOKEN` | vazio (sem auth, com aviso no log) | o mesmo `VISAO_TOKEN` da API |
| `LIMIAR_CONFIANCA` | 0.4 | abaixo disso a detecção é descartada |
| `LIMIAR_IOU` | 0.45 | NMS |
| `ONNX_THREADS` | 0 (todas) | ajuste às vCPUs do container |
| `METRICS_TOKEN` | vazio | protege `GET /metrics` (Prometheus) |

`GET /saude` devolve o modelo no ar e as métricas de teste. Os testes (`pytest`) conferem o
contrato com um modelo ONNX falso, sem precisar treinar.

## Estrutura

```
config/      classes.yaml (cópia do contrato) e datasets.yaml (fontes, licenças e remapeamento)
visao/       pacote compartilhado: classes, caixas/IoU/NMS, detector ONNX, imagens sem EXIF
scripts/     00 a 10, na ordem do trabalho, + LGPD (borrar_manual, checar_lgpd)
servico/     FastAPI do POST /deteccoes
tests/       contrato do serviço, detector e utilitários
modelos/     detector.onnx + detector.json (o modelo no ar)
resultados/  tabelas, métricas, matriz de confusão e erros que vão para o PDF
docs/        relatório (PDF) e roteiro do vídeo
```

## Dados, licenças e LGPD

- Datasets públicos e licenças: `resultados/tabela_datasets.md`.
- As fotos próprias são publicadas só depois de borrar rostos e placas e tirar todos os
  metadados (GPS incluído). `scripts/checar_lgpd.py` confere antes do push.
- O Ultralytics é AGPL-3.0, o que este repositório público atende. O serviço em produção não
  usa o Ultralytics, só o ONNX Runtime (MIT).

## Repositórios do projeto
- Principal: [passo-seguro](https://github.com/Passo-Seguro/passo-seguro)
- GenAI: [passo-seguro-genai](https://github.com/Passo-Seguro/passo-seguro-genai) · Dados: [passo-seguro-dados](https://github.com/Passo-Seguro/passo-seguro-dados) · Visão: [passo-seguro-visao](https://github.com/Passo-Seguro/passo-seguro-visao) · RPA: [passo-seguro-rpa](https://github.com/Passo-Seguro/passo-seguro-rpa) · Robô: [passo-seguro-robo](https://github.com/Passo-Seguro/passo-seguro-robo)
