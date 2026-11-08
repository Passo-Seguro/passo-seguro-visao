# Registro de execução e testes — 08/11/2026

O que foi rodado de verdade nesta máquina, com as saídas salvas nesta pasta.

## Ambiente

| Item | Versão |
|---|---|
| Sistema | Windows 11 Pro |
| Python | 3.12.10 (venv em `.venv`) |
| GPU | NVIDIA GeForce RTX 4080, 16 GB, driver 617.14, CUDA 13.0 |
| PyTorch | 2.14.1+cu130 (torchvision 0.29.1+cu130) |
| Ultralytics | 8.4.208 |
| ONNX Runtime | 1.31.1 |

Saída completa: [`00_ambiente.txt`](00_ambiente.txt). Batch sugerido: nano 32, small 16.

**Problema encontrado no guia e corrigido:** seguindo o passo 1 como estava (índice `cu124`), o
`pip install -r requirements.txt` trocou o PyTorch por uma versão mais nova **só-CPU**, sem
nenhum erro. O `cu124` para no torch 2.6. O passo 1 agora usa `cu130` e tem uma linha que confere
`torch.cuda.is_available()`.

## Testes automatizados

`python -m pytest -v` → **18 passaram, 0 falharam** ([`pytest.txt`](pytest.txt)).

| Arquivo | O que cobre |
|---|---|
| `test_caixas.py` | conversão centro → canto, normalização 0..1, IoU, NMS, pareamento de erros |
| `test_classes.py` | ordem das classes igual à do contrato, rótulo e cor |
| `test_detector.py` | o detector ONNX desfaz o letterbox |
| `test_servico.py` | 200 no contrato, PNG, 401, 413, 422, 400, 503 sem modelo, saúde e métricas, lista vazia |

Os testes usam um ONNX falso com a mesma interface do YOLO exportado. Não é preciso treinar.

## Serviço `/deteccoes` rodando

O `uvicorn servico.app:app` subiu com esse ONNX falso, que ficou fora do repositório, e foi
chamado com curl ([`servico_deteccoes_local.txt`](servico_deteccoes_local.txt)):

| Chamada | Esperado | Obtido |
|---|---|---|
| `GET /saude` | 200, 4 classes, limiar 0,4 | 200 |
| `POST` sem token | 401 `nao_autenticado` | 401 |
| `POST` JPEG 800×600 | caixas acima de 0,4, sem duplicata, normalizadas | `buraco` 0,91 e `piso_danificado` 0,70; o obstáculo de 0,30 foi filtrado e a caixa sobreposta saiu no NMS |
| `POST` arquivo que não é imagem | 422 `dados_invalidos` | 422 |
| `GET /metrics` | contadores e latência | p50 de 29 ms (modelo falso) |
| log | sem a imagem | só bytes, tamanho, número de caixas e ms |

Isso prova o contrato e o pós-processamento. **Não é** uma medida do modelo de verdade.

## O que falta, e por quê não foi rodado aqui

Não há dados no repositório: `dados/` está vazio e nenhum dataset está `ativo` no
`config/datasets.yaml`. Daqui para frente, os passos precisam de uma pessoa:

| Passo | O que precisa |
|---|---|
| 2. Datasets públicos | escolher no Roboflow Universe, conferir a licença e baixar em YOLOv8 |
| 3. Fotos próprias | ~300 fotos em 6 trechos ou mais, em `dados/brutos/proprias/<trecho>/` |
| 4. Anotação | revisar as pré-anotações e desenhar as caixas no Roboflow ou no CVAT |
| 5 a 10 | rodam sozinhos depois disso. A RTX 4080 aguenta o treino final do nano e do small |
| Docker | o `Dockerfile` copia `modelos/detector.onnx`, que só existe depois do passo 9 (`08_exportar_onnx.py`) |

Para repetir esta execução, siga o [PASSO_A_PASSO.md](../../PASSO_A_PASSO.md), passo 1.
