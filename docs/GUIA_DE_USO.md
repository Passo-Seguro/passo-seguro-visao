# Guia de uso: serviço `POST /deteccoes`

O serviço recebe a foto de um relato e devolve os problemas de calçada encontrados, com classe,
confiança e posição. Roda em CPU, só com o ONNX Runtime: não precisa de PyTorch nem de GPU.

- [Subir o serviço](#subir-o-serviço)
- [Contrato](#contrato)
- [Exemplos de chamada](#exemplos-de-chamada)
- [Erros](#erros)
- [Saúde e métricas](#saúde-e-métricas)
- [Configuração](#configuração)
- [Usar o detector sem o serviço](#usar-o-detector-sem-o-serviço)
- [Tempo real (webcam ou vídeo)](#tempo-real-webcam-ou-vídeo)

## Subir o serviço

Precisa de `modelos/detector.onnx` e `modelos/detector.json`, que saem do
`scripts/08_exportar_onnx.py` (ver [PASSO_A_PASSO.md](../PASSO_A_PASSO.md), passo 9).

**Local (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
$env:VISAO_TOKEN="token-local"
uvicorn servico.app:app --port 8001
```

**Docker:**

```powershell
docker build -t ghcr.io/passo-seguro/passo-seguro-visao:local .
docker run --rm -p 8001:8000 -e VISAO_TOKEN=token-local ghcr.io/passo-seguro/passo-seguro-visao:local
```

A documentação interativa (Swagger) fica em <http://localhost:8001/docs>.

## Contrato

É o contrato do `openapi.yaml` do repositório principal, operação `detectarProblemas`.

**Requisição:** `multipart/form-data` com o campo `imagem` (JPEG ou PNG, até 10 MB) e o
cabeçalho `Authorization: Bearer <VISAO_TOKEN>`.

**Resposta 200:** uma lista ordenada da maior para a menor confiança. A lista vazia significa que
nada foi detectado acima do limiar.

```json
[
  {
    "classe": "buraco",
    "confianca": 0.91,
    "caixa": {"x": 0.4219, "y": 0.4479, "largura": 0.1562, "altura": 0.1042}
  }
]
```

| Campo | Tipo | Significado |
|---|---|---|
| `classe` | string | `buraco`, `piso_danificado`, `obstaculo` ou `veiculo_na_calcada` |
| `confianca` | número 0..1 | sempre ≥ `LIMIAR_CONFIANCA` (padrão 0,4) |
| `caixa.x`, `caixa.y` | número 0..1 | canto **superior esquerdo**, como fração da largura e da altura da foto |
| `caixa.largura`, `caixa.altura` | número 0..1 | tamanho como fração da foto |

Para desenhar a caixa numa imagem de `W × H` pixels: `x * W`, `y * H`, `largura * W`, `altura * H`.

A foto é lida com a orientação do EXIF aplicada (como o celular mostra). Os metadados não são
guardados nem registrados no log.

## Exemplos de chamada

**curl (PowerShell ou bash):**

```bash
curl -H "Authorization: Bearer token-local" \
     -F "imagem=@foto.jpg;type=image/jpeg" \
     http://localhost:8001/deteccoes
```

**Python (httpx):**

```python
import httpx

with open("foto.jpg", "rb") as f:
    r = httpx.post(
        "http://localhost:8001/deteccoes",
        headers={"Authorization": "Bearer token-local"},
        files={"imagem": ("foto.jpg", f, "image/jpeg")},
        timeout=5,
    )
r.raise_for_status()
for d in r.json():
    print(d["classe"], d["confianca"], d["caixa"])
```

**JavaScript (navegador ou Node 18+):**

```js
const form = new FormData();
form.append("imagem", arquivo); // um File/Blob JPEG ou PNG
const r = await fetch("http://localhost:8001/deteccoes", {
  method: "POST",
  headers: { Authorization: "Bearer token-local" },
  body: form,
});
const deteccoes = await r.json(); // [] quando nada passa do limiar
```

**Desenhar as caixas (Python + Pillow):**

```python
from PIL import Image, ImageDraw

img = Image.open("foto.jpg")
W, H = img.size
desenho = ImageDraw.Draw(img)
for d in r.json():
    c = d["caixa"]
    x0, y0 = c["x"] * W, c["y"] * H
    desenho.rectangle([x0, y0, x0 + c["largura"] * W, y0 + c["altura"] * H], outline="red", width=4)
    desenho.text((x0 + 4, y0 + 4), f'{d["classe"]} {d["confianca"]:.2f}', fill="red")
img.save("foto_anotada.jpg")
```

## Erros

Todos os erros usam o mesmo formato:

```json
{"codigo": "dados_invalidos", "mensagem": "Não foi possível ler a imagem.",
 "detalhes": [{"campo": "imagem", "problema": "imagem ilegível"}]}
```

| Status | `codigo` | Quando |
|---|---|---|
| 400 | `requisicao_invalida` | falta o campo `imagem` ou o corpo não é multipart |
| 401 | `nao_autenticado` | sem o cabeçalho `Authorization` ou com o token errado |
| 413 | `arquivo_grande` | o arquivo passa de 10 MB (`MAX_BYTES`) |
| 422 | `dados_invalidos` | o tipo não é JPEG/PNG, o arquivo está vazio ou a imagem não abre |
| 503 | `visao_indisponivel` | o modelo não carregou (veja `GET /saude`) |

Do lado da API principal: com 503 ou tempo esgotado (5 s), o relato segue sem a classificação
automática.

## Saúde e métricas

`GET /saude` não pede token. Devolve 200 com o modelo carregado e 503 sem ele:

```json
{"status": "ok", "modelo": "yolo11n_final_640", "classes": ["buraco", "piso_danificado", "obstaculo", "veiculo_na_calcada"],
 "limiar_confianca": 0.4, "metricas_teste": {"mAP50": 0.0, "mAP50_95": 0.0, "precisao": 0.0, "recall": 0.0}}
```

`GET /metrics` responde no formato do Prometheus. Se `METRICS_TOKEN` estiver definido, pede
`Authorization: Bearer <METRICS_TOKEN>`.

```
visao_requisicoes_total{rota="/deteccoes",status="200"} 12
visao_deteccoes_total{classe="buraco"} 7
visao_latencia_segundos{quantil="0.95"} 0.0617
```

## Configuração

| Variável | Padrão | Para quê |
|---|---|---|
| `VISAO_TOKEN` | vazio | token Bearer. Vazio desliga a autenticação e gera um aviso no log |
| `MODELO_PATH` | `modelos/detector.onnx` | caminho do modelo |
| `LIMIAR_CONFIANCA` | 0.4 | abaixo disso a detecção é descartada |
| `LIMIAR_IOU` | 0.45 | NMS: caixas da mesma classe com IoU maior que isso viram uma só |
| `MAX_BYTES` | 10485760 | tamanho máximo do arquivo |
| `ONNX_THREADS` | 0 (todas) | threads do ONNX Runtime. Ajuste às vCPUs do container |
| `METRICS_TOKEN` | vazio | protege o `GET /metrics` |
| `NIVEL_LOG` | INFO | nível do log (JSON, uma linha por evento) |

## Usar o detector sem o serviço

```python
from visao.classes import carregar_classes
from visao.detector_onnx import DetectorOnnx
from visao.imagens import abrir_rgb

det = DetectorOnnx("modelos/detector.onnx", carregar_classes(), limiar_confianca=0.4, limiar_iou=0.45)
rgb = abrir_rgb(open("foto.jpg", "rb").read())   # aplica o EXIF e devolve um array RGB
for d in det.detectar(rgb):
    print(d.para_contrato())
```

## Tempo real (webcam ou vídeo)

```powershell
python scripts/10_tempo_real.py --fonte 0                                    # webcam
python scripts/10_tempo_real.py --fonte videos/caminhada.mp4 --salvar resultados/demo.mp4
python scripts/10_tempo_real.py --fonte videos/caminhada.mp4 --pt yolo11n_final_640   # GPU (PyTorch)
```

O FPS aparece no canto do vídeo. O padrão usa o mesmo ONNX do serviço, em CPU.
