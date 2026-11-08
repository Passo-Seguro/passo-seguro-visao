# Passo a passo: do zero ao `/deteccoes` no ar, treinando no seu PC

Tudo roda no seu computador, no VS Code. O treino usa a sua GPU NVIDIA (CUDA). Sem GPU ele
cai para a CPU sozinho, só que mais devagar. Cada passo também é uma **tarefa do VS Code**:
`Ctrl+Shift+P` → **Tasks: Run Task** → escolha pelo número.

Os comandos estão em PowerShell (Windows). No Linux ou no Mac, troque
`.venv\Scripts\Activate.ps1` por `source .venv/bin/activate`.

| Data | Passo | Comando / tarefa | Sai daqui |
|---|---|---|---|
| 08/11 | 1. Ambiente | `00_verificar_ambiente.py` | GPU reconhecida |
| 06–08/11 | 2. Datasets públicos | `01_importar_publicos.py` | `resultados/tabela_datasets.md` |
| 07–08/11 | 3. Fotos próprias + LGPD | `02_preparar_fotos_proprias.py` | fotos borradas, sem GPS |
| 08–09/11 | 4. Anotação | `03_pre_anotar.py` + Roboflow/CVAT | `dados/anotado/proprias/` |
| 09/11 | 5. Dataset 70/20/10 por local | `04_montar_dataset.py` | `resultados/tabela_split.md` |
| 09/11 | 6. Baseline | `05_treinar.py --preset baseline` | pipeline validado |
| 10–11/11 | 7. Treino final nano × small | `05_treinar.py --preset final --modelo n/s` | `modelos/pesos/*.pt` |
| 11/11 | 8. Teste + erros | `06_avaliar.py --todos --coco`, `07_analise_erros.py` | `resultados/comparacao.md`, `erros/` |
| 12/11 | 9. ONNX + serviço | `08_exportar_onnx.py`, `09_latencia.py`, Docker | container do `/deteccoes` |
| 12–13/11 | 10. Tempo real | `10_tempo_real.py` | `resultados/demo.mp4` |
| 14–16/11 | 11. PDF e vídeo | `docs/relatorio.md`, `docs/roteiro-video.md` | entrega |

---

## 1. Ambiente (uma vez)

Pré-requisitos: Python 3.11 ou 3.12, Git, driver NVIDIA atualizado (`nvidia-smi` no terminal
precisa mostrar a placa) e, para o passo 9, Docker Desktop.

```powershell
git clone https://github.com/Passo-Seguro/passo-seguro-visao
cd passo-seguro-visao
code .
```

No terminal do VS Code (`Ctrl+'`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # se der erro de política: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip

# PyTorch COM CUDA primeiro (senão o pip instala a versão só-CPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# cu130 exige driver NVIDIA 580 ou mais novo; com driver antigo use .../whl/cu126.
# (o índice cu124 parou no torch 2.6: o pip troca por uma versão só-CPU mais nova.)
# sem placa NVIDIA:  pip install torch torchvision

pip install -r requirements.txt
pip install -r requirements-dev.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # tem de sair True
python scripts/00_verificar_ambiente.py
python -m pytest -q                  # 21 testes devem passar
```

Escolha o interpretador `.venv` no VS Code (`Ctrl+Shift+P` → **Python: Select Interpreter**).
O `00_verificar_ambiente.py` mostra a GPU, a VRAM e o batch sugerido.

## 2. Datasets públicos (Roboflow Universe)

1. Procure em <https://universe.roboflow.com> por: `pothole`, `sidewalk`, `sidewalk damage`,
   `pavement crack`, `obstacle sidewalk`, `car on sidewalk`. Prefira os de foto de rua feita a
   pé, não de drone ou de painel de carro.
2. Em cada um, **anote a licença** (CC BY 4.0, CC0 e MIT servem). Descarte NC se houver dúvida.
3. **Download Dataset** → formato **YOLOv8** → zip → descompacte em `dados/brutos/publicos/<nome-curto>/`.
4. Preencha o bloco do dataset em `config/datasets.yaml`: url, licença e `remapear` (classe
   do dataset → classe do contrato ou `descartar`), com `ativo: true`.
5. `python scripts/01_importar_publicos.py`

A tabela `resultados/tabela_datasets.md` vai direto para o PDF.

**Sem login no Roboflow:** `python scripts/01a_baixar_publicos.py` baixa e converte quatro
datasets abertos (rachaduras da Ultralytics e três de buracos do Hugging Face), remove imagens
repetidas e rótulos corrompidos e deixa tudo pronto para o `01`. Os blocos já estão ativos em
`config/datasets.yaml`. Eles só cobrem `buraco` e `piso_danificado`, e algumas fotos de buraco
têm marca d'água de banco de imagens: confira a licença antes de publicar imagens deles.

## 3. Fotos próprias e LGPD

Coloque as fotos originais em `dados/brutos/proprias/<trecho>/`, **uma pasta por trecho da
caminhada**, por exemplo `paulista-trianon/` ou `lins-fiap/`. São 300 fotos em 6 trechos ou mais.

```powershell
python scripts/02_preparar_fotos_proprias.py
```

O script borra rostos e placas e grava as fotos sem metadados nenhum (sem GPS). Os detectores
automáticos erram, então abra as fotos com pessoas ou carros perto e borre o que escapou:

```powershell
python scripts/borrar_manual.py dados/intermediario/proprias/<trecho>/<foto>.jpg
```

**Nunca** publique `dados/brutos/`: ele tem o GPS original e está no `.gitignore`.

## 4. Anotação (Roboflow ou CVAT)

```powershell
python scripts/03_pre_anotar.py
```

O script já marca os veículos usando o YOLO do COCO. Na revisão:
- **apague** as caixas de carros parados corretamente na rua (só conta veículo **sobre a calçada**);
- **desenhe** buraco, piso danificado e obstáculo, seguindo `config/classes.yaml`.
  - Bueiro **sem tampa** é `buraco`. Bueiro **tampado** não é nada: é o falso positivo típico.
  - Raiz levantando o piso é `piso_danificado`.

**Roboflow:** crie um projeto *Object Detection* com as 4 classes **nesta ordem**: `buraco`,
`piso_danificado`, `obstaculo`, `veiculo_na_calcada`. Faça **Upload** da pasta
`dados/pre_anotado/roboflow/`. Ao gerar a versão, **não** ligue augmentation nem resize; só
*Auto-Orient*. Exporte em **YOLOv8** e descompacte em `dados/anotado/proprias/`.

**CVAT:** crie a tarefa com as imagens e importe `dados/pre_anotado/cvat_yolo_1.1.zip`. Exporte
em **YOLO 1.1** e descompacte em `dados/anotado/proprias/`.

Não precisa se preocupar com a divisão treino/validação do Roboflow: o passo 5 refaz tudo por trecho.

## 5. Dataset final (70/20/10 por local)

```powershell
python scripts/04_montar_dataset.py
```

Fotos do mesmo trecho ficam todas no mesmo grupo. As fotos próprias e as públicas são divididas
separadamente, para o teste sempre ter fotos de São Paulo. O `--razao-max 3` equilibra as
classes no treino, tirando imagens públicas da classe que sobra; as suas fotos nunca saem. A
tabela vai para `resultados/tabela_split.md`.

## 6. Baseline (09/11)

```powershell
python scripts/05_treinar.py --preset baseline
```

São 20 épocas do nano, só para provar que o pipeline vai de ponta a ponta. Se o resultado sair
perto de zero, o problema está nos rótulos (classe trocada ou caixa fora do lugar), não no modelo.

## 7. Treino final (10–11/11)

```powershell
python scripts/05_treinar.py --preset final --modelo n
python scripts/05_treinar.py --preset final --modelo s
```

São 100 épocas, imagem de 640 e parada antecipada (20 épocas sem melhorar). O batch é escolhido
pela VRAM. Com 4 GB ou menos, use `--batch 8`; sem GPU, `--modelo n --imgsz 512 --batch 8`. Se o
treino cair, retome com `--retomar --nome <execucao>`.

Opcional, para o PDF: `--do-zero` treina a mesma arquitetura sem os pesos do COCO e mostra o
ganho do transfer learning.

## 8. Teste e análise de erros (11/11)

```powershell
python scripts/06_avaliar.py --todos --coco
python scripts/07_analise_erros.py --nome yolo11n_final_640
```

- `resultados/comparacao.md`: nano × small × COCO sem ajuste, com mAP50, mAP50-95, precisão e recall por classe.
- `resultados/<execucao>/confusion_matrix_normalized.png`: vai para o PDF.
- `resultados/<execucao>/erros/`: abra as imagens, preencha a coluna `causa` do `erros.csv`
  (sombra, bueiro_tampado, folhas…) e rode de novo com `--so-resumo`.

**Qual modelo escolher:** o de maior mAP50-95 no teste, **desde que** a latência em CPU (passo 9)
fique bem abaixo de 1 s. Normalmente o nano basta para o serviço.

**Plano B (mAP50 abaixo de 0,5):** junte mais datasets públicos, aumente o augmentation ou reduza
para 3 classes (por exemplo, juntando buraco e piso_danificado). Mudar as classes exige mudar o
contrato no repositório principal: combine com o grupo.

## 9. ONNX e o serviço `/deteccoes` (12/11)

```powershell
python scripts/08_exportar_onnx.py --nome yolo11n_final_640
python scripts/09_latencia.py --threads 2
$env:VISAO_TOKEN="token-local"; uvicorn servico.app:app --port 8001
```

Teste em outro terminal:

```powershell
curl.exe -H "Authorization: Bearer token-local" -F "imagem=@dados/dataset/test/images/<foto>.jpg" http://localhost:8001/deteccoes
```

Container:

```powershell
docker build -t ghcr.io/passo-seguro/passo-seguro-visao:local .
docker run --rm -p 8001:8000 -e VISAO_TOKEN=token-local ghcr.io/passo-seguro/passo-seguro-visao:local
```

**Integração com a API** (repositório principal), a combinar com a Júlia:
- Na API, configure `VISAO_URL=http://<host-do-servico>:8000` e o mesmo `VISAO_TOKEN`.
- No `infra/apps.tf` ainda não há Container App para a visão. É preciso criar um com ingress
  interno, `min_replicas = 1`, porque partida a frio passa dos 5 s de timeout da API, e 1 vCPU / 2 GB.
- Commite `modelos/detector.onnx` e `modelos/detector.json`: o nano tem cerca de 10 MB.

## 10. Tempo real (12–13/11)

```powershell
python scripts/10_tempo_real.py --fonte 0
python scripts/10_tempo_real.py --fonte videos/caminhada.mp4 --salvar resultados/demo.mp4
```

Grave o vídeo da caminhada com o celular na horizontal, na altura do peito, andando devagar. O
FPS aparece no canto. Para mostrar o FPS da GPU, use `--pt yolo11n_final_640`; para o do
serviço (CPU), use o padrão, que é o ONNX.

## 11. PDF e vídeo (14–16/11)

Os modelos estão em [`docs/relatorio.md`](docs/relatorio.md) e
[`docs/roteiro-video.md`](docs/roteiro-video.md). Os números saem dos arquivos de `resultados/`.

Antes do push final:

```powershell
python scripts/checar_lgpd.py      # nenhuma foto com GPS
python -m pytest -q
```
