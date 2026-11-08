# Passo Seguro — Detector de problemas em calçadas de São Paulo

> Modelo do PDF de Applied Computer Vision (até 5 pontos). Preencha o que está entre `[ ]` com
> os números de `resultados/` e exporte para PDF (no VS Code: extensão *Markdown PDF*, ou cole
> no Google Docs). Respeite os limites: resumo com até **250 palavras** e objetivo com até **50**.

**Integrantes:** Guilherme Morais (RM 551981) · Júlia Marques (RM 98680) · Matheus Gusmão (RM 550826)
**Repositório:** https://github.com/Passo-Seguro/passo-seguro-visao · **Vídeo:** [link]

## Resumo (até 250 palavras)

Calçadas danificadas e obstruídas causam quedas e empurram o pedestre para a rua, sobretudo
idosos e pessoas com deficiência. As solicitações de calçada no SP156 cresceram mais de 200%
entre 2020 e 2025, mas reclamar continua lento, e a Prefeitura não identifica automaticamente o
tipo de problema. Este trabalho treina um detector de objetos YOLO para reconhecer quatro
problemas (buraco, piso danificado, obstáculo e veículo na calçada) em fotos enviadas pelo app
Passo Seguro e em vídeo em tempo real. Juntamos [N] imagens de [N] datasets públicos do Roboflow
Universe, remapeadas para as quatro classes, com [N] fotos próprias tiradas em [N] trechos de São
Paulo, anotadas no [Roboflow/CVAT]. A divisão 70/20/10 foi feita por local, e não por foto, para
que a avaliação não fosse otimista. Comparamos as versões nano e small do YOLO11 com transfer
learning a partir do COCO. O melhor modelo, [nano/small], alcançou mAP50 de [0,00] e mAP50-95 de
[0,00] no teste, com recall de [0,00]. A classe mais difícil foi [classe]. Os falsos positivos
típicos foram [sombras, bueiros tampados e folhas]. Exportado para ONNX, o modelo responde em
[000] ms (p95) em CPU, sem GPU, atrás do endpoint `POST /deteccoes`, que a API usa para
classificar cada relato e que a triagem usa para priorizar o encaminhamento à subprefeitura.
[Uma frase de conclusão.]

## Introdução

[Contexto do problema com os números do SP156 e do Infosiga. O Passo Seguro em uma frase. Por que
visão computacional: a Prefeitura já mapeia buracos no asfalto de forma automatizada, enquanto a
calçada fica para trás.]

## Objetivo (até 50 palavras)

Treinar um detector de objetos para identificar, em fotos e vídeo em tempo real, quatro
problemas de calçada em São Paulo (buraco, piso danificado, obstáculo e veículo estacionado) e
entregar classe e posição ao Passo Seguro para triagem e priorização.

## Ferramentas

- **YOLO11 (Ultralytics):** detector de um estágio. Prevê caixa e classe numa passada só, o que
  permite tempo real. Usamos transfer learning a partir dos pesos do COCO. A licença é AGPL-3.0.
- **Roboflow Universe / Roboflow / CVAT:** origem dos datasets públicos e ferramenta de anotação.
- **ONNX e ONNX Runtime:** formato aberto do modelo e motor de inferência em CPU, sem PyTorch na
  produção; a imagem Docker fica enxuta.
- **OpenCV:** captura de webcam e vídeo, desenho das caixas e do FPS; borrar rostos e placas
  (classificadores Haar).
- **FastAPI:** microsserviço `POST /deteccoes`, no contrato OpenAPI do projeto.
- **Hardware de treino:** [GPU, VRAM] · tempo de treino: [min] (de `resultados/*/treino.json`).

## Metodologia

1. **Classes:** definidas no contrato do projeto (`config/classes.yaml`), com exemplos e
   contraexemplos de cada uma.
2. **Datasets públicos:** [tabela de `resultados/tabela_datasets.md`].
3. **Fotos próprias:** [N] fotos em [N] trechos, de manhã e à tarde. Rostos e placas borrados,
   metadados (GPS) removidos.
4. **Anotação:** pré-anotação automática de veículos com o YOLO do COCO, depois revisão manual e
   anotação das outras classes. Critério para casos duvidosos: [ex.: bueiro tampado não é buraco].
5. **Divisão 70/20/10 por local:** [tabela de `resultados/tabela_split.md`]. Os trechos do teste
   não aparecem no treino.
6. **Baseline:** nano, 20 épocas, para validar o pipeline: mAP50 = [0,00].
7. **Treino final:** nano e small, até 100 épocas, imagem de 640, parada antecipada com paciência
   de 20, augmentation (espelho horizontal, rotação de 5°, cor, mosaico).
8. **Avaliação no teste:** mAP50, mAP50-95, precisão e recall por classe, matriz de confusão.
9. **Serviço:** export para ONNX, limiar de confiança de 0,4 e latência medida em CPU.

## Resultados

### Métricas no teste

[Tabela de `resultados/comparacao.md`, com o COCO sem ajuste como linha de base para veículo]

### Matriz de confusão

![Matriz de confusão](../resultados/[execucao]/confusion_matrix_normalized.png)

[Leitura em 2 ou 3 frases: quais classes se confundem e por quê.]

### Acertos e erros

| Acerto | Falso positivo | Falso negativo |
|---|---|---|
| ![](../resultados/[execucao]/erros/acertos/[img]) | ![](../resultados/[execucao]/erros/falsos_positivos/[img]) | ![](../resultados/[execucao]/erros/falsos_negativos/[img]) |

[Causas mais comuns, de `erros/resumo.md`: sombra, bueiro tampado, folhas… e o que faríamos para corrigir.]

### Desempenho em CPU

[Tabela de `resultados/latencia_cpu.md`]

## Conclusão

[O que o modelo resolve bem, onde falha, se os números bastam para a triagem (com confiança
abaixo de 0,5 o relato vai para revisão humana) e os próximos passos: mais fotos de SP, fotos
noturnas, classe de ausência de calçada.]
