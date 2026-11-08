# Roteiro do vídeo (até 5 pontos)

Duração sugerida: 3 a 4 minutos. Grave a tela com o OBS e a voz junto. Suba no YouTube como
**não listado**, teste o link numa janela anônima e coloque no README.

| Tempo | O que aparece | O que falar |
|---|---|---|
| 0:00–0:20 | Foto de uma calçada quebrada da caminhada | O problema em uma frase e as 4 classes |
| 0:20–1:30 | `10_tempo_real.py` no vídeo da caminhada, **FPS visível** no canto | Detecção em tempo real; apontar uma caixa de cada classe; dizer o FPS e o hardware |
| 1:30–2:20 | App (PWA) → tirar/enviar foto → relato criado com a **classe detectada** | Integração: o app chama a API, a API chama `POST /deteccoes`; mostrar o JSON de resposta |
| 2:20–3:10 | Uma imagem de `erros/falsos_positivos/` (ex.: sombra ou bueiro tampado) | **Um caso de erro e o que ele ensina**: por que o modelo errou e como corrigir (mais exemplos negativos, anotar bueiro tampado como fundo) |
| 3:10–3:40 | `resultados/comparacao.md` e a matriz de confusão | mAP50 e recall do modelo escolhido, nano × small, latência em CPU |
| 3:40–4:00 | README no GitHub | Repositório, integrantes e link do vídeo |

Checklist antes de gravar:
- [ ] Vídeo anotado gerado com `--salvar resultados/demo.mp4`, sem rostos nem placas legíveis
- [ ] API com `VISAO_URL` apontando para o serviço real (**não** o fake: `GET /health` sem `modo_fake`)
- [ ] Uma foto de teste que o modelo acerta, separada para a demo do app
