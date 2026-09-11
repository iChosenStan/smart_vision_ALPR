# Etapa 8 — Avaliação e Métricas Finais

## Objetivo

Construir a infraestrutura de avaliação real contra as metas definidas
no início do projeto (OCR ≥95%, Classificação ≥80%, ≤100ms/20 FPS),
separando **avaliação isolada por módulo** (capacidade "crua" de cada
modelo) de **avaliação ponta a ponta** (desempenho real do sistema,
incluindo erros compostos de detecção).

## Decisões Técnicas Tomadas Nesta Etapa

| Decisão | Escolha | Justificativa |
|---|---|---|
| Métrica oficial de OCR Accuracy | **Exact-match** (placa inteira), reportando também acurácia por caractere | Escolhida pelo usuário — reflete o uso real: uma placa com 1 caractere errado é tão inútil quanto totalmente errada para busca/identificação. |
| Métrica oficial de Classificação | **Acurácia simples**, reportando também acurácia balanceada (macro) | Escolhida pelo usuário — é o que a meta original de 80% pede literalmente; a balanceada fica como diagnóstico para não mascarar mau desempenho em classes raras (`type`, `model` — ver Etapa 6). |

## Arquitetura da Avaliação

Três avaliações complementares, todas rodando sobre o **split de TESTE**
do manifest (nunca usado em treino nem em seleção de modelo/early-stopping
nas Etapas 4-6):

1. **OCR isolado** (`evaluate_ocr_isolated`) — recorta a placa usando o
   **bbox ground-truth** (não o `PlateDetector`), isolando a capacidade
   do reconhecedor de texto de erros de detecção.
2. **Classificação isolada** (`evaluate_classifier_isolated`) — roda o
   classificador na imagem completa, mesma distribuição usada no treino
   (Etapa 6).
3. **Ponta a ponta** (`evaluate_pipeline_end_to_end`) — roda a imagem
   bruta pelo `SmartVisionPipeline` (Etapa 7) completo, incluindo
   detecção de veículo e placa reais. Assume 1 veículo principal por
   imagem (premissa do próprio dataset — ver docs/02_dataset.md): quando
   o `VehicleDetector` encontra mais de um veículo na cena, usa o de
   maior bbox como o veículo a comparar com o ground-truth.

Para o detector de placa (Etapa 4), reutilizei o avaliador nativo do
Ultralytics (`model.val()`) em vez de reimplementar cálculo de mAP —
menos código, menos risco de erro (`scripts/evaluate_plate_detector.py`).

## Arquivos Criados

```
src/evaluation/
├── metrics.py          # funções puras: exact_match, character_accuracy,
│                        # classification_accuracy, balanced_accuracy, known_format_rate
├── schemas.py           # OCRMetrics, AttributeMetrics, ClassificationMetrics, EndToEndMetrics
└── evaluator.py         # evaluate_ocr_isolated, evaluate_classifier_isolated,
                          # evaluate_pipeline_end_to_end

scripts/
├── evaluate_plate_detector.py   # mAP via ultralytics model.val()
└── run_evaluation.py            # script mestre — gera relatório .md + .json

tests/
├── test_metrics.py       # 13 testes (funções puras, sem nenhum modelo)
└── test_evaluator.py      # 6 testes (componentes fake + imagens sintéticas)
```

## Por que a Acurácia por Caractere é Sempre ≥ Exact-match

`character_accuracy` é baseada em distância de edição (Levenshtein) —
uma placa com 1 caractere errado em 7 ainda pontua ~86% nessa métrica,
mas 0% no exact-match. Isso é esperado e é exatamente por isso que o
exact-match foi escolhido como métrica oficial: ele não permite que
"quase certo" pareça bom o suficiente numa aplicação onde a placa
precisa estar 100% correta para ser útil.

## Como Executar (no Colab, com todos os modelos treinados)

```bash
# Pré-requisitos: Etapas 4, 5 e 6 treinadas (pesos em models/checkpoints/)

# Avaliação do detector de placa (mAP)
python scripts/evaluate_plate_detector.py

# Avaliação completa (OCR isolado + classificação isolada + end-to-end)
python scripts/run_evaluation.py
# Gera: outputs/reports/evaluation_report.{json,md}

# Testes (rodam em qualquer ambiente, sem pesos reais — 124 testes no total do projeto)
pytest tests/ -v
```

## Limitação do Ambiente de Desenvolvimento

Mesma limitação de todas as etapas anteriores: sem os pesos treinados
reais nem as imagens do UFPR-VeSV, não é possível gerar aqui o
`evaluation_report.md` de verdade. Toda a lógica de cálculo de métricas
foi validada com **funções puras** (testáveis sem nenhum modelo) e
**componentes fake** (duck typing, mesmo padrão da Etapa 7) simulando
cenários conhecidos (ex: 1 acerto em 2, detecção ausente contando como
erro), com os valores esperados calculados manualmente e conferidos nos
testes.

## Estado do Projeto

Com a Etapa 8, o SmartVision ALPR tem:
- Pipeline completo e testado (Etapas 1-7)
- Infraestrutura de avaliação pronta para gerar as métricas finais reais
  assim que rodado no Colab com o dataset completo
- 124 testes automatizados cobrindo toda a lógica que não depende de
  pesos/dados reais

## Próximos Passos (Sugestões, Fora do Roadmap Original)

O roadmap definido no início do projeto (`README.md`) está completo. Como
possíveis frentes adicionais, caso o projeto avance para produção:

- Rodar `scripts/run_evaluation.py` no Colab e usar os números reais para
  decidir se algum modelo precisa de mais épocas, mais dados, ou troca de
  arquitetura (ex: se `model` — 136 classes — não bater 80%, como já era
  esperado desde a Etapa 6).
- Otimização de inferência (ONNX/TensorRT) se a latência real não bater
  20 FPS.
- Tracking entre frames para vídeo (evitar reprocessar o mesmo veículo).
- CI (GitHub Actions) rodando os 124 testes a cada push.

Quer que eu avance para alguma dessas frentes, ou há algo que prefira
ajustar no que já foi construído?
