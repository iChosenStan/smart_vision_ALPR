# Etapa 7 — Pipeline de Inferência em Tempo Real

## Objetivo

Integrar os 4 módulos construídos nas Etapas 3-6 em um único fluxo por
frame, com medição real de latência por estágio, fechando o pipeline
completo definido no início do projeto:

```
Entrada → Detecção de Veículos → Detecção da Placa → Recorte → OCR → Classificação → Saída
```

## Decisão Técnica Tomada Nesta Etapa

| Decisão | Escolha | Justificativa |
|---|---|---|
| Entrada do `PlateDetector` e `VehicleClassifier` | **Recorte do veículo** (crop), não o frame inteiro | Escolhida pelo usuário — é a única abordagem que escala para múltiplos veículos por frame, essencial para o objetivo declarado de monitoramento urbano em produção. Aceito o descompasso com o frame completo usado no treino de ambos (Etapas 4 e 6) como limitação conhecida. |

## Limitação Conhecida (Documentada, Não Escondida)

`PlateDetector` (Etapa 4) e `VehicleClassifier` (Etapa 6) foram treinados
em **imagens completas** do UFPR-VeSV (o dataset não fornece bbox de
veículo — ver docs/02_dataset.md). No pipeline integrado, ambos agora
recebem o **recorte do veículo** detectado pelo `VehicleDetector`
(Etapa 3), o que é uma distribuição de entrada diferente da vista em
treino (enquadramento mais apertado, sem o contexto da cena ao redor).
Isso pode causar alguma perda de acurácia em relação às métricas
medidas nos conjuntos de validação das Etapas 4 e 6.

**Mitigação parcial:** uso uma margem de 10% (`crop_margin_ratio`,
configurável) ao redor do bbox do veículo, evitando recortes
excessivamente apertados.

**Melhoria futura recomendada:** se o projeto evoluir para produção,
vale re-treinar o `PlateDetector` e o `VehicleClassifier` usando recortes
de veículo (gerados pelo próprio `VehicleDetector`) em vez de frame
inteiro, eliminando esse descompasso — mas isso depende de re-anotar ou
re-processar os dados de treino, então não fiz isso nesta etapa.

## Arquivos Criados

```
src/utils/
└── geometry.py                     # NOVO — crop_with_margin (extraído do rec_converter, Etapa 5)

src/preprocessing/
└── rec_converter.py                # MODIFICADO — usa geometry.crop_with_margin (remove duplicação)

src/pipeline/
├── schemas.py                      # VehicleResult, FrameResult
└── smart_vision_pipeline.py        # SmartVisionPipeline (orquestrador)

configs/
└── pipeline.yaml                   # crop_margin_ratio, min_vehicle_area

scripts/
└── run_pipeline_demo.py            # demo real (imagem ou vídeo) — requer todos os pesos treinados

tests/
├── test_geometry.py                # 3 testes
└── test_pipeline.py                # 10 testes (componentes fake, sem nenhum peso real)
```

## Refatoração Realizada

Extraí `crop_with_margin` para `src/utils/geometry.py`, compartilhado
entre `rec_converter.py` (Etapa 5) e `smart_vision_pipeline.py` (Etapa 7)
— ambos precisavam exatamente da mesma lógica de recorte com margem.
Segue o mesmo padrão da refatoração da Etapa 4 (`BaseYOLODetector`):
identificar duplicação real entre etapas e extrair um utilitário
compartilhado, em vez de copiar a função. Os testes da Etapa 5
(`test_rec_converter.py`) foram re-executados após a mudança e continuam
passando sem alteração.

## Fluxo Completo

```python
pipeline = SmartVisionPipeline.from_config()  # carrega os 4 sub-modelos
result = pipeline.process_frame(frame)         # FrameResult

for vehicle in result.vehicles:
    print(vehicle.vehicle_detection.class_name)   # "car", "truck", ...
    print(vehicle.plate_text)                     # "ABC1234" ou None
    print(vehicle.classification.make.label)       # "fiat", ...
    print(vehicle.plate_detection_ms, vehicle.ocr_ms, vehicle.classification_ms)

print(result.total_latency_ms, result.fps, result.meets_latency_target)
```

Por veículo detectado (após filtro de área mínima):
1. Recorta o veículo do frame (`crop_with_margin`, margem configurável)
2. Roda `PlateDetector.detect()` no recorte
3. Se placa encontrada, recorta a placa (`crop_best_plate`) e roda `OCRReader.read()`
4. Roda `VehicleClassifier.classify()` no recorte do veículo (independente de ter achado placa)
5. Agrega tudo em um `VehicleResult`, com latência de cada estágio em ms

## Testando a Orquestração sem Nenhum Peso Real

Diferente das etapas anteriores (que usavam mocks pontuais), aqui usei
**duck typing completo**: os testes de `test_pipeline.py` criam
implementações *fake* de `VehicleDetector`, `PlateDetector`, `OCRReader`
e `VehicleClassifier` (mesma interface pública, sem herdar de nada),
permitindo testar 100% da lógica de orquestração — inclusive validando
explicitamente que o recorte do veículo (não o frame inteiro) é o que
chega em `PlateDetector`/`VehicleClassifier`, e que a margem configurada
é aplicada corretamente — sem precisar de nenhum modelo real carregado.

## Como Executar (no Colab, com todos os modelos treinados)

```bash
# Pré-requisitos: Etapas 4, 5 e 6 treinadas (pesos em models/checkpoints/)

# Imagem única
python scripts/run_pipeline_demo.py --image caminho/para/imagem.jpg

# Vídeo (imprime estatísticas de latência/FPS médios ao final)
python scripts/run_pipeline_demo.py --video caminho/para/video.mp4

# Testes (rodam em qualquer ambiente, sem pesos reais — 105 testes no total do projeto)
pytest tests/ -v
```

## Próximos Passos

O pipeline ponta a ponta está funcionalmente completo (Etapas 1-7). Os
próximos passos naturais, conforme o roadmap original do projeto, seriam:

- **Avaliação end-to-end** com métricas reais (OCR accuracy ≥95%,
  classificação ≥80%, latência ≤100ms) rodando no Colab com o dataset
  completo — só é possível fora deste sandbox.
- **Otimização de inferência** (exportar para ONNX/TensorRT, quantização)
  se a latência medida no Colab não bater a meta de 20 FPS.
- **Tracking entre frames** (ex: ByteTrack/DeepSORT) para vídeo, evitando
  reprocessar OCR/classificação a cada frame para o mesmo veículo.

Quer que eu avance para alguma dessas frentes, ou há algum ajuste que
prefira fazer no que já foi construído?
