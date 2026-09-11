# Etapa 4 — Detecção de Placa (Fine-tuning Real)

## Objetivo

Treinar um detector de placa (YOLOv8n) usando o ground-truth real do
UFPR-VeSV (`plate_bbox_x1..y2`, calculado na Etapa 2), permitindo
localizar a placa com precisão dentro da imagem — entrada necessária
para o OCR na Etapa 5.

## Decisões Técnicas Tomadas Nesta Etapa

| Decisão | Escolha | Justificativa |
|---|---|---|
| Épocas de treino | **50** | Escolhida pelo usuário — prioriza velocidade de iteração. |
| Tamanho de imagem (imgsz) | **640** | Escolhida pelo usuário — padrão Ultralytics, mais rápido que 960/1280. |
| Formato de bbox usado | Axis-aligned (derivado do quadrilátero original) | O UFPR-VeSV anota a placa como quadrilátero de 4 pontos (pode estar levemente rotacionado); YOLO espera retângulo alinhado aos eixos. Usar o axis-aligned é a abordagem padrão — o recorte resultante pode incluir uma margem extra de fundo em placas fotografadas em ângulo, o que não prejudica o OCR na próxima etapa. |
| Batch size | Auto (`-1`) | Deixado automático — Ultralytics escolhe conforme a memória de GPU disponível no Colab, evitando OOM sem necessidade de ajuste manual. |
| Early stopping | `patience=15` | Evita overfitting/desperdício de tempo se o val loss estagnar antes das 50 épocas completarem. |

## Refatoração Realizada

Extraí `src/detection/base_yolo_detector.py` (`BaseYOLODetector`) a partir
da lógica já existente em `VehicleDetector` (Etapa 3), pois `PlateDetector`
precisa exatamente da mesma lógica de carregamento de modelo, resolução de
device (cuda→cpu) e parsing/filtragem de boxes — evitando duplicação
(SOLID, DRY). `VehicleDetector` foi atualizado para herdar dessa base;
**sua API pública não mudou** e os 10 testes da Etapa 3 continuam
passando sem alteração.

## Arquivos Criados/Modificados

```
src/detection/
└── base_yolo_detector.py           # NOVO — lógica compartilhada (refatoração)
└── vehicle_detector.py             # MODIFICADO — agora herda de BaseYOLODetector

src/preprocessing/
└── yolo_converter.py               # manifest → estrutura de dataset YOLO

src/alpr/
├── schemas.py                      # PlateDetection
└── plate_detector.py               # PlateDetector (herda de BaseYOLODetector)

configs/
├── plate_detection.yaml            # inferência (pesos fine-tunados, thresholds)
└── training_plate_detection.yaml   # hiperparâmetros de treino

scripts/
├── build_plate_detection_dataset.py    # manifest.csv → dataset YOLO
└── train_plate_detector.py             # fine-tuning propriamente dito

tests/
├── test_yolo_converter.py          # 7 testes (imagens sintéticas)
└── test_plate_detector.py          # 8 testes (mocks, sem pesos reais)
```

## Fluxo Completo (do manifest ao modelo treinado)

```
manifest.csv (Etapa 2)
    │
    ▼  scripts/build_plate_detection_dataset.py
datasets/processed/plate_detection/{images,labels}/{train,val,test} + data.yaml
    │
    ▼  scripts/train_plate_detector.py
outputs/training_runs/plate_detector_yolov8n/weights/best.pt
    │
    ▼  (copiado automaticamente pelo script de treino)
models/checkpoints/plate_detector_best.pt
    │
    ▼  PlateDetector.from_config()  (configs/plate_detection.yaml)
detect(image) → List[PlateDetection] → crop_best_plate() → recorte pronto p/ OCR
```

## Limitação do Ambiente de Desenvolvimento

Igual à Etapa 3: este sandbox não tem acesso às imagens reais nem
consegue baixar pesos via internet. Por isso:

- `test_yolo_converter.py` usa **imagens sintéticas geradas em memória**
  (via OpenCV) para validar 100% da lógica de conversão — cópia de
  arquivo, normalização de bbox, geração de `data.yaml`, tratamento de
  imagem ausente/ilegível.
- `test_plate_detector.py` usa os mesmos **mocks** da Etapa 3 para validar
  a lógica de inferência sem depender de pesos reais.
- `PlateDetector` **exige que os pesos existam em disco** (diferente do
  `VehicleDetector`, que baixa automaticamente) — testado explicitamente
  em `test_load_model_raises_when_weights_file_does_not_exist`.
- **O treino real (`scripts/train_plate_detector.py`) precisa ser
  executado no Colab**, com as imagens do UFPR-VeSV em
  `datasets/raw/images/` e GPU disponível.

## Como Executar (no Colab, com dataset completo)

```bash
# 1. Gerar manifest (se ainda não gerado)
python scripts/build_dataset_manifest.py

# 2. Converter para formato YOLO
python scripts/build_plate_detection_dataset.py

# 3. Treinar (50 épocas, imgsz=640 — ~alguns minutos em GPU T4)
python scripts/train_plate_detector.py

# 4. Rodar os testes (não dependem do treino real)
pytest tests/ -v
```

## Próximos Passos

**Etapa 5 (proposta):** OCR — reconhecimento do texto da placa a partir
do recorte gerado por `PlateDetector.crop_best_plate()`. Conforme
definido no início do projeto, a ferramenta desejada é o **PaddleOCR**.
Antes de implementar, vou apresentar a estratégia (PaddleOCR pré-treinado
vs. fine-tuning de um reconhecedor específico para placas brasileiras/
Mercosul, considerando os dois padrões presentes no UFPR-VeSV) para sua
validação.

Quer que eu avance para a Etapa 5?
