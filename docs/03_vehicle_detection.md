# Etapa 3 — Detecção de Veículos

## Objetivo

Detectar veículos em uma imagem/frame usando um modelo genérico
pré-treinado (sem fine-tuning), servindo como primeiro estágio do
pipeline: `Entrada → Detecção de Veículos → ...`.

## Por que sem fine-tuning?

O UFPR-VeSV (ver `docs/02_dataset.md`) fornece apenas o quadrilátero da
**placa** e atributos de classificação do veículo — não há bounding box
do veículo completo. Treinar um detector de veículo exigiria outro
dataset com essa anotação. Por isso, esta etapa usa um detector genérico
pré-treinado em COCO, que já reconhece bem as classes de veículo do
mundo real.

## Decisão Técnica Tomada Nesta Etapa

| Decisão | Escolha | Justificativa |
|---|---|---|
| Arquitetura do detector | **YOLOv8** | Escolhida pelo usuário — prioriza estabilidade e documentação abundante em detrimento dos ganhos marginais de mAP do YOLO11/YOLO26. |
| Tamanho do modelo | **YOLOv8n (nano)** | Escolhido pelo usuário — prioriza velocidade máxima, alinhado à meta de ≤100ms/20 FPS do projeto. |
| Classes mantidas | `car`, `motorcycle`, `bus`, `truck` (IDs COCO 2, 3, 5, 7) | Cobrem os tipos de veículo relevantes ao domínio ALPR; classes irrelevantes (pessoa, animal, objetos) são descartadas na saída. |

> Nota: pesquisei o estado atual do ecossistema Ultralytics antes desta
> decisão e descobri que existe um YOLO26 (lançado em jan/2026, após meu
> corte de conhecimento), focado em inferência NMS-free e latência menor
> em CPU. Ficou registrado como opção para uma futura revisão, caso o
> projeto evolua para dispositivos edge sem GPU.

## Arquivos Criados

```
configs/
└── detection.yaml              # pesos, thresholds, classes-alvo

src/detection/
├── schemas.py                  # VehicleDetection (bbox, confiança, classe)
└── vehicle_detector.py         # VehicleDetector (wrapper sobre ultralytics.YOLO)

scripts/
└── run_vehicle_detection_demo.py   # smoke-test real (requer internet — rodar no Colab)

tests/
└── test_vehicle_detector.py    # 11 testes com mocks (sem depender de download)
```

## Fluxo

1. `VehicleDetector.from_config()` lê `configs/detection.yaml` e instancia
   o wrapper, resolvendo automaticamente `cuda`→`cpu` caso CUDA não esteja
   disponível.
2. `detector.detect(image)` roda `YOLO.predict(...)` com os thresholds
   configurados e retorna apenas `VehicleDetection` das classes definidas
   em `target_classes` — todas as outras 76 classes do COCO (pessoas,
   animais, objetos) são descartadas.
3. Cada `VehicleDetection` expõe `bbox_xyxy`, `confidence`, `class_id`,
   `class_name`, além de propriedades derivadas (`width`, `height`, `area`)
   úteis para filtrar detecções muito pequenas nas próximas etapas.

## Limitação do Ambiente de Desenvolvimento

O sandbox usado neste desenvolvimento tem uma allowlist de rede restrita
e **bloqueia o download automático dos pesos** (`yolov8n.pt`) via GitHub
Releases (erro 403 no redirecionamento). Por isso:

- Os 11 testes automatizados usam **mocks** da saída do Ultralytics
  (`_FakeYOLOModel`/`_FakeBoxes`), validando 100% da lógica de filtragem,
  parsing e tratamento de erro sem depender de rede.
- O smoke-test real (`scripts/run_vehicle_detection_demo.py`) **precisa
  ser rodado no Google Colab** (ou máquina local com internet liberada),
  onde o download automático dos pesos funcionará normalmente.

## Como Executar

```bash
# Testes (rodam em qualquer ambiente, sem internet)
pytest tests/test_vehicle_detector.py -v

# Smoke-test real (Colab / máquina com internet)
python scripts/run_vehicle_detection_demo.py caminho/para/imagem.jpg
```

## Próximos Passos

**Etapa 4 (proposta):** Detecção de placa. Diferente da Etapa 3, aqui
**temos** ground-truth real (`plate_bbox_x1..y2`, calculado a partir dos
corners no manifest da Etapa 2) — então esta etapa envolve fine-tuning de
um detector (provavelmente YOLOv8n novamente, por consistência) no
`datasets/processed/manifest.csv`, recortando a região do veículo
detectado na Etapa 3 e buscando a placa dentro dela.

Antes de implementar, vou apresentar a estratégia de treino (formato de
anotação YOLO a gerar a partir do manifest, hiperparâmetros propostos,
estratégia de data augmentation) para sua validação.

Quer que eu avance para a Etapa 4?
