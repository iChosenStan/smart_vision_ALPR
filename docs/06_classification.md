# Etapa 6 — Classificação de Veículo (Multi-tarefa)

## Objetivo

Classificar `type`, `make`, `model` e `color` do veículo a partir da
imagem, usando os atributos já presentes no manifest (Etapa 2) como
rótulos — completando o pipeline: `... → Classificação → Saída`.

## Decisões Técnicas Tomadas Nesta Etapa

| Decisão | Escolha | Justificativa |
|---|---|---|
| Arquitetura | **Multi-tarefa** (1 backbone + 4 cabeças) | Escolhida pelo usuário — 1 forward pass em vez de 4, essencial dado o orçamento de latência já consumido por detecção+OCR nas etapas anteriores. |
| Backbone | **ResNet34** (ImageNet) | Escolhida pelo usuário — prioriza precisão sobre velocidade máxima. |
| Entrada do modelo | **Imagem completa** (não um recorte de veículo) | O UFPR-VeSV não fornece bbox de veículo (mesma limitação já documentada na Etapa 3) — cada imagem já enquadra um único veículo principal, então essa é a abordagem viável. |
| Pesos das 4 losses | `type=1.0, make=1.0, model=1.5, color=0.5` | Calibrados com a distribuição real dos dados (ver abaixo): `model` tem 136 classes de cauda longa (peso maior ajuda o gradiente a não ignorá-la); `color` tem 25% de rótulos `"unknown"` (peso menor evita que esse ruído domine o treino compartilhado). |

## Achado nos Dados que Embasou a Decisão de Pesos

Analisei a distribuição real das 4 colunas-alvo no `annotations.json` completo:

| Atributo | Classes | Observação |
|---|---|---|
| `type` | 14 | Desbalanceado: `car`=13.952 vs `semi-trailer`=64 |
| `color` | 13 | **25% `"unknown"`** (6.327/24.945) — tratado como classe própria, não descartado nem inventado |
| `make` | 26 | Distribuição razoável |
| `model` | 136 | Muito granular — média de ~183 exemplos/classe, cauda longa. **Risco real de não atingir 80% de acurácia** especificamente nesta tarefa; documentado como limitação conhecida, não escondido. |

## Arquivos Criados

```
src/classification/
├── schemas.py                  # AttributePrediction, VehicleClassification
├── label_encoder.py            # LabelEncoders (classe↔índice, fit/save/load)
├── multi_task_model.py         # VehicleMultiTaskNet (ResNet34 + 4 cabeças)
├── transforms.py                # pré-processamento compartilhado treino/inferência
├── dataset.py                   # VehicleClassificationDataset (PyTorch)
└── vehicle_classifier.py       # VehicleClassifier (wrapper de inferência)

configs/
├── classification.yaml                  # inferência
└── training_classification.yaml         # hiperparâmetros de treino

scripts/
└── train_vehicle_classifier.py # loop de treino completo (PyTorch puro)

tests/
├── test_label_encoder.py             # 7 testes
├── test_multi_task_model.py          # 4 testes (forward pass REAL)
├── test_transforms.py                # 4 testes
├── test_classification_dataset.py    # 5 testes (imagens sintéticas)
└── test_vehicle_classifier.py        # 6 testes
```

## Bug Real Encontrado e Corrigido Durante os Testes

A primeira versão de `VehicleMultiTaskNet` usava os nomes das tarefas
(`"type"`, `"make"`, etc.) diretamente como chaves do `nn.ModuleDict`.
Isso quebrou em tempo de execução: `nn.Module` já tem um método nativo
chamado `.type()` (usado para converter o dtype dos parâmetros), e o
PyTorch levanta `KeyError: attribute 'type' already exists` ao tentar
registrar um submódulo com esse nome. Corrigido prefixando as chaves
internas do `ModuleDict` (`"head_type"`, `"head_make"`, ...), mantendo a
API pública (`num_classes`, saída do `forward()`) com os nomes originais.
Isso só foi pego porque os testes desta etapa rodam um forward pass real
(possível aqui porque `torch`/`torchvision` não exigem download quando
`pretrained=False`) — reforça o valor de testes que exercitam a
arquitetura de verdade, não só mocks.

## Fluxo Completo

```
manifest.csv (Etapa 2, colunas type/make/model/color já presentes)
    │
    ▼  scripts/train_vehicle_classifier.py
    │  1. LabelEncoders.fit() no split de TREINO (evita vazamento de classes)
    │  2. Treina ResNet34 + 4 cabeças, salva o melhor checkpoint por val_acc médio
    │
models/checkpoints/vehicle_classifier_best.pt
models/checkpoints/vehicle_classifier_encoders.json
    │
    ▼  VehicleClassifier.from_config()  (configs/classification.yaml)
classify(image) → VehicleClassification(type, make, model, color)
                   cada atributo com label + confidence
```

## Limitação do Ambiente de Desenvolvimento

Mesma limitação de rede das etapas anteriores — o download dos pesos
ImageNet do ResNet34 (`pretrained=True`) está bloqueado neste sandbox.
Diferente das Etapas 3-5, porém, aqui consegui rodar **testes com forward
pass real** (não apenas mocks), usando `pretrained=False` — validando de
fato a arquitetura do modelo, incluindo o bug real relatado acima.

- `test_multi_task_model.py` roda inferência real, sem pesos pré-treinados.
- `test_vehicle_classifier.py` injeta um `VehicleMultiTaskNet` real
  (pequeno) como `_model`, testando toda a mecânica de pré-processamento
  → forward → decodificação, sem depender de pesos treinados.
- **O treino real precisa rodar no Colab**, com as imagens do UFPR-VeSV
  e conexão à internet (para os pesos ImageNet iniciais).

## Como Executar (no Colab, com dataset completo)

```bash
# 1. Gerar manifest (se ainda não gerado)
python scripts/build_dataset_manifest.py

# 2. Treinar (usa datasets/raw/images/ diretamente — sem etapa de conversão)
python scripts/train_vehicle_classifier.py

# 3. Rodar os testes (não dependem do treino real)
pytest tests/ -v
```

## Próximos Passos

**Etapa 7 (proposta):** Pipeline de inferência em tempo real — integra
todos os módulos construídos (`VehicleDetector` → `PlateDetector` →
`OCRReader` → `VehicleClassifier`) em um único fluxo, processando
vídeo/webcam frame a frame e medindo a latência real contra a meta de
≤100ms/20 FPS definida no início do projeto.

Quer que eu avance para a Etapa 7?
