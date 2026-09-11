# SmartVision ALPR

Sistema inteligente de reconhecimento automático de placas veiculares (ALPR) com detecção de veículos, detecção de placas, OCR e classificação de veículo (tipo, marca, modelo, cor) em tempo real.

> Projeto de finalidade acadêmica, desenvolvido em etapas incrementais.

## Pipeline

```
Entrada → Detecção de Veículos → Detecção da Placa → Recorte → OCR → Classificação → Saída
```

Saída esperada por frame/veículo:
- bounding boxes (veículo e placa)
- texto da placa + confiança do OCR
- tipo, marca, modelo e cor do veículo

## Metas de Performance

| Métrica | Meta |
|---|---|
| OCR Accuracy | ≥ 95% |
| Classificação | ≥ 80% |
| Latência de inferência | ≤ 100 ms |
| FPS | 20 (quando possível) |

## Stack

Python 3.11 · PyTorch · Ultralytics YOLO · PaddleOCR · OpenCV · TensorFlow/Keras (pontual) · NumPy · Pandas · Matplotlib

Ambiente alvo: Google Colab (GPU NVIDIA), versionado no GitHub.

## Estrutura do Projeto

```
smartvision-alpr/
├── configs/            # Arquivos de configuração (YAML)
├── datasets/
│   ├── raw/             # Dados brutos (não versionado)
│   └── processed/       # Dados processados (não versionado)
├── notebooks/          # Notebooks exploratórios (Colab)
├── scripts/            # Scripts utilitários (CLI, entrypoints)
├── src/
│   ├── detection/       # Detecção de veículos e placas (YOLO)
│   ├── alpr/            # Reconhecimento de texto da placa (OCR)
│   ├── classification/  # Classificação do veículo
│   ├── preprocessing/   # Pré-processamento de imagens/vídeo
│   ├── training/        # Rotinas de treinamento
│   ├── evaluation/      # Métricas e avaliação
│   ├── visualization/   # Visualização de resultados (bboxes, overlays)
│   └── utils/           # Logging, exceções, helpers gerais
├── tests/               # Testes automatizados (pytest)
├── models/
│   ├── pretrained/       # Pesos pré-treinados (não versionado)
│   └── checkpoints/      # Checkpoints de treino (não versionado)
├── outputs/
│   ├── logs/             # Logs de execução
│   ├── predictions/      # Resultados de inferência
│   └── reports/          # Relatórios de avaliação
└── docs/                # Documentação incremental por etapa
```

## Setup

```bash
# 1. Clonar o repositório
git clone <url-do-repo>
cd smartvision-alpr

# 2. Instalar dependências
pip install -r requirements.txt

# 3. PyTorch (instalar separadamente, conforme GPU/CUDA disponível)
# Ver: https://pytorch.org/get-started/locally/

# 4. Rodar os testes
pytest tests/ -v
```

## Status do Desenvolvimento

- [x] Etapa 1 — Estruturação base do projeto
- [x] Etapa 2 — Integração do dataset UFPR-VeSV
- [x] Etapa 3 — Detecção de veículos
- [x] Etapa 4 — Detecção de placas
- [x] Etapa 5 — OCR (ALPR)
- [x] Etapa 6 — Classificação de veículo
- [x] Etapa 7 — Pipeline de inferência em tempo real
- [x] Etapa 8 — Avaliação e métricas finais

Documentação detalhada de cada etapa em [`docs/`](docs/).

## Licença

Projeto acadêmico — uso educacional.
