# Etapa 5 — OCR (Reconhecimento de Texto da Placa)

## Objetivo

Reconhecer o texto da placa a partir do recorte gerado na Etapa 4
(`PlateDetector.crop_best_plate()`), completando o estágio ALPR do
pipeline: `... → Detecção da Placa → Recorte → OCR → ...`.

## Decisões Técnicas Tomadas Nesta Etapa

| Decisão | Escolha | Justificativa |
|---|---|---|
| Estratégia de OCR | **Fine-tuning no UFPR-VeSV** | Escolhida pelo usuário — já temos ground-truth real (crops de placa + texto), e a meta do projeto (≥95% OCR accuracy) é mais realista com um modelo especializado do que com um genérico. |
| Modelo base | **PP-OCRv5_mobile_rec** | Escolhida pelo usuário — prioriza velocidade, alinhado à meta de ≤100ms/20 FPS. |
| Módulo usado | `TextRecognition` (reconhecimento isolado, não o pipeline completo `PaddleOCR`) | A Etapa 4 já entrega um recorte apertado da placa — rodar detecção de texto novamente seria redundante e mais lento. |

## Descoberta Importante: API do PaddleOCR Mudou

Minha referência inicial era da API 2.x (`PaddleOCR(use_angle_cls=True, lang='en').ocr(img)`).
Verifiquei durante esta etapa que a versão atual no PyPI é a **3.7.0**,
com uma API nova baseada em módulos/pipelines (`TextDetection`,
`TextRecognition`, `PaddleOCR` como pipeline completo). Todo o código
desta etapa já usa a API nova. Atualizei `requirements.txt`
(`paddleocr>=3.0`, `paddlepaddle-gpu>=3.0`) para refletir isso.

## Como o Fine-tuning Funciona (descoberto inspecionando o pacote)

O PaddleOCR 3.x delega treino ao **PaddleX**, cujo motor
(`paddlex.engine.Engine`) é dirigido por um config YAML (`-c`) + overrides
de linha de comando (`-o Global.mode=train ...`). Os configs de cada
modelo (incluindo `PP-OCRv5_mobile_rec.yaml`) já vêm empacotados dentro
do próprio pacote `paddlex` — não é necessário clonar nenhum repositório
externo. `scripts/train_plate_recognizer.py` localiza esse config
automaticamente e invoca o `Engine` programaticamente (montando
`sys.argv`, já que é assim que `Engine()` lê seus argumentos).

O dataset de treino segue o formato esperado pelo PaddleX: arquivos
`train.txt`/`val.txt`/`test.txt` com linhas `caminho_da_imagem\tTEXTO`
(confirmado inspecionando `paddlex/modules/text_recognition/dataset_checker`).

## Arquivos Criados

```
src/preprocessing/
└── rec_converter.py                # manifest → dataset de reconhecimento (recorte + label)

src/alpr/
├── plate_format.py                 # validação de formato BR/Mercosul
├── ocr_reader.py                   # OCRReader (wrapper de TextRecognition)
└── schemas.py                      # MODIFICADO — adiciona OCRResult

configs/
├── plate_recognition.yaml          # inferência
└── training_plate_recognition.yaml # hiperparâmetros de treino

scripts/
├── build_plate_recognition_dataset.py  # manifest.csv → dataset de OCR
└── train_plate_recognizer.py           # fine-tuning via paddlex.engine.Engine

tests/
├── test_rec_converter.py           # 5 testes (imagens sintéticas)
├── test_plate_format.py            # 10 testes (validação de padrão de placa)
└── test_ocr_reader.py              # 7 testes (mocks, sem pesos reais)
```

## Fluxo Completo

```
manifest.csv (Etapa 2)
    │
    ▼  scripts/build_plate_recognition_dataset.py
datasets/processed/plate_recognition/{images, train.txt, val.txt, test.txt}
    │
    ▼  scripts/train_plate_recognizer.py  (treino + exportação automática)
models/checkpoints/plate_recognizer/  (modelo pronto para inferência)
    │
    ▼  OCRReader.from_config()  (configs/plate_recognition.yaml)
read(plate_crop) → OCRResult(text, confidence, normalized_text, matches_known_format)
```

`OCRResult.matches_known_format` cruza o texto reconhecido com os
padrões de placa Brasil (`AAA9999`) e Mercosul (`AAA9A99`) — um sinal
extra de qualidade além da confiança do próprio modelo, útil para
decidir se um resultado deve ser aceito automaticamente ou sinalizado
para revisão manual.

## Limitação do Ambiente de Desenvolvimento

Mesma limitação de rede das Etapas 3 e 4 — confirmei aqui que o download
automático dos pesos do PaddleOCR também está bloqueado neste sandbox
(erro "No available model hosting platforms detected"). Por isso:

- Os 22 novos testes usam **mocks** e **imagens sintéticas**, sem
  depender de pesos reais ou do dataset completo.
- `scripts/train_plate_recognizer.py` **precisa rodar no Colab**, com
  internet liberada e as imagens do UFPR-VeSV disponíveis.

## Como Executar (no Colab, com dataset completo)

```bash
# 1. Gerar manifest (se ainda não gerado)
python scripts/build_dataset_manifest.py

# 2. Gerar dataset de reconhecimento (recortes + labels)
python scripts/build_plate_recognition_dataset.py

# 3. Treinar (fine-tuning, ~20 épocas por padrão)
python scripts/train_plate_recognizer.py

# 4. Rodar os testes (não dependem do treino real)
pytest tests/ -v
```

## Próximos Passos

**Etapa 6 (proposta):** Classificação de veículo (tipo, marca, modelo,
cor) — usando os atributos já presentes no manifest (`make`, `model`,
`color`, `type`) como rótulos para treinar um classificador de imagem
(provavelmente sobre o recorte do veículo da Etapa 3).

Antes de implementar, há uma decisão importante a apresentar: treinar
**um classificador multi-tarefa único** (uma rede, várias cabeças de
saída) vs. **quatro classificadores independentes** — com trade-offs de
custo computacional, complexidade e facilidade de treino/manutenção.

Quer que eu avance para a Etapa 6?
