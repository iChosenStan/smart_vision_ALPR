# Guia de Setup Local — Windows + GPU NVIDIA

Este guia complementa o `README.md` com instruções específicas para
rodar o SmartVision ALPR localmente no Windows com GPU NVIDIA/CUDA,
incluindo os passos exatos para colocar o dataset UFPR-VeSV no lugar e
rodar o pipeline completo (treino das 3 etapas + avaliação + demo).

## 0. Pré-requisitos

- Python 3.11 instalado (`python --version`)
- Driver NVIDIA atualizado — rode `nvidia-smi` no PowerShell/CMD e
  confira a versão do CUDA suportada (aparece no canto superior direito,
  ex: `CUDA Version: 12.4`). Você vai precisar dela no passo 2.
- Git (opcional, só se for clonar em vez de extrair o `.zip`)

## 1. Extrair o projeto e criar o ambiente virtual

```powershell
# Extraia o .zip entregue nas etapas anteriores, depois:
cd smartvision-alpr

python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip
```

## 2. Instalar PyTorch com CUDA (ANTES do requirements.txt)

Isso é importante: se você instalar `requirements.txt` primeiro, o `pip`
pode resolver uma versão de PyTorch sem suporte a CUDA. Instale o PyTorch
explicitamente primeiro, escolhendo a URL que bate com a versão do CUDA
que você viu no `nvidia-smi`:

```powershell
# Exemplo para CUDA 12.4 (ajuste cuXXX conforme sua versão real):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Se não tiver certeza da versão exata, veja o seletor oficial:
https://pytorch.org/get-started/locally/

**Valide antes de continuar:**

```powershell
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
```

Se aparecer `False`, pare aqui e corrija antes de prosseguir (normalmente
é incompatibilidade entre a versão do CUDA instalada e o wheel do
PyTorch escolhido).

## 3. Instalar o restante das dependências

```powershell
pip install -r requirements.txt
```

**Sobre o PaddlePaddle-GPU no Windows:** o comando acima tenta instalar
`paddlepaddle-gpu` genérico. Se der erro ou se `paddle.utils.run_check()`
(abaixo) não detectar a GPU, use o instalador oficial com a versão exata
do seu CUDA:
https://www.paddlepaddle.org.cn/en/install/quick

**Fallback aceitável:** se o PaddlePaddle-GPU continuar difícil de
instalar no seu ambiente, instale a versão CPU (`pip install paddlepaddle`
no lugar de `paddlepaddle-gpu`) e mude `device: "gpu"` para `device: "cpu"`
em `configs/plate_recognition.yaml` e `configs/training_plate_recognition.yaml`.
O modelo de OCR (`PP-OCRv5_mobile_rec`) é leve — rodar em CPU só para esse
estágio tem impacto pequeno na latência total.

**IMPORTANTE (descoberto na prática, Windows):** PaddlePaddle-GPU e
PyTorch (usado por `ultralytics`/YOLO e pelo classificador) têm um
**conflito binário conhecido e sem correção** quando ambos rodam no
mesmo processo Python no Windows — erro típico:
`ImportError: generic_type: type "_gpuDeviceProperties" is already registered!`.
Como o pipeline usa os dois juntos, a solução é usar o **PaddlePaddle em
CPU** (só para o estágio de OCR — o resto continua em GPU via PyTorch):

```powershell
pip uninstall paddlepaddle-gpu -y
pip install paddlepaddle
```

E ajuste `device: "gpu"` → `device: "cpu"` em `configs/plate_recognition.yaml`
e `configs/training_plate_recognition.yaml` (já vem assim por padrão no
projeto, a partir desta versão). O modelo de OCR (`mobile_rec`) é leve —
o impacto de rodar em CPU é pequeno.

**Valide:**

```powershell
python -c "import paddle; paddle.utils.run_check()"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
```

**Instalar o plugin de treino do PaddleOCR (obrigatório para treinar, não
para inferência):** `pip install paddleocr` só traz o necessário para
*rodar* modelos já treinados. Para fazer fine-tuning (Etapa 5), é preciso
instalar o plugin de treino:

```powershell
paddlex --install PaddleOCR
```

Se der erro de pasta não encontrada (`repo_manager\repos`), crie a pasta
manualmente antes de rodar de novo:

```powershell
mkdir "$env:LOCALAPPDATA\Programs\Python\Python310\Lib\site-packages\paddlex\repo_manager\repos"
```

(ajuste o caminho conforme a localização real do seu Python/venv). Esse
comando também exige `git` instalado e no PATH.

## 4. Colocar o dataset no lugar certo

```powershell
# Copie o annotations.json que você já tem para:
copy caminho\para\seu\annotations.json datasets\raw\annotations.json

# Copie (ou mova) TODAS as imagens do UFPR-VeSV para:
# datasets\raw\images\img_00001.jpg, img_00002.jpg, ...
```

Estrutura esperada ao final:

```
datasets\raw\
├── annotations.json
└── images\
    ├── img_00001.jpg
    ├── img_00002.jpg
    └── ... (24.945 arquivos)
```

## 5. Rodar o pipeline completo, em ordem

```powershell
# 1. Gerar o manifest (parsing + split 70/15/15 por veículo)
python scripts\build_dataset_manifest.py

# 2. Rodar os testes (validam toda a lógica antes de gastar tempo com treino)
pytest tests\ -v

# 3. Dataset de detecção de placa (formato YOLO)
python scripts\build_plate_detection_dataset.py

# 4. Treinar o detector de placa (YOLOv8n, 50 épocas — alguns minutos em GPU)
python scripts\train_plate_detector.py

# 5. Avaliar o detector de placa (mAP)
python scripts\evaluate_plate_detector.py

# 6. Dataset de reconhecimento de placa (recortes + labels)
python scripts\build_plate_recognition_dataset.py

# 7. Treinar o OCR (fine-tuning do PP-OCRv5_mobile_rec)
python scripts\train_plate_recognizer.py

# 8. Treinar o classificador multi-tarefa (ResNet34, ~30 épocas)
python scripts\train_vehicle_classifier.py

# 9. Avaliação final completa (OCR isolado + classificação + end-to-end)
python scripts\run_evaluation.py

# 10. Demo visual em uma imagem
python scripts\run_pipeline_demo.py --image caminho\para\uma_imagem.jpg
```

Os passos 4, 7 e 8 são os que demoram (treino de verdade) — o tempo varia
bastante conforme sua GPU; espere de alguns minutos a algumas horas no
total para os três juntos, dependendo do hardware.

## 6. Onde ficam os resultados

- Pesos treinados: `models/checkpoints/`
- Métricas de treino (curvas, etc.): `outputs/training_runs/`
- Relatório final de avaliação: `outputs/reports/evaluation_report.md`
- Imagem/vídeo anotado da demo: `outputs/predictions/`

## Problemas Comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `torch.cuda.is_available()` retorna `False` | Wheel do PyTorch sem suporte a CUDA, ou driver desatualizado | Reinstale o PyTorch com o índice `cuXXX` correto (passo 2) |
| Erro ao instalar `paddlepaddle-gpu` | Wheel genérico incompatível com seu CUDA | Use o instalador oficial do Paddle, ou faça fallback para CPU (passo 3) |
| `FileNotFoundError` em algum script | Passo anterior não foi rodado, ou dataset não está no lugar certo | Siga a ordem exata da seção 5 — cada script depende do anterior |
| Treino muito lento | Rodando em CPU sem perceber | Confira `device` nos configs (`configs/*.yaml`) e rode as validações do passo 2/3 |
