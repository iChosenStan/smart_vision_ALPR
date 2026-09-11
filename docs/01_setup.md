# Etapa 1 — Estruturação Base do Projeto

## Objetivo

Estabelecer a fundação estrutural do SmartVision ALPR antes de qualquer
implementação de lógica de visão computacional: árvore de diretórios,
gerenciamento de dependências, configuração centralizada, logging e
tratamento de exceções.

## Arquivos Criados

```
smartvision-alpr/
├── requirements.txt          # Dependências do projeto (pip)
├── .gitignore                 # Regras de exclusão de versionamento
├── README.md                  # Documentação principal
├── configs/
│   └── base.yaml              # Configuração global (paths, hardware, metas)
├── src/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Logging centralizado (loguru)
│       └── exceptions.py      # Hierarquia de exceções customizadas
├── tests/
│   ├── __init__.py
│   └── test_setup.py          # Testes da estrutura base
├── datasets/{raw,processed}/.gitkeep
├── models/{pretrained,checkpoints}/.gitkeep
└── outputs/{logs,predictions,reports}/.gitkeep
```

Demais diretórios de `src/` (`detection`, `alpr`, `classification`,
`preprocessing`, `training`, `evaluation`, `visualization`) foram criados
com `__init__.py` vazio, prontos para receber código nas próximas etapas.

## Fluxo

1. `configs/base.yaml` centraliza parâmetros globais (paths, seed, hardware,
   metas de performance). Módulos específicos (ex: config de treino de um
   modelo) devem futuramente referenciar ou herdar deste arquivo, evitando
   duplicação de parâmetros.
2. `src/utils/logger.py` expõe `get_logger(__name__)`, configurando o
   loguru uma única vez (idempotente) com saída colorida em console e
   arquivo rotativo em `outputs/logs/smartvision.log`.
3. `src/utils/exceptions.py` define uma hierarquia de exceções específicas
   do domínio (`VehicleDetectionError`, `PlateDetectionError`, `OCRError`,
   `ClassificationError`, etc.), todas derivando de `SmartVisionError`,
   permitindo tratamento granular por estágio do pipeline.

## Como Executar

```bash
cd smartvision-alpr
pip install -r requirements.txt
pytest tests/ -v
```

Saída esperada: 4 testes passando, validando config, logger, exceções e
estrutura de diretórios.

## Decisões Técnicas Tomadas Nesta Etapa

| Decisão | Escolha | Justificativa |
|---|---|---|
| Gerenciamento de dependências | `requirements.txt` | Mais simples de instalar diretamente em células do Google Colab (`!pip install -r requirements.txt`), sem overhead de ambiente virtual isolado que o Colab não usa de forma nativa. |
| Biblioteca de logging | `loguru` | API mais simples que `logging` nativo, com rotação de arquivo e formatação colorida prontas, reduzindo boilerplate. |

## Próximos Passos

**Etapa 2 (proposta):** Integração do dataset UFPR-VeSV — script de
download/organização, exploração inicial (EDA) em notebook, e definição
do formato de anotação a ser usado internamente (ex: conversão para
formato YOLO se necessário).

Antes de implementar a Etapa 2, será apresentado: objetivo, estratégia,
dependências necessárias — aguardando sua validação, conforme definido
no fluxo de trabalho do projeto.
