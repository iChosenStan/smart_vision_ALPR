# Etapa 2 — Integração do Dataset UFPR-VeSV

## Objetivo

Ingerir, validar e estruturar as anotações reais do UFPR-VeSV, gerando um
manifest tabular pronto para EDA e para as próximas etapas do pipeline
(detecção, OCR, classificação), com split treino/validação/teste sem
vazamento de dados.

## Descobertas sobre o Dataset Real

- **Acesso:** o UFPR-VeSV não possui download público direto; requer
  assinatura de *license agreement* e solicitação por e-mail institucional
  ao autor. Uso restrito a fins acadêmicos não-comerciais.
- **Schema real de `annotations.json`** (lista de 24.945 registros):
  ```json
  {
    "filename": "img_00001.jpg",
    "make": "chevrolet",
    "model": "pickup_corsa",
    "color": "unknown",
    "type": "compact-pickup",
    "infrared": "yes",
    "rear_view": "yes",
    "corners": [{"x":445,"y":290}, {"x":597,"y":281}, {"x":598,"y":329}, {"x":445,"y":338}],
    "plate": "MAV6C14"
  }
  ```
- **`corners` é um quadrilátero da PLACA**, não do veículo. O dataset
  **não** fornece bounding box do veículo completo — apenas atributos de
  classificação (make/model/color/type) e a localização da placa. Isso
  impacta a Etapa 3 (detecção de veículos): será necessário um detector
  genérico pré-treinado (ex: YOLO em COCO), já que não há ground-truth de
  veículo neste dataset.
- **`plate` é o identificador único do veículo.** Validado nos dados reais:
  16.297 placas únicas em 24.945 imagens (bate exatamente com o paper),
  ou seja, 6.035 veículos aparecem em mais de uma imagem.
- **Qualidade dos dados:** 100% dos 24.945 registros são válidos (sem
  campos ausentes, sem corners malformados). Inconsistências de dado
  observadas (não corrigidas, apenas registradas): valores como
  `"peugeuot"` (typo de "peugeot") existem na coluna `make` original.
- `infrared` e `rear_view` vêm como string `"yes"/"no"` no JSON original
  e são normalizados para `bool` no parsing.

## Arquivos Criados

```
src/preprocessing/
├── schemas.py              # PlateAnnotation, Point (dataclasses tipados)
├── annotation_parser.py    # load_annotations() + ParseReport
└── dataset_registry.py     # build_manifest(), validate_images_exist(), split_by_vehicle()

scripts/
└── build_dataset_manifest.py   # CLI: annotations.json → manifest.csv com split

tests/
├── fixtures/sample_annotations.json   # 19 registros reais (subset determinístico)
└── test_annotation_parser.py          # 11 testes cobrindo parser + registry

configs/base.yaml   # atualizado: dataset.annotations_file, images_dir, split ratios
```

## Fluxo

1. `load_annotations(path)` lê o JSON, valida cada registro e retorna
   `(List[PlateAnnotation], ParseReport)`. Registros inválidos **não**
   interrompem o carregamento — são reportados em `ParseReport.errors` e
   excluídos, tornando o pipeline resiliente a pequenas inconsistências.
2. `build_manifest(annotations)` converte para `pandas.DataFrame`, já
   incluindo o bounding box axis-aligned (`plate_bbox_x1..y2`) derivado do
   quadrilátero original — necessário para treinar um detector de placa
   baseado em bbox (YOLO) nas próximas etapas.
3. `validate_images_exist(manifest, images_dir)` checa quantas imagens
   referenciadas realmente existem em disco (não lança erro; útil durante
   download parcial do dataset).
4. `split_by_vehicle(manifest, ...)` agrupa por `plate` antes de dividir
   em train/val/test, garantindo que o mesmo veículo nunca apareça em
   splits diferentes.

## Como Executar

```bash
# 1. Colocar os arquivos reais do dataset (após obter acesso):
#    datasets/raw/annotations.json
#    datasets/raw/images/*.jpg

# 2. Gerar o manifest
python scripts/build_dataset_manifest.py

# 3. Rodar os testes
pytest tests/ -v
```

### Resultado real obtido (rodando com o `annotations.json` completo)

```
Anotações carregadas: 24945/24945 válidas (100.00%)
Manifest construído: 24945 imagens, 16297 veículos únicos
Split por veículo — imagens: {'train': 17348, 'test': 3811, 'val': 3786}
                     veículos: train=11407, val=2444, test=2446
```

> Nota: a validação de imagens acusou diretório `datasets/raw/images` não
> encontrado neste ambiente, pois apenas o `annotations.json` foi
> disponibilizado até o momento — comportamento esperado e tratado sem erro.

## Decisões Técnicas Tomadas Nesta Etapa

| Decisão | Escolha | Justificativa |
|---|---|---|
| Proporção do split | 70/15/15 | Definida pelo usuário. |
| Estratégia de split | Agrupado por `plate` (veículo) | Evita vazamento de dados — sem isso, o mesmo veículo poderia aparecer em treino e teste simultaneamente, inflando artificialmente as métricas. |
| Localização das imagens | `datasets/raw/images/` | Definida pelo usuário. |
| Tratamento de registros inválidos | Log + exclusão, sem interromper o parsing | Dataset real veio 100% íntegro, mas o parser precisa ser robusto para uso em produção/monitoramento urbano futuro, onde a fonte de dados pode ser menos controlada. |

## Próximos Passos

**Etapa 3 (proposta):** Detecção de veículos. Como o UFPR-VeSV não anota
bounding box de veículo, a estratégia recomendada é usar um detector
genérico pré-treinado (YOLO11 ou YOLOv8 em COCO, classes `car`, `truck`,
`motorcycle`, `bus`) para essa etapa, reservando o fine-tuning para a
detecção de placa (Etapa 4), onde temos ground-truth real via
`plate_bbox_x1..y2`.

Antes de implementar, será apresentado o comparativo de opções de
detector (YOLO11 vs YOLOv8, custo computacional, precisão) — aguardando
sua decisão, conforme o fluxo de trabalho do projeto.
