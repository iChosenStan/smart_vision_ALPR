"""Script de treino do reconhecedor de texto da placa (fine-tuning de
PP-OCRv5_mobile_rec via PaddleX).

Internamente, o PaddleOCR 3.x delega treino ao PaddleX, cujo motor
(`paddlex.engine.Engine`) é dirigido por um arquivo de config `-c` +
overrides `-o` (interface de linha de comando). Este script monta esses
argumentos programaticamente e chama o `Engine` diretamente — evitando
depender do `main.py` do repositório do PaddleX (não incluído no pacote
pip), usando apenas os configs de módulo já empacotados em `paddlex`.

Requer:
    - datasets/processed/plate_recognition/{train,val}.txt (gerado por
      scripts/build_plate_recognition_dataset.py)
    - GPU disponível (recomendado)
    - Conexão à internet na primeira execução, para baixar os pesos base
      pré-treinados (mesma limitação de rede documentada em
      docs/03_vehicle_detection.md)

Uso:
    python scripts/train_plate_recognizer.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path("configs/training_plate_recognition.yaml")


def _run_engine(base_config_path: Path, overrides: list[str]) -> None:
    """Roda o `paddlex.engine.Engine` com os overrides fornecidos.

    Manipula `sys.argv` porque `Engine()` usa `argparse` internamente
    (mesmo padrão do `main.py` documentado pelo PaddleX).
    """
    from paddlex.engine import Engine

    argv_backup = sys.argv
    try:
        sys.argv = ["train_plate_recognizer.py", "-c", str(base_config_path)]
        for override in overrides:
            sys.argv += ["-o", override]
        Engine().run()
    finally:
        sys.argv = argv_backup


def _find_base_config(base_model_name: str) -> Path:
    import paddlex

    config_path = (
        Path(paddlex.__file__).parent
        / "configs"
        / "modules"
        / "text_recognition"
        / f"{base_model_name}.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config base '{base_model_name}' não encontrado em {config_path}. "
            "Verifique o nome do modelo em configs/training_plate_recognition.yaml."
        )
    return config_path


def _find_exported_inference_dir(output_dir: Path) -> Path:
    """Localiza a pasta com o modelo exportado (formato de inferência),
    procurando pelo arquivo `inference.yml` gerado pela exportação.

    Evita depender de um caminho fixo adivinhado — a estrutura exata de
    subpastas do PaddleX/PaddleOCR pode variar entre versões (ver bug real
    corrigido nesta função: a suposição original de `best_accuracy/inference`
    estava incorreta).
    """
    matches = list(output_dir.rglob("inference.yml"))
    if not matches:
        raise FileNotFoundError(
            f"Não encontrei 'inference.yml' em nenhuma subpasta de {output_dir}. "
            "A exportação pode ter falhado — confira os logs acima."
        )
    return matches[0].parent


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_dir = Path(config["dataset"]["dataset_dir"]).resolve()
    if not (dataset_dir / "train.txt").exists():
        raise FileNotFoundError(
            f"{dataset_dir / 'train.txt'} não encontrado. Rode primeiro: "
            "python scripts/build_plate_recognition_dataset.py"
        )

    base_model_name = config["model"]["base_model_name"]
    base_config_path = _find_base_config(base_model_name)
    output_dir = Path(config["training"]["output_dir"]).resolve()

    common_overrides = [
        f"Global.dataset_dir={dataset_dir}",
        f"Global.device={config['model']['device']}",
        f"Global.output={output_dir}",
    ]

    # 1) Treino
    logger.info(f"Iniciando treino: base={base_model_name}, dataset={dataset_dir}")
    _run_engine(
        base_config_path,
        common_overrides
        + [
            "Global.mode=train",
            f"Train.epochs_iters={config['training']['epochs']}",
            f"Train.batch_size={config['training']['batch_size']}",
            f"Train.learning_rate={config['training']['learning_rate']}",
        ],
    )

    # 2) Exportação para formato de inferência
    # NOTA: o checkpoint fica DIRETO em output_dir (não numa subpasta) —
    # confirmado na documentação oficial do PaddleOCR: best_accuracy.pdparams,
    # latest.pdparams e iter_epoch_X.pdparams são salvos em Global.save_model_dir.
    best_weights = output_dir / "best_accuracy.pdparams"
    logger.info(f"Exportando modelo treinado ({best_weights}) para formato de inferência")
    _run_engine(
        base_config_path,
        common_overrides + ["Global.mode=export", f"Export.weight_path={best_weights}"],
    )

    # 3) Copia o modelo exportado para models/checkpoints/
    exported_inference_dir = _find_exported_inference_dir(output_dir)
    final_dest = Path(config["output"]["exported_model_dest"])
    if final_dest.exists():
        shutil.rmtree(final_dest)
    shutil.copytree(exported_inference_dir, final_dest)

    logger.info(f"Treino concluído. Modelo exportado para {final_dest}")
    print(f"\nModelo pronto para inferência em: {final_dest}")
    print(f"Métricas completas em: {output_dir}")


if __name__ == "__main__":
    main()
