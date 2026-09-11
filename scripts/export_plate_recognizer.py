"""Exporta um checkpoint do reconhecedor de placa (mesmo que o treino
ainda não tenha terminado) para o formato de inferência, permitindo
testar o pipeline com um modelo PARCIALMENTE treinado.

Útil quando o treino completo demora muito (ex: CPU) e você quer validar
o restante do pipeline antes dele terminar. A qualidade do OCR será
proporcional a quantas épocas já rodaram — não é o resultado final.

Uso:
    # Usa o melhor checkpoint disponível até agora (recomendado)
    python scripts/export_plate_recognizer.py

    # Usa o checkpoint da última época rodada (mesmo que não seja o melhor)
    python scripts/export_plate_recognizer.py --checkpoint latest
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path("configs/training_plate_recognition.yaml")


def _run_engine(base_config_path: Path, overrides: list[str]) -> None:
    from paddlex.engine import Engine

    argv_backup = sys.argv
    try:
        sys.argv = ["export_plate_recognizer.py", "-c", str(base_config_path)]
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
        raise FileNotFoundError(f"Config base '{base_model_name}' não encontrado em {config_path}.")
    return config_path


def _find_exported_inference_dir(output_dir: Path) -> Path:
    matches = list(output_dir.rglob("inference.yml"))
    if not matches:
        raise FileNotFoundError(
            f"Não encontrei 'inference.yml' em nenhuma subpasta de {output_dir}. "
            "A exportação pode ter falhado — confira os logs acima."
        )
    return matches[0].parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta um checkpoint do OCR para formato de inferência.")
    parser.add_argument(
        "--checkpoint",
        default="best_accuracy",
        choices=["best_accuracy", "latest"],
        help="Qual checkpoint exportar: 'best_accuracy' (melhor até agora, recomendado) ou "
        "'latest' (última época rodada, mesmo que não seja a melhor).",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["training"]["output_dir"]).resolve()
    checkpoint_path = output_dir / f"{args.checkpoint}.pdparams"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} não encontrado. O treino ainda não salvou esse checkpoint "
            "— espere pelo menos 1 época completa (ou 1 avaliação, para 'best_accuracy')."
        )

    base_model_name = config["model"]["base_model_name"]
    base_config_path = _find_base_config(base_model_name)

    logger.info(f"Exportando checkpoint parcial: {checkpoint_path}")
    _run_engine(
        base_config_path,
        [
            f"Global.dataset_dir={Path(config['dataset']['dataset_dir']).resolve()}",
            f"Global.device={config['model']['device']}",
            f"Global.output={output_dir}",
            "Global.mode=export",
            f"Export.weight_path={checkpoint_path}",
        ],
    )

    exported_inference_dir = _find_exported_inference_dir(output_dir)
    final_dest = Path(config["output"]["exported_model_dest"])
    if final_dest.exists():
        shutil.rmtree(final_dest)
    shutil.copytree(exported_inference_dir, final_dest)

    logger.info(f"Checkpoint parcial exportado para {final_dest}")
    print(f"\nModelo (PARCIAL, checkpoint='{args.checkpoint}') pronto para teste em: {final_dest}")
    print("Lembre-se: a qualidade do OCR ainda não é a final, o treino completo continua rodando.")


if __name__ == "__main__":
    main()
