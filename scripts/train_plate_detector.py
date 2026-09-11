"""Script de treino do detector de placa (fine-tuning de YOLOv8n).

Requer:
    - datasets/processed/plate_detection/data.yaml (gerado por
      scripts/build_plate_detection_dataset.py)
    - GPU disponível (recomendado — treina em CPU, mas MUITO mais lento)
    - Conexão à internet na primeira execução, para baixar os pesos base
      yolov8n.pt (ver limitação de rede documentada em docs/03_vehicle_detection.md)

Uso:
    python scripts/train_plate_detector.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path("configs/training_plate_detection.yaml")


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Pacote 'ultralytics' não instalado. Rode: pip install ultralytics") from exc

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_yaml = config["dataset"]["data_yaml"]
    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"{data_yaml} não encontrado. Rode primeiro: "
            "python scripts/build_plate_detection_dataset.py"
        )

    logger.info(f"Iniciando treino: base={config['model']['base_weights']}, "
                f"epochs={config['training']['epochs']}, imgsz={config['training']['imgsz']}")

    model = YOLO(config["model"]["base_weights"])

    results = model.train(
        data=data_yaml,
        epochs=config["training"]["epochs"],
        imgsz=config["training"]["imgsz"],
        batch=config["training"]["batch"],
        optimizer=config["training"]["optimizer"],
        patience=config["training"]["patience"],
        project=config["training"]["project"],
        name=config["training"]["name"],
        device=config["model"]["device"],
    )

    best_weights_src = Path(results.save_dir) / "weights" / "best.pt"
    best_weights_dest = Path(config["output"]["best_weights_dest"])
    best_weights_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights_src, best_weights_dest)

    logger.info(f"Treino concluído. Melhores pesos copiados para {best_weights_dest}")
    print(f"\nPesos finais salvos em: {best_weights_dest}")
    print(f"Métricas completas em: {results.save_dir}")


if __name__ == "__main__":
    main()
