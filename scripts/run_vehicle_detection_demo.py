"""Demonstração/smoke-test do VehicleDetector com pesos reais.

Requer conexão à internet para baixar `yolov8n.pt` na primeira execução
(não funciona no sandbox de desenvolvimento por restrição de rede — rode
isto no Google Colab ou em sua máquina local).

Uso:
    python scripts/run_vehicle_detection_demo.py caminho/para/imagem.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.detection.vehicle_detector import VehicleDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python scripts/run_vehicle_detection_demo.py <caminho_da_imagem>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")

    logger.info("Carregando VehicleDetector a partir de configs/detection.yaml")
    detector = VehicleDetector.from_config()

    detections = detector.detect(image)
    logger.info(f"{len(detections)} veículo(s) detectado(s)")

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"  - {label} | bbox={det.bbox_xyxy}")

    output_path = Path("outputs/predictions") / f"detected_{image_path.name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info(f"Imagem anotada salva em {output_path}")


if __name__ == "__main__":
    main()
