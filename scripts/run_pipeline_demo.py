"""Demonstração/smoke-test do pipeline completo (Etapas 3-7) com pesos reais.

Requer todos os modelos treinados (Etapas 4, 5 e 6) e conexão à internet
para o YOLOv8n genérico da Etapa 3 — não funciona no sandbox de
desenvolvimento (ver limitação de rede documentada em
docs/03_vehicle_detection.md). Rode isto no Google Colab.

Uso:
    # Imagem única
    python scripts/run_pipeline_demo.py --image caminho/para/imagem.jpg

    # Vídeo (processa e salva anotado; imprime estatísticas de FPS/latência)
    python scripts/run_pipeline_demo.py --video caminho/para/video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.pipeline.smart_vision_pipeline import SmartVisionPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _draw_frame(frame, frame_result) -> None:
    for vehicle in frame_result.vehicles:
        x1, y1, x2, y2 = (int(v) for v in vehicle.vehicle_detection.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_parts = [vehicle.vehicle_detection.class_name]
        if vehicle.classification is not None:
            label_parts.append(vehicle.classification.make.label)
            label_parts.append(vehicle.classification.color.label)
        if vehicle.plate_text:
            label_parts.append(vehicle.plate_text)
        label = " | ".join(label_parts)

        cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def run_on_image(pipeline: SmartVisionPipeline, image_path: Path) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")

    result = pipeline.process_frame(frame)
    logger.info(
        f"{len(result.vehicles)} veículo(s) — {result.total_latency_ms:.1f}ms "
        f"(meta ≤100ms: {'OK' if result.meets_latency_target else 'FORA DA META'})"
    )
    for v in result.vehicles:
        print(
            f"  - {v.vehicle_detection.class_name} | placa={v.plate_text or 'não lida'} | "
            f"make={v.classification.make.label} | color={v.classification.color.label}"
        )

    _draw_frame(frame, result)
    output_path = Path("outputs/predictions") / f"pipeline_{image_path.name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    logger.info(f"Imagem anotada salva em {output_path}")


def run_on_video(pipeline: SmartVisionPipeline, video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path("outputs/predictions") / f"pipeline_{video_path.name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    latencies = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pipeline.process_frame(frame)
        latencies.append(result.total_latency_ms)
        _draw_frame(frame, result)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        print(f"\n{frame_idx} frames processados")
        print(f"Latência média: {avg_latency:.1f}ms | FPS médio: {avg_fps:.1f}")
        print(f"Meta do projeto: ≤100ms / 20 FPS — {'OK' if avg_latency <= 100 else 'FORA DA META'}")

    logger.info(f"Vídeo anotado salvo em {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo do pipeline SmartVision ALPR completo.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Caminho para uma imagem única")
    group.add_argument("--video", type=str, help="Caminho para um vídeo")
    args = parser.parse_args()

    logger.info("Carregando pipeline completo a partir dos configs (Etapas 3-7)")
    pipeline = SmartVisionPipeline.from_config()

    if args.image:
        run_on_image(pipeline, Path(args.image))
    else:
        run_on_video(pipeline, Path(args.video))


if __name__ == "__main__":
    main()
