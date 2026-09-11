"""Script de treino do classificador multi-tarefa de veículo (Etapa 6).

Fine-tuning de um ResNet34 pré-treinado em ImageNet, com 4 cabeças de
classificação (type, make, model, color) treinadas simultaneamente sobre
a imagem completa (o UFPR-VeSV não fornece bbox de veículo — ver
docs/06_classification.md).

Requer:
    - datasets/processed/manifest.csv (Etapa 2)
    - Imagens reais em datasets/raw/images/ (Etapa 2)
    - GPU disponível (recomendado)
    - Conexão à internet na primeira execução (pesos ImageNet do ResNet34)

Uso:
    python scripts/train_vehicle_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.classification.dataset import VehicleClassificationDataset
from src.classification.label_encoder import LabelEncoders
from src.classification.multi_task_model import VehicleMultiTaskNet
from src.classification.transforms import build_eval_transform, build_train_transform
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path("configs/training_classification.yaml")
_TASKS = ("type", "make", "model", "color")


def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA solicitado mas não disponível — usando CPU.")
        return "cpu"
    return device


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = Path(config["dataset"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} não encontrado. Rode primeiro: "
            "python scripts/build_dataset_manifest.py"
        )

    manifest = pd.read_csv(manifest_path)
    images_dir = config["dataset"]["images_dir"]
    image_size = config["dataset"]["image_size"]

    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    logger.info(f"Treino: {len(train_manifest)} imagens | Validação: {len(val_manifest)} imagens")

    # Encoders ajustados SOMENTE no split de treino, evitando vazamento de
    # classes vistas apenas em validação para dentro do vocabulário do modelo.
    encoders = LabelEncoders.fit(train_manifest, tasks=list(_TASKS))
    encoders_dest = Path(config["output"]["encoders_dest"])
    encoders.save(encoders_dest)

    train_dataset = VehicleClassificationDataset(
        train_manifest, images_dir, encoders, transform=build_train_transform(image_size)
    )
    val_dataset = VehicleClassificationDataset(
        val_manifest, images_dir, encoders, transform=build_eval_transform(image_size)
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = _resolve_device(config["model"]["device"])
    model = VehicleMultiTaskNet(
        num_classes=encoders.num_classes,
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)

    loss_weights = config["training"]["task_loss_weights"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    best_val_acc = 0.0
    best_weights_dest = Path(config["output"]["best_weights_dest"])
    best_weights_dest.parent.mkdir(parents=True, exist_ok=True)

    epochs = config["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = torch.zeros(1, device=device)
            for task in _TASKS:
                labels = batch[f"{task}_label"].to(device)
                task_loss = criterion(outputs[task], labels)
                loss = loss + loss_weights[task] * task_loss

            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        model.eval()
        val_accs = {task: [] for task in _TASKS}
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                for task in _TASKS:
                    labels = batch[f"{task}_label"].to(device)
                    val_accs[task].append(_batch_accuracy(outputs[task], labels))

        avg_val_accs = {task: (sum(v) / len(v) if v else 0.0) for task, v in val_accs.items()}
        mean_val_acc = sum(avg_val_accs.values()) / len(avg_val_accs)

        logger.info(
            f"Época {epoch}/{epochs} — train_loss={train_loss_total / max(len(train_loader), 1):.4f} — "
            f"val_acc={ {k: round(v, 4) for k, v in avg_val_accs.items()} } — média={mean_val_acc:.4f}"
        )

        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            torch.save(model.state_dict(), best_weights_dest)
            logger.info(f"Novo melhor modelo salvo (val_acc médio={best_val_acc:.4f})")

    print(f"\nTreino concluído. Melhor val_acc médio: {best_val_acc:.4f}")
    print(f"Pesos salvos em: {best_weights_dest}")
    print(f"Encoders salvos em: {encoders_dest}")


if __name__ == "__main__":
    main()
