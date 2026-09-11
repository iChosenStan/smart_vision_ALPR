"""Modelo multi-tarefa de classificação de veículo (Etapa 6).

Uma única rede — backbone ResNet34 compartilhado — com 4 cabeças de
classificação independentes (type, make, model, color), escolhida para
manter a inferência em um único forward pass (decisão registrada em
docs/06_classification.md), já que o orçamento de latência do pipeline
(≤100ms) já é consumido por detecção de veículo, detecção de placa e OCR.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class VehicleMultiTaskNet(nn.Module):
    """Backbone ResNet34 compartilhado + N cabeças lineares independentes."""

    def __init__(
        self,
        num_classes: Dict[str, int],
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            num_classes: Mapeamento {"type": N, "make": N, "model": N,
                "color": N} com o número de classes de cada tarefa.
            pretrained: Se True, inicializa o backbone com pesos ImageNet
                (requer download na primeira execução — ver limitação de
                rede documentada em docs/03_vehicle_detection.md).
            dropout: Probabilidade de dropout aplicada antes de cada cabeça.
        """
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)
        feature_dim = backbone.fc.in_features  # 512 para ResNet34
        backbone.fc = nn.Identity()  # remove a cabeça original de 1000 classes (ImageNet)
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.task_names = tuple(num_classes.keys())

        # Prefixo necessário: "type" colide com o método nativo
        # `nn.Module.type()`, então as chaves do ModuleDict não podem usar
        # os nomes de tarefa diretamente.
        self._head_prefix = "head_"
        self.heads = nn.ModuleDict(
            {f"{self._head_prefix}{task}": nn.Linear(feature_dim, n) for task, n in num_classes.items()}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Batch de imagens, shape (B, 3, H, W), já normalizadas.

        Returns:
            Dicionário {task: logits}, onde `logits` tem shape
            (B, num_classes[task]).
        """
        features = self.backbone(x)
        features = self.dropout(features)
        return {
            task: self.heads[f"{self._head_prefix}{task}"](features) for task in self.task_names
        }
