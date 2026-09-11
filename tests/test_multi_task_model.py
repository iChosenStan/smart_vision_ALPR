"""Testes da Etapa 6 — arquitetura do modelo multi-tarefa.

Como `torch`/`torchvision` já estão instalados (dependência do
Ultralytics), estes testes rodam um forward pass REAL do modelo — usando
`pretrained=False` para não depender do download dos pesos ImageNet
(bloqueado neste sandbox, mesma limitação de rede das Etapas 3-5).
"""

from __future__ import annotations

import torch

from src.classification.multi_task_model import VehicleMultiTaskNet

_NUM_CLASSES = {"type": 5, "make": 3, "model": 10, "color": 4}


def test_forward_pass_returns_correct_output_shapes() -> None:
    model = VehicleMultiTaskNet(num_classes=_NUM_CLASSES, pretrained=False)
    model.eval()

    batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(batch)

    assert set(outputs.keys()) == set(_NUM_CLASSES.keys())
    for task, num_classes in _NUM_CLASSES.items():
        assert outputs[task].shape == (2, num_classes)


def test_forward_pass_is_deterministic_in_eval_mode() -> None:
    model = VehicleMultiTaskNet(num_classes=_NUM_CLASSES, pretrained=False, dropout=0.5)
    model.eval()

    batch = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)

    for task in _NUM_CLASSES:
        assert torch.allclose(out1[task], out2[task])


def test_heads_are_independent_linear_layers_with_correct_output_dim() -> None:
    model = VehicleMultiTaskNet(num_classes=_NUM_CLASSES, pretrained=False)
    for task, num_classes in _NUM_CLASSES.items():
        assert model.heads[f"head_{task}"].out_features == num_classes


def test_model_accepts_single_sample_batch() -> None:
    model = VehicleMultiTaskNet(num_classes=_NUM_CLASSES, pretrained=False)
    model.eval()
    batch = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(batch)
    for task in _NUM_CLASSES:
        assert outputs[task].shape[0] == 1
