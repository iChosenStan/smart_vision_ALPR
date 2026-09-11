"""Exceções customizadas do SmartVision ALPR."""
from __future__ import annotations


class SmartVisionError(Exception):
    """Exceção base para todos os erros específicos do projeto."""


class ConfigError(SmartVisionError):
    pass


class DatasetError(SmartVisionError):
    pass


class ModelLoadError(SmartVisionError):
    pass


class VehicleDetectionError(SmartVisionError):
    pass


class PlateDetectionError(SmartVisionError):
    pass


class OCRError(SmartVisionError):
    pass


class ClassificationError(SmartVisionError):
    pass


class InferencePipelineError(SmartVisionError):
    pass
