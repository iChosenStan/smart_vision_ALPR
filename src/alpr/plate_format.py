"""Validação de formato de placa veicular — padrões Brasil e Mercosul.

O UFPR-VeSV contém os dois padrões (ver docs/02_dataset.md), então o
resultado do OCR deve ser validado contra ambos antes de ser considerado
confiável — útil como sinal extra de qualidade além da confiança do
próprio modelo de reconhecimento.
"""

from __future__ import annotations

import re

# Padrão antigo brasileiro: 3 letras + 4 dígitos (ex: ABC1234)
_BR_PATTERN = re.compile(r"^[A-Z]{3}[0-9]{4}$")

# Padrão Mercosul: 3 letras + 1 dígito + 1 letra + 2 dígitos (ex: ABC1D23)
_MERCOSUL_PATTERN = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")


def normalize_plate_text(text: str) -> str:
    """Remove espaços/símbolos e converte para maiúsculas."""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def matches_known_plate_format(text: str) -> bool:
    """Verifica se o texto (já normalizado) corresponde a um dos padrões
    de placa conhecidos (Brasil antigo ou Mercosul).
    """
    normalized = normalize_plate_text(text)
    return bool(_BR_PATTERN.match(normalized) or _MERCOSUL_PATTERN.match(normalized))
