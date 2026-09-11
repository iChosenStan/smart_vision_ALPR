"""DTOs Pydantic relacionados a Vehicle."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class VehicleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    plate: str
    make: str
    model: str
    color: str
    type: str
