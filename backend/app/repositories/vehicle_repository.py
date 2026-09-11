"""Repository de Vehicle — acesso a dados, sem lógica de negócio."""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from backend.app.models.vehicle import Vehicle


class VehicleRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_plate(self, plate: str) -> Optional[Vehicle]:
        return self.db.query(Vehicle).filter(Vehicle.plate == plate).first()

    def get_or_create(
        self, plate: str, make: str = "", model: str = "", color: str = "", type_: str = ""
    ) -> Vehicle:
        """Retorna o veículo existente (por placa) ou cria um novo.

        Se o veículo já existir, os atributos de classificação (make/model/
        color/type) NÃO são sobrescritos aqui — a primeira leitura confiável
        prevalece. Atualização de atributos fica para uma decisão futura
        (ex: manter o valor de maior confiança histórica).
        """
        vehicle = self.get_by_plate(plate)
        if vehicle is not None:
            return vehicle

        vehicle = Vehicle(plate=plate, make=make, model=model, color=color, type=type_)
        self.db.add(vehicle)
        self.db.flush()  # garante vehicle.id disponível sem precisar commitar ainda
        return vehicle

    def count_all(self) -> int:
        return self.db.query(Vehicle).count()
