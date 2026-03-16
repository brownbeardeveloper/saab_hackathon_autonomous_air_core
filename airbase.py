from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config dataclasses (frozen – these are read-only reference data)
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class TransferConfig:
    duration_hours: float
    fuel_cost: float


@dataclass(slots=True, frozen=True)
class DeliveryBounds:
    max_fuel_per_delivery: float
    max_weapons_per_delivery: Dict[int, int]
    max_spare_parts_per_delivery: int


@dataclass(slots=True, frozen=True)
class StorageLimits:
    fuel_max: float
    weapons_max: Dict[int, int]
    spare_parts_max: int


# ---------------------------------------------------------------------------
# Airbase
# ---------------------------------------------------------------------------

NUM_WEAPON_TYPES = 4


@dataclass(slots=True)
class Airbase:
    id: int
    name: str
    runways: int
    parking_slots: int
    maintenance_slots: int
    fuel: float
    fuel_max: float
    weapons: Dict[int, int]
    weapons_max: Dict[int, int]
    spare_parts: int
    spare_parts_max: int
    aircraft_docked: List[int] = field(default_factory=list)

    def get_features(self) -> np.ndarray:
        fuel_norm = self.fuel / self.fuel_max if self.fuel_max > 0 else 0.0
        spare_norm = (
            self.spare_parts / self.spare_parts_max
            if self.spare_parts_max > 0
            else 0.0
        )
        weapon_norms = [
            self.weapons.get(wid, 0) / self.weapons_max[wid]
            if self.weapons_max.get(wid, 0) > 0
            else 0.0
            for wid in sorted(self.weapons_max)
        ]
        return np.array(
            [
                self.runways,
                self.parking_slots,
                self.maintenance_slots,
                fuel_norm,
                spare_norm,
            ]
            + weapon_norms,
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _normalize_int_keys(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            int(k) if isinstance(k, str) and k.isdigit() else k: _normalize_int_keys(v)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_normalize_int_keys(item) for item in data]
    return data


def _load_yaml(source: Union[str, Path, dict]) -> dict:
    if isinstance(source, dict):
        return _normalize_int_keys(source)
    path = Path(source)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _normalize_int_keys(raw)


# ---------------------------------------------------------------------------
# Config parsing – strict, no fallbacks
# ---------------------------------------------------------------------------

_REQUIRED_BASE_KEYS = frozenset({
    "name", "runways", "parking_slots", "maintenance_slots", "storage",
})

_REQUIRED_STORAGE_KEYS = frozenset({"fuel", "weapons", "spare_parts"})

_REQUIRED_FUEL_KEYS = frozenset({"start", "max"})

_REQUIRED_SPARE_PARTS_KEYS = frozenset({"start", "max"})

_REQUIRED_WEAPON_STORAGE_KEYS = frozenset({"start", "max"})

_REQUIRED_TRANSFER_KEYS = frozenset({"duration_hours", "fuel_cost"})

_REQUIRED_DELIVERY_KEYS = frozenset({
    "max_fuel_per_delivery", "max_weapons_per_delivery",
    "max_spare_parts_per_delivery",
})


def _check_keys(data: dict, required: frozenset, label: str) -> None:
    missing = required - data.keys()
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")


def load_bases_config(source: Union[str, Path, dict] = "config.yml") -> dict:
    """Load and validate the *bases* section. Raises KeyError on any missing field."""
    config = _load_yaml(source)
    bases = config["bases"]

    for bid, bdata in bases.items():
        _check_keys(bdata, _REQUIRED_BASE_KEYS, f"bases.{bid}")
        storage = bdata["storage"]
        _check_keys(storage, _REQUIRED_STORAGE_KEYS, f"bases.{bid}.storage")
        _check_keys(storage["fuel"], _REQUIRED_FUEL_KEYS, f"bases.{bid}.storage.fuel")
        _check_keys(
            storage["spare_parts"],
            _REQUIRED_SPARE_PARTS_KEYS,
            f"bases.{bid}.storage.spare_parts",
        )
        for wid, wdata in storage["weapons"].items():
            _check_keys(
                wdata,
                _REQUIRED_WEAPON_STORAGE_KEYS,
                f"bases.{bid}.storage.weapons.{wid}",
            )

    return bases


def load_transfer_config(source: Union[str, Path, dict] = "config.yml") -> dict:
    """Load and validate the *transfer* section. Raises KeyError on any missing field."""
    config = _load_yaml(source)
    transfer = config["transfer"]
    _check_keys(transfer, _REQUIRED_TRANSFER_KEYS, "transfer")
    return transfer


def load_delivery_config(source: Union[str, Path, dict] = "config.yml") -> dict:
    """Load and validate the *delivery* section. Raises KeyError on any missing field."""
    config = _load_yaml(source)
    delivery = config["delivery"]
    _check_keys(delivery["bounds"], _REQUIRED_DELIVERY_KEYS, "delivery.bounds")
    return delivery


# ---------------------------------------------------------------------------
# Typed parsers for each sub-section
# ---------------------------------------------------------------------------

def parse_transfer(transfer_cfg: dict) -> TransferConfig:
    return TransferConfig(
        duration_hours=transfer_cfg["duration_hours"],
        fuel_cost=transfer_cfg["fuel_cost"],
    )


def parse_delivery_bounds(delivery_cfg: dict) -> DeliveryBounds:
    b = delivery_cfg["bounds"]
    return DeliveryBounds(
        max_fuel_per_delivery=b["max_fuel_per_delivery"],
        max_weapons_per_delivery=b["max_weapons_per_delivery"],
        max_spare_parts_per_delivery=b["max_spare_parts_per_delivery"],
    )


def parse_storage_limits(base_data: dict) -> StorageLimits:
    s = base_data["storage"]
    return StorageLimits(
        fuel_max=s["fuel"]["max"],
        weapons_max={wid: wd["max"] for wid, wd in s["weapons"].items()},
        spare_parts_max=s["spare_parts"]["max"],
    )


# ---------------------------------------------------------------------------
# Airbase builder
# ---------------------------------------------------------------------------

def build_airbases(
    source: Union[str, Path, dict] = "config.yml",
) -> tuple[Dict[int, Airbase], DeliveryBounds, TransferConfig]:
    """Build all airbases, delivery bounds, and transfer config. No fallbacks."""
    bases_cfg = load_bases_config(source)
    delivery_cfg = load_delivery_config(source)
    transfer_cfg = load_transfer_config(source)

    delivery_bounds = parse_delivery_bounds(delivery_cfg)
    transfer = parse_transfer(transfer_cfg)

    airbases: Dict[int, Airbase] = {}
    for bid, bd in bases_cfg.items():
        s = bd["storage"]
        airbases[bid] = Airbase(
            id=bid,
            name=bd["name"],
            runways=bd["runways"],
            parking_slots=bd["parking_slots"],
            maintenance_slots=bd["maintenance_slots"],
            fuel=s["fuel"]["start"],
            fuel_max=s["fuel"]["max"],
            weapons={wid: wd["start"] for wid, wd in s["weapons"].items()},
            weapons_max={wid: wd["max"] for wid, wd in s["weapons"].items()},
            spare_parts=s["spare_parts"]["start"],
            spare_parts_max=s["spare_parts"]["max"],
        )

    return airbases, delivery_bounds, transfer
