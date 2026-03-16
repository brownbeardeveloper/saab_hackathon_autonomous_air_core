from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class AircraftStatus(IntEnum):
    AVAILABLE = 0
    IN_TRANSIT = 1
    MAINTENANCE = 2
    ON_MISSION = 3


# ---------------------------------------------------------------------------
# Config dataclasses (frozen – these are read-only reference data)
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class AircraftBounds:
    max_fuel: float
    max_flight_hours: float
    hardpoint_slots: Dict[int, int]
    allowed_weapons: Dict[int, List[int]]
    equipment_slots: Dict[int, int]


@dataclass(slots=True, frozen=True)
class MaintenanceConfig:
    full_service_interval_hours: float
    full_service_duration_hours: float
    requires_maintenance_slot: bool


@dataclass(slots=True, frozen=True)
class WeaponType:
    id: int
    name: str
    description: str
    install_time_hours: float
    uninstall_time_hours: float


@dataclass(slots=True, frozen=True)
class EquipmentType:
    id: int
    name: str
    description: str
    install_time_hours: float
    uninstall_time_hours: float


# ---------------------------------------------------------------------------
# Aircraft
# ---------------------------------------------------------------------------

NUM_HARDPOINTS = 6
HARDPOINT_GROUPS = (
    (0, 1),
    (2, 3),
    (4, 5),
)
HARDPOINT_GROUP_LABELS = (
    "H1/H2",
    "H3/H4",
    "H5/H6",
)
NUM_WEAPON_GROUPS = len(HARDPOINT_GROUPS)
NUM_EQUIPMENT_SLOTS = 4


@dataclass(slots=True)
class Aircraft:
    id: int
    name: str
    base_id: int
    total_flight_hours: float
    flight_hours_since_last_mission: float
    fuel_level: float
    weapons: np.ndarray
    equipment: np.ndarray
    last_mission_hours: float
    status: AircraftStatus = AircraftStatus.AVAILABLE

    def get_features(self, bounds: AircraftBounds) -> np.ndarray:
        return np.concatenate([
            np.array([
                self.base_id,
                self.status,
                self.fuel_level / bounds.max_fuel,
                self.total_flight_hours / bounds.max_flight_hours,
                self.flight_hours_since_last_mission / bounds.max_flight_hours,
                self.last_mission_hours / bounds.max_flight_hours,
            ], dtype=np.float32),
            self.weapons,
            self.equipment,
        ])


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

_REQUIRED_BOUNDS_KEYS = frozenset({
    "max_fuel", "max_flight_hours",
    "hardpoint_slots_per_aircraft", "allowed_weapons_per_aircraft",
    "equipment_slots_per_aircraft",
})

_REQUIRED_MAINTENANCE_KEYS = frozenset({
    "full_service_interval_hours", "full_service_duration_hours",
    "requires_maintenance_slot",
})

_REQUIRED_TYPE_KEYS = frozenset({
    "name", "description", "install_time_hours", "uninstall_time_hours",
})

_REQUIRED_FLEET_KEYS = frozenset({
    "name", "total_flight_hours", "flight_hours_since_last_mission",
    "fuel_level", "weapons", "start_airbase", "equipment", "last_mission_hours",
})


def _check_keys(data: dict, required: frozenset, label: str) -> None:
    missing = required - data.keys()
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")


def load_aircraft_config(source: Union[str, Path, dict] = "config.yml") -> dict:
    """Load and validate the *aircraft* section. Raises KeyError on any missing field."""
    config = _load_yaml(source)
    ac = config["aircraft"]

    _check_keys(ac["bounds"], _REQUIRED_BOUNDS_KEYS, "aircraft.bounds")
    _check_keys(ac["maintenance"], _REQUIRED_MAINTENANCE_KEYS, "aircraft.maintenance")

    for wid, wdata in ac["weapons"].items():
        _check_keys(wdata, _REQUIRED_TYPE_KEYS, f"aircraft.weapons.{wid}")

    for eid, edata in ac["equipment_types"].items():
        _check_keys(edata, _REQUIRED_TYPE_KEYS, f"aircraft.equipment_types.{eid}")

    for ac_id, ac_data in ac["fleet"].items():
        _check_keys(ac_data, _REQUIRED_FLEET_KEYS, f"aircraft.fleet.{ac_id}")

    return ac


# ---------------------------------------------------------------------------
# Typed parsers for each sub-section
# ---------------------------------------------------------------------------

def parse_bounds(aircraft_cfg: dict) -> AircraftBounds:
    b = aircraft_cfg["bounds"]
    return AircraftBounds(
        max_fuel=b["max_fuel"],
        max_flight_hours=b["max_flight_hours"],
        hardpoint_slots=b["hardpoint_slots_per_aircraft"],
        allowed_weapons=b["allowed_weapons_per_aircraft"],
        equipment_slots=b["equipment_slots_per_aircraft"],
    )


def parse_maintenance(aircraft_cfg: dict) -> MaintenanceConfig:
    m = aircraft_cfg["maintenance"]
    return MaintenanceConfig(
        full_service_interval_hours=m["full_service_interval_hours"],
        full_service_duration_hours=m["full_service_duration_hours"],
        requires_maintenance_slot=m["requires_maintenance_slot"],
    )


def parse_weapon_types(aircraft_cfg: dict) -> Dict[int, WeaponType]:
    return {
        wid: WeaponType(
            id=wid,
            name=w["name"],
            description=w["description"],
            install_time_hours=w["install_time_hours"],
            uninstall_time_hours=w["uninstall_time_hours"],
        )
        for wid, w in aircraft_cfg["weapons"].items()
    }


def parse_equipment_types(aircraft_cfg: dict) -> Dict[int, EquipmentType]:
    return {
        eid: EquipmentType(
            id=eid,
            name=e["name"],
            description=e["description"],
            install_time_hours=e["install_time_hours"],
            uninstall_time_hours=e["uninstall_time_hours"],
        )
        for eid, e in aircraft_cfg["equipment_types"].items()
    }


# ---------------------------------------------------------------------------
# Fleet builder
# ---------------------------------------------------------------------------

def build_aircraft_fleet(
    source: Union[str, Path, dict] = "config.yml",
) -> tuple[Dict[int, Aircraft], AircraftBounds]:
    """Build the full fleet and its bounds from config. No fallbacks."""
    aircraft_cfg = load_aircraft_config(source)
    bounds = parse_bounds(aircraft_cfg)

    fleet: Dict[int, Aircraft] = {}
    for ac_id, d in aircraft_cfg["fleet"].items():
        weapons = np.array(
            [
                float(d["weapons"][s]) if d["weapons"][s] is not None else 0.0
                for s in sorted(d["weapons"])
            ],
            dtype=np.float32,
        )
        equipment = np.array(
            [float(d["equipment"][s]) for s in sorted(d["equipment"])],
            dtype=np.float32,
        )

        fleet[ac_id] = Aircraft(
            id=ac_id,
            name=d["name"],
            base_id=d["start_airbase"],
            total_flight_hours=d["total_flight_hours"],
            flight_hours_since_last_mission=d["flight_hours_since_last_mission"],
            fuel_level=d["fuel_level"],
            weapons=weapons,
            equipment=equipment,
            last_mission_hours=d["last_mission_hours"],
        )

    return fleet, bounds
