from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config dataclasses (frozen – these are read-only reference data)
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class MissionBounds:
    max_aircraft_per_mission: int
    max_flight_hours_per_mission: float
    max_fuel_cost_per_mission: float
    max_weapon_required: Dict[int, int]


@dataclass(slots=True, frozen=True)
class MissionProfile:
    key: str
    name: str
    description: str
    weight: float
    flight_hours_range: Tuple[float, float]
    fuel_cost_range: Tuple[float, float]
    weapon_requirements: Dict[int, int]
    recommended_equipment: Tuple[int, ...]


@dataclass(slots=True, frozen=True)
class DiceConfig:
    extra_repair_time: Dict[int, float]


@dataclass(slots=True, frozen=True)
class RepairTimeCategory:
    label: str
    flight_hours_range: Tuple[float, float]
    distribution: Dict[int, float]


@dataclass(slots=True, frozen=True)
class RepairType:
    id: int
    name: str
    description: str
    spare_parts_required: int
    duration_hours: float


@dataclass(slots=True, frozen=True)
class FullServiceConfig:
    name: str
    description: str
    spare_parts_required: int
    duration_hours: float


# ---------------------------------------------------------------------------
# Mission (mutable runtime state)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Mission:
    id: int
    name: str
    description: str
    flight_hours: float
    fuel_cost: float
    weapon_requirements: Dict[int, int]
    recommended_equipment: Tuple[int, ...]
    assigned_aircraft: List[int]
    completed: bool = False

    def get_features(self, bounds: MissionBounds) -> np.ndarray:
        hours_norm = (
            self.flight_hours / bounds.max_flight_hours_per_mission
            if bounds.max_flight_hours_per_mission > 0
            else 0.0
        )
        fuel_norm = (
            self.fuel_cost / bounds.max_fuel_cost_per_mission
            if bounds.max_fuel_cost_per_mission > 0
            else 0.0
        )
        weapon_norms = [
            self.weapon_requirements.get(wid, 0) / bounds.max_weapon_required[wid]
            if bounds.max_weapon_required.get(wid, 0) > 0
            else 0.0
            for wid in sorted(bounds.max_weapon_required)
        ]
        return np.array(
            [
                hours_norm,
                fuel_norm,
                len(self.assigned_aircraft) / bounds.max_aircraft_per_mission,
                float(self.completed),
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

_REQUIRED_MISSION_BOUNDS_KEYS = frozenset({
    "max_aircraft_per_mission", "max_flight_hours_per_mission",
    "max_fuel_cost_per_mission", "max_weapon_required_per_mission",
})

_REQUIRED_MISSION_KEYS = frozenset({"bounds", "profiles"})

_REQUIRED_MISSION_PROFILE_KEYS = frozenset({
    "name",
    "description",
    "weight",
    "flight_hours_range",
    "fuel_cost_range",
    "weapon_requirements",
    "recommended_equipment",
})

_REQUIRED_DICE_KEYS = frozenset({"extra_repair_time"})

_REQUIRED_REPAIR_CATEGORY_KEYS = frozenset({
    "flight_hours_since_last_mission_range", "distribution",
})

_REQUIRED_REPAIR_TYPE_KEYS = frozenset({
    "name", "description", "spare_parts_required", "duration_hours",
})

_REQUIRED_FULL_SERVICE_KEYS = frozenset({
    "name", "description", "spare_parts_required", "duration_hours",
})

_REQUIRED_REPAIRS_KEYS = frozenset({
    "extra_repair_time", "repair_types", "full_service",
})


def _check_keys(data: dict, required: frozenset, label: str) -> None:
    missing = required - data.keys()
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")


def load_missions_config(source: Union[str, Path, dict] = "config.yml") -> dict:
    """Load and validate the *missions* section. Raises KeyError on any missing field."""
    config = _load_yaml(source)
    missions = config["missions"]
    _check_keys(missions, _REQUIRED_MISSION_KEYS, "missions")
    _check_keys(missions["bounds"], _REQUIRED_MISSION_BOUNDS_KEYS, "missions.bounds")
    for profile_name, profile_data in missions["profiles"].items():
        _check_keys(
            profile_data,
            _REQUIRED_MISSION_PROFILE_KEYS,
            f"missions.profiles.{profile_name}",
        )
    return missions


def load_dice_config(source: Union[str, Path, dict] = "config.yml") -> dict:
    """Load and validate the *dice* section. Raises KeyError on any missing field."""
    config = _load_yaml(source)
    dice = config["dice"]
    _check_keys(dice, _REQUIRED_DICE_KEYS, "dice")
    return dice


def load_repairs_config(source: Union[str, Path, dict] = "config.yml") -> dict:
    """Load and validate the *repairs* section. Raises KeyError on any missing field."""
    config = _load_yaml(source)
    repairs = config["repairs"]
    _check_keys(repairs, _REQUIRED_REPAIRS_KEYS, "repairs")

    for cat_name, cat_data in repairs["extra_repair_time"].items():
        _check_keys(
            cat_data,
            _REQUIRED_REPAIR_CATEGORY_KEYS,
            f"repairs.extra_repair_time.{cat_name}",
        )

    for rt_id, rt_data in repairs["repair_types"].items():
        _check_keys(
            rt_data,
            _REQUIRED_REPAIR_TYPE_KEYS,
            f"repairs.repair_types.{rt_id}",
        )

    _check_keys(
        repairs["full_service"],
        _REQUIRED_FULL_SERVICE_KEYS,
        "repairs.full_service",
    )

    return repairs


# ---------------------------------------------------------------------------
# Typed parsers for each sub-section
# ---------------------------------------------------------------------------

def parse_mission_bounds(missions_cfg: dict) -> MissionBounds:
    b = missions_cfg["bounds"]
    return MissionBounds(
        max_aircraft_per_mission=b["max_aircraft_per_mission"],
        max_flight_hours_per_mission=b["max_flight_hours_per_mission"],
        max_fuel_cost_per_mission=b["max_fuel_cost_per_mission"],
        max_weapon_required=b["max_weapon_required_per_mission"],
    )


def parse_mission_profiles(
    missions_cfg: dict,
    mission_bounds: MissionBounds,
) -> Dict[str, MissionProfile]:
    profiles: Dict[str, MissionProfile] = {}

    for profile_key, profile_cfg in missions_cfg["profiles"].items():
        hours_lo, hours_hi = tuple(profile_cfg["flight_hours_range"])
        fuel_lo, fuel_hi = tuple(profile_cfg["fuel_cost_range"])
        if hours_lo <= 0 or hours_hi < hours_lo:
            raise ValueError(
                f"missions.profiles.{profile_key}.flight_hours_range must be a positive [min, max] pair"
            )
        if fuel_lo <= 0 or fuel_hi < fuel_lo:
            raise ValueError(
                f"missions.profiles.{profile_key}.fuel_cost_range must be a positive [min, max] pair"
            )
        if hours_hi > mission_bounds.max_flight_hours_per_mission:
            raise ValueError(
                f"missions.profiles.{profile_key}.flight_hours_range exceeds mission bounds"
            )
        if fuel_hi > mission_bounds.max_fuel_cost_per_mission:
            raise ValueError(
                f"missions.profiles.{profile_key}.fuel_cost_range exceeds mission bounds"
            )

        weapon_requirements = {
            int(wid): int(qty)
            for wid, qty in profile_cfg["weapon_requirements"].items()
            if int(qty) > 0
        }
        for wid, qty in weapon_requirements.items():
            if qty > mission_bounds.max_weapon_required.get(wid, 0):
                raise ValueError(
                    f"missions.profiles.{profile_key}.weapon_requirements[{wid}] exceeds mission bounds"
                )

        recommended_equipment = tuple(
            sorted({int(eid) for eid in profile_cfg["recommended_equipment"]})
        )
        weight = float(profile_cfg["weight"])
        if weight <= 0:
            raise ValueError(
                f"missions.profiles.{profile_key}.weight must be greater than 0"
            )

        profiles[profile_key] = MissionProfile(
            key=profile_key,
            name=profile_cfg["name"],
            description=profile_cfg["description"],
            weight=weight,
            flight_hours_range=(float(hours_lo), float(hours_hi)),
            fuel_cost_range=(float(fuel_lo), float(fuel_hi)),
            weapon_requirements=weapon_requirements,
            recommended_equipment=recommended_equipment,
        )

    if not profiles:
        raise ValueError("missions.profiles must contain at least one mission profile")

    return profiles


def parse_dice(dice_cfg: dict) -> DiceConfig:
    return DiceConfig(
        extra_repair_time=dice_cfg["extra_repair_time"],
    )


def parse_repair_time_categories(
    repairs_cfg: dict,
) -> Dict[str, RepairTimeCategory]:
    return {
        cat_name: RepairTimeCategory(
            label=cat_name,
            flight_hours_range=tuple(cat_data["flight_hours_since_last_mission_range"]),
            distribution=cat_data["distribution"],
        )
        for cat_name, cat_data in repairs_cfg["extra_repair_time"].items()
    }


def parse_repair_types(repairs_cfg: dict) -> Dict[int, RepairType]:
    return {
        rt_id: RepairType(
            id=rt_id,
            name=rt["name"],
            description=rt["description"],
            spare_parts_required=rt["spare_parts_required"],
            duration_hours=rt["duration_hours"],
        )
        for rt_id, rt in repairs_cfg["repair_types"].items()
    }


def parse_full_service(repairs_cfg: dict) -> FullServiceConfig:
    fs = repairs_cfg["full_service"]
    return FullServiceConfig(
        name=fs["name"],
        description=fs["description"],
        spare_parts_required=fs["spare_parts_required"],
        duration_hours=fs["duration_hours"],
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_mission_config(
    source: Union[str, Path, dict] = "config.yml",
) -> tuple[
    MissionBounds,
    Dict[str, MissionProfile],
    DiceConfig,
    Dict[str, RepairTimeCategory],
    Dict[int, RepairType],
    FullServiceConfig,
]:
    """Build all mission/repair config objects from YAML. No fallbacks."""
    missions_cfg = load_missions_config(source)
    dice_cfg = load_dice_config(source)
    repairs_cfg = load_repairs_config(source)

    mission_bounds = parse_mission_bounds(missions_cfg)
    mission_profiles = parse_mission_profiles(missions_cfg, mission_bounds)
    dice = parse_dice(dice_cfg)
    repair_categories = parse_repair_time_categories(repairs_cfg)
    repair_types = parse_repair_types(repairs_cfg)
    full_service = parse_full_service(repairs_cfg)

    return (
        mission_bounds,
        mission_profiles,
        dice,
        repair_categories,
        repair_types,
        full_service,
    )


def sample_mission(
    slot_id: int,
    mission_bounds: MissionBounds,
    mission_profiles: Dict[str, MissionProfile],
    rng: np.random.Generator,
) -> Mission:
    """Sample a mission from the configured profile mix."""
    profile_keys = list(mission_profiles)
    weights = np.array(
        [mission_profiles[key].weight for key in profile_keys],
        dtype=np.float64,
    )
    weights /= weights.sum()
    profile_idx = int(rng.choice(len(profile_keys), p=weights))
    profile = mission_profiles[profile_keys[profile_idx]]

    flight_hours = float(rng.uniform(*profile.flight_hours_range))
    fuel_cost = float(rng.uniform(*profile.fuel_cost_range))
    flight_hours = min(flight_hours, mission_bounds.max_flight_hours_per_mission)
    fuel_cost = min(fuel_cost, mission_bounds.max_fuel_cost_per_mission)

    return Mission(
        id=slot_id,
        name=profile.name,
        description=profile.description,
        flight_hours=flight_hours,
        fuel_cost=fuel_cost,
        weapon_requirements=dict(profile.weapon_requirements),
        recommended_equipment=profile.recommended_equipment,
        assigned_aircraft=[],
    )


def load_mission_manifest(source: Union[str, Path]) -> List[Mission]:
    """Load a concrete mission list from JSON."""
    path = Path(source)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    raw_missions = data.get("missions")
    if not isinstance(raw_missions, list) or not raw_missions:
        raise ValueError("Mission manifest must contain a non-empty 'missions' list")

    missions: List[Mission] = []
    for idx, raw in enumerate(raw_missions, start=1):
        weapon_requirements = {
            int(wid): int(payload["quantity"])
            for wid, payload in raw.get("weapon_requirements", {}).items()
        }
        recommended_equipment = tuple(
            int(item["id"]) if isinstance(item, dict) else int(item)
            for item in raw.get("recommended_equipment", [])
        )
        missions.append(
            Mission(
                id=int(raw.get("id", idx)),
                name=raw["name"],
                description=raw.get("description", ""),
                flight_hours=float(raw["flight_hours"]),
                fuel_cost=float(raw["fuel_cost"]),
                weapon_requirements=weapon_requirements,
                recommended_equipment=recommended_equipment,
                assigned_aircraft=[],
            )
        )

    return missions
