"""FleetEnv – Gymnasium environment for HRL / GNN aircraft fleet management.

Each step the agent selects actions for every aircraft simultaneously.
Time advances by a fixed increment; aircraft execute transfers, weapon /
equipment changes, missions, and undergo stochastic post-mission repairs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from aircraft import (
    Aircraft,
    AircraftBounds,
    AircraftStatus,
    HARDPOINT_GROUPS,
    MaintenanceConfig,
    NUM_WEAPON_GROUPS,
    build_aircraft_fleet,
    load_aircraft_config,
    parse_bounds,
    parse_maintenance,
    parse_weapon_types,
    parse_equipment_types,
    NUM_HARDPOINTS,
    NUM_EQUIPMENT_SLOTS,
)
from airbase import (
    Airbase,
    TransferConfig,
    DeliveryBounds,
    build_airbases,
    NUM_WEAPON_TYPES,
)
from missions import (
    Mission,
    MissionBounds,
    DiceConfig,
    RepairTimeCategory,
    RepairType,
    FullServiceConfig,
    build_mission_config,
    load_mission_manifest,
    sample_mission,
)
from action_masking import (
    ActionMasker,
    WEAPON_ACTION_KEEP,
    WEAPON_ACTION_UNLOAD,
    WEAPON_ACTION_WEAPON_OFFSET,
)


# ──────────────────────────────────────────────────────────────────────
# Reward constants (centralised for easy tuning)
# ──────────────────────────────────────────────────────────────────────

_R_MISSION_COMPLETE = 10.0
_R_MISSION_PARTIAL = 2.0
_R_ALL_DONE_BONUS = 50.0
_R_TRANSFER_COST = -0.5
_R_MINOR_MAINT = -1.0
_R_MAJOR_MAINT = -3.0
_R_FULL_SERVICE = -5.0
_R_NO_SPARE_PARTS = -2.0
_R_TIME_PENALTY = -0.01


# ──────────────────────────────────────────────────────────────────────
# YAML loader (one-shot, avoids re-reading the file per module)
# ──────────────────────────────────────────────────────────────────────

def _normalize_int_keys(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            int(k) if isinstance(k, str) and k.isdigit() else k: _normalize_int_keys(v)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_normalize_int_keys(item) for item in data]
    return data


def _load_full_config(source: Union[str, Path, dict]) -> dict:
    if isinstance(source, dict):
        return _normalize_int_keys(source)
    with open(source, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _normalize_int_keys(raw)


# ══════════════════════════════════════════════════════════════════════
# FleetEnv
# ══════════════════════════════════════════════════════════════════════

class FleetEnv(gym.Env):
    """Gymnasium environment for fleet management with action masking.

    Parameters
    ----------
    config_source : path to YAML or a pre-loaded dict.
    max_active_missions : how many missions can be pending at any time.
    time_step_hours : simulation hours per ``step()`` call.
    max_episode_hours : episode is truncated after this many sim-hours.
    delivery_interval_hours : bases receive resupply every N sim-hours.
    render_mode : optional Gymnasium render mode.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        config_source: Union[str, Path, dict] = "config.yml",
        mission_manifest_path: Optional[Union[str, Path]] = None,
        record_events: bool = False,
        max_active_missions: int = 5,
        time_step_hours: float = 1.0,
        max_episode_hours: float = 2000.0,
        delivery_interval_hours: float = 24.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_active_missions = max_active_missions
        self.time_step_hours = time_step_hours
        self.max_episode_hours = max_episode_hours
        self.delivery_interval_hours = delivery_interval_hours
        self.record_events = record_events
        self.mission_manifest_path = (
            Path(mission_manifest_path)
            if mission_manifest_path is not None
            else None
        )

        # ── load full config once ─────────────────────────────────────
        full_cfg = _load_full_config(config_source)
        self.total_missions: int = full_cfg["training"]["total_missions"]
        self.sim_toggles: dict = full_cfg.get("config", {})

        # ── build typed config objects via module builders ─────────────
        self._fleet_template, self.bounds = build_aircraft_fleet(full_cfg)
        ac_cfg = load_aircraft_config(full_cfg)
        self.maintenance_cfg: MaintenanceConfig = parse_maintenance(ac_cfg)
        self.weapon_types = parse_weapon_types(ac_cfg)
        self.equipment_types = parse_equipment_types(ac_cfg)

        self._base_template, self.delivery_bounds, self.transfer = build_airbases(full_cfg)

        (
            self.mission_bounds,
            self.mission_profiles,
            self.dice,
            self.repair_categories,
            self.repair_types,
            self.full_service_cfg,
        ) = build_mission_config(full_cfg)

        self._mission_manifest: List[Mission] = []
        if self.mission_manifest_path is not None:
            self._mission_manifest = load_mission_manifest(self.mission_manifest_path)
            self.total_missions = len(self._mission_manifest)

        self.n_aircraft = len(self._fleet_template)
        self.n_bases = len(self._base_template)

        # ── action masker ─────────────────────────────────────────────
        self.masker = ActionMasker(
            base_ids=list(self._base_template.keys()),
            weapon_type_ids=list(self.weapon_types.keys()),
            equipment_type_ids=list(self.equipment_types.keys()),
            num_mission_slots=max_active_missions + 1,  # slot 0 = idle
            bounds=self.bounds,
            transfer=self.transfer,
            mission_bounds=self.mission_bounds,
        )

        # ── action space (fleet-wide MultiDiscrete) ───────────────────
        per_ac_dims = (
            [self.masker.n_bases]
            + [self.masker.n_wpn] * NUM_WEAPON_GROUPS
            + [self.masker.n_eqp]
            + [self.masker.n_msn]
        )
        self._per_ac_dims = per_ac_dims
        self._n_sub = len(per_ac_dims)
        self.action_space = spaces.MultiDiscrete(per_ac_dims * self.n_aircraft)

        # ── observation space ─────────────────────────────────────────
        feat_ac = 6 + NUM_HARDPOINTS + NUM_EQUIPMENT_SLOTS     # 16
        feat_base = 5 + NUM_WEAPON_TYPES                       # 9
        feat_msn = 4 + NUM_WEAPON_TYPES                        # 8

        self.observation_space = spaces.Dict({
            "aircraft": spaces.Box(
                -np.inf, np.inf,
                shape=(self.n_aircraft, feat_ac),
                dtype=np.float32,
            ),
            "bases": spaces.Box(
                -np.inf, np.inf,
                shape=(self.n_bases, feat_base),
                dtype=np.float32,
            ),
            "missions": spaces.Box(
                0.0, 1.0,
                shape=(max_active_missions, feat_msn),
                dtype=np.float32,
            ),
            "action_mask": spaces.MultiBinary(
                self.n_aircraft * self.masker.flat_width
            ),
        })

        # ── mutable runtime state (populated by reset) ────────────────
        self.fleet: Dict[int, Aircraft] = {}
        self.bases: Dict[int, Airbase] = {}
        self.missions: Dict[int, Mission] = {}
        self.current_time: float = 0.0
        self.missions_completed: int = 0
        self._busy_until: Dict[int, float] = {}
        self._hours_at_last_service: Dict[int, float] = {}
        self._last_delivery_time: float = 0.0
        self._next_manifest_mission_idx: int = 0
        self.step_index: int = 0
        self.episode_events: List[Dict[str, Any]] = []
        self._mission_offer_log: List[Dict[str, Any]] = []
        self._mission_offer_ids: Dict[int, int] = {}
        self._next_mission_offer_id: int = 1
        self._aircraft_last_mission: Dict[int, Dict[str, Any]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.current_time = 0.0
        self.missions_completed = 0
        self._busy_until.clear()
        self._last_delivery_time = 0.0
        self._next_manifest_mission_idx = 0
        self.step_index = 0
        self.episode_events = []
        self._mission_offer_log = []
        self._mission_offer_ids = {}
        self._next_mission_offer_id = 1
        self._aircraft_last_mission = {}

        # deep-copy fleet
        self.fleet = {}
        for ac_id, ac in self._fleet_template.items():
            self.fleet[ac_id] = Aircraft(
                id=ac.id,
                name=ac.name,
                base_id=ac.base_id,
                total_flight_hours=ac.total_flight_hours,
                flight_hours_since_last_mission=ac.flight_hours_since_last_mission,
                fuel_level=ac.fuel_level,
                weapons=ac.weapons.copy(),
                equipment=ac.equipment.copy(),
                last_mission_hours=ac.last_mission_hours,
                status=AircraftStatus.AVAILABLE,
            )
            self._hours_at_last_service[ac_id] = ac.total_flight_hours

        # deep-copy bases & dock aircraft
        self.bases = {}
        for bid, b in self._base_template.items():
            self.bases[bid] = Airbase(
                id=b.id, name=b.name, runways=b.runways,
                parking_slots=b.parking_slots,
                maintenance_slots=b.maintenance_slots,
                fuel=b.fuel, fuel_max=b.fuel_max,
                weapons=dict(b.weapons), weapons_max=dict(b.weapons_max),
                spare_parts=b.spare_parts,
                spare_parts_max=b.spare_parts_max,
                aircraft_docked=[],
            )
        for ac_id, ac in self.fleet.items():
            self.bases[ac.base_id].aircraft_docked.append(ac_id)

        # seed mission queue (slots 1 … max_active_missions)
        self.missions = {}
        self._fill_mission_queue()

        return self._get_obs(), self._get_info()

    # ──────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        reward = 0.0
        self.step_index += 1
        ac_ids = sorted(self.fleet.keys())

        # ── parse fleet-wide action into per-aircraft dicts ───────────
        parsed: Dict[int, dict] = {}
        for i, ac_id in enumerate(ac_ids):
            off = i * self._n_sub
            a = action[off: off + self._n_sub]
            parsed[ac_id] = {
                "base_idx": int(a[0]),
                "wpn": [int(a[1 + h]) for h in range(NUM_WEAPON_GROUPS)],
                "eqp": int(a[1 + NUM_WEAPON_GROUPS]),
                "msn": int(a[2 + NUM_WEAPON_GROUPS]),
            }

        # ── execute actions for every AVAILABLE aircraft ──────────────
        for ac_id in ac_ids:
            ac = self.fleet[ac_id]
            if ac.status != AircraftStatus.AVAILABLE:
                continue
            act = parsed[ac_id]
            base = self.bases.get(ac.base_id)

            # ─── transfer ─────────────────────────────────────────────
            target_bid = self.masker.base_ids[act["base_idx"]]
            if target_bid != ac.base_id:
                dest = self.bases.get(target_bid)
                if (
                    dest is not None
                    and len(dest.aircraft_docked) < dest.parking_slots
                    and ac.fuel_level >= self.transfer.fuel_cost
                ):
                    if base is not None and ac_id in base.aircraft_docked:
                        base.aircraft_docked.remove(ac_id)
                    ac.fuel_level -= self.transfer.fuel_cost
                    ac.base_id = target_bid
                    dest.aircraft_docked.append(ac_id)
                    ac.status = AircraftStatus.IN_TRANSIT
                    self._busy_until[ac_id] = (
                        self.current_time + self.transfer.duration_hours
                    )
                    reward += _R_TRANSFER_COST
                    self._record_event(
                        "transfer",
                        aircraft_id=ac_id,
                        aircraft_name=ac.name,
                        from_base_id=base.id if base is not None else None,
                        from_base_name=base.name if base is not None else None,
                        to_base_id=dest.id,
                        to_base_name=dest.name,
                        fuel_cost=float(self.transfer.fuel_cost),
                        duration_hours=float(self.transfer.duration_hours),
                    )
                continue  # rest of actions meaningless while in transit

            # ─── weapon load / unload ─────────────────────────────────
            if base is not None:
                for group_idx, hardpoints in enumerate(HARDPOINT_GROUPS):
                    self._apply_weapon_group_action(
                        ac=ac,
                        base=base,
                        action_idx=act["wpn"][group_idx],
                        hardpoints=hardpoints,
                    )

            # ─── equipment toggle ─────────────────────────────────────
            if not self.sim_toggles.get("skip_equipment", False):
                eqp_act = act["eqp"]
                if eqp_act > 0:
                    eqp_id = self.masker.equipment_ids[eqp_act - 1]
                    slot = eqp_id - 1
                    if 0 <= slot < NUM_EQUIPMENT_SLOTS:
                        ac.equipment[slot] = 1.0 - ac.equipment[slot]

            # ─── mission assignment ───────────────────────────────────
            if self.sim_toggles.get("skip_mission", False):
                continue

            msn_idx = act["msn"]
            if msn_idx <= 0 or msn_idx not in self.missions:
                continue

            mission = self.missions[msn_idx]
            if mission.completed:
                continue
            if len(mission.assigned_aircraft) >= self.mission_bounds.max_aircraft_per_mission:
                continue
            if ac.fuel_level < mission.fuel_cost:
                continue

            # commit to mission
            loadout_before = self._aircraft_loadout_snapshot(ac)
            mission.assigned_aircraft.append(ac_id)
            ac.fuel_level -= mission.fuel_cost
            ac.total_flight_hours += mission.flight_hours
            ac.flight_hours_since_last_mission += mission.flight_hours
            ac.last_mission_hours = mission.flight_hours
            ac.status = AircraftStatus.ON_MISSION
            self._busy_until[ac_id] = self.current_time + mission.flight_hours

            # consume required weapons from hardpoints
            weapons_met = True
            consumed_weapons: Dict[int, int] = {}
            for wid, qty_needed in mission.weapon_requirements.items():
                consumed = 0
                for hp in range(NUM_HARDPOINTS):
                    if consumed >= qty_needed:
                        break
                    if int(ac.weapons[hp]) == wid:
                        ac.weapons[hp] = 0.0
                        consumed += 1
                if consumed < qty_needed:
                    weapons_met = False
                if consumed > 0:
                    consumed_weapons[wid] = consumed

            active_equipment = tuple(
                eid
                for eid in sorted(self.equipment_types)
                if ac.equipment[eid - 1] > 0.5
            )
            equipment_match = all(
                eid in active_equipment for eid in mission.recommended_equipment
            )
            mission_success = weapons_met and equipment_match
            mission_outcome = "success" if mission_success else "partial"
            mission_offer_id = self._mission_offer_ids.get(msn_idx)

            mission_event = {
                "mission_offer_id": mission_offer_id,
                "mission_slot": msn_idx,
                "mission_name": mission.name,
                "mission_description": mission.description,
                "aircraft_id": ac_id,
                "aircraft_name": ac.name,
                "base_id": ac.base_id,
                "base_name": self.bases[ac.base_id].name if ac.base_id in self.bases else None,
                "flight_hours": float(mission.flight_hours),
                "fuel_cost": float(mission.fuel_cost),
                "reward_awarded": float(
                    _R_MISSION_COMPLETE if mission_success else _R_MISSION_PARTIAL
                ),
                "outcome": mission_outcome,
                "weapons_met": weapons_met,
                "equipment_match": equipment_match,
                "weapon_requirements": dict(mission.weapon_requirements),
                "recommended_equipment": list(mission.recommended_equipment),
                "loadout_before": loadout_before,
                "active_equipment": list(active_equipment),
                "consumed_weapons": consumed_weapons,
            }
            self._aircraft_last_mission[ac_id] = {
                **mission_event,
                "mission_end_time": float(self._busy_until[ac_id]),
            }
            self._record_event("mission_execution", **mission_event)

            mission.completed = True
            self.missions_completed += 1
            reward += _R_MISSION_COMPLETE if mission_success else _R_MISSION_PARTIAL

        # ── advance simulation clock ──────────────────────────────────
        self.current_time += self.time_step_hours

        # ── resolve busy timers (transit / mission / maintenance) ─────
        reward += self._resolve_busy_aircraft()

        # ── periodic base resupply ────────────────────────────────────
        if self.current_time - self._last_delivery_time >= self.delivery_interval_hours:
            self._process_deliveries()
            self._last_delivery_time = self.current_time

        # ── recycle completed missions, refill queue ──────────────────
        spent = [mid for mid, m in self.missions.items() if m.completed]
        for mid in spent:
            del self.missions[mid]
            self._mission_offer_ids.pop(mid, None)
        self._fill_mission_queue()

        # ── small per-step cost to encourage speed ────────────────────
        reward += _R_TIME_PENALTY

        # ── termination / truncation ──────────────────────────────────
        terminated = self.missions_completed >= self.total_missions
        truncated = self.current_time >= self.max_episode_hours
        if terminated:
            reward += _R_ALL_DONE_BONUS

        if terminated or truncated:
            self._record_unfinished_missions(outcome="unresolved")

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _resolve_busy_aircraft(self) -> float:
        """Move aircraft whose timers have expired back to AVAILABLE
        (or into MAINTENANCE after a mission). Returns reward delta."""
        reward = 0.0
        for ac_id in list(self._busy_until):
            if self.current_time < self._busy_until[ac_id]:
                continue
            ac = self.fleet[ac_id]

            if ac.status == AircraftStatus.ON_MISSION:
                # post-mission repair roll
                if not self.sim_toggles.get("skip_repair", False):
                    repair = self._roll_repair(ac.flight_hours_since_last_mission)
                    if repair.duration_hours > 0:
                        base = self.bases.get(ac.base_id)
                        mission_ctx = self._aircraft_last_mission.get(ac_id, {})
                        if base is not None and base.spare_parts >= repair.spare_parts_required:
                            base.spare_parts -= repair.spare_parts_required
                            ac.status = AircraftStatus.MAINTENANCE
                            self._busy_until[ac_id] = (
                                self.current_time + repair.duration_hours
                            )
                            ac.flight_hours_since_last_mission = 0.0
                            reward += _R_MINOR_MAINT if repair.id <= 2 else _R_MAJOR_MAINT
                            self._record_event(
                                "maintenance",
                                aircraft_id=ac_id,
                                aircraft_name=ac.name,
                                base_id=base.id if base is not None else None,
                                base_name=base.name if base is not None else None,
                                maintenance_name=repair.name,
                                maintenance_type="repair",
                                spare_parts_used=int(repair.spare_parts_required),
                                duration_hours=float(repair.duration_hours),
                                spare_parts_available=True,
                                triggered_by_mission=mission_ctx.get("mission_name"),
                            )
                            continue
                        else:
                            ac.status = AircraftStatus.MAINTENANCE
                            self._busy_until[ac_id] = (
                                self.current_time + repair.duration_hours * 1.5
                            )
                            reward += _R_NO_SPARE_PARTS
                            self._record_event(
                                "maintenance",
                                aircraft_id=ac_id,
                                aircraft_name=ac.name,
                                base_id=base.id if base is not None else None,
                                base_name=base.name if base is not None else None,
                                maintenance_name=repair.name,
                                maintenance_type="repair",
                                spare_parts_used=0,
                                duration_hours=float(repair.duration_hours * 1.5),
                                spare_parts_available=False,
                                triggered_by_mission=mission_ctx.get("mission_name"),
                            )
                            continue

                # full-service trigger
                if (
                    not self.sim_toggles.get("skip_full_service", False)
                    and (ac.total_flight_hours
                         - self._hours_at_last_service.get(ac_id, 0.0))
                    >= self.maintenance_cfg.full_service_interval_hours
                ):
                    base = self.bases.get(ac.base_id)
                    had_spares = (
                        base is not None
                        and base.spare_parts >= self.full_service_cfg.spare_parts_required
                    )
                    if had_spares:
                        base.spare_parts -= self.full_service_cfg.spare_parts_required
                    ac.status = AircraftStatus.MAINTENANCE
                    self._busy_until[ac_id] = (
                        self.current_time + self.full_service_cfg.duration_hours
                    )
                    self._hours_at_last_service[ac_id] = ac.total_flight_hours
                    reward += _R_FULL_SERVICE
                    self._record_event(
                        "maintenance",
                        aircraft_id=ac_id,
                        aircraft_name=ac.name,
                        base_id=base.id if base is not None else None,
                        base_name=base.name if base is not None else None,
                        maintenance_name=self.full_service_cfg.name,
                        maintenance_type="full_service",
                        spare_parts_used=(
                            int(self.full_service_cfg.spare_parts_required)
                            if had_spares
                            else 0
                        ),
                        duration_hours=float(self.full_service_cfg.duration_hours),
                        spare_parts_available=had_spares,
                        triggered_by_mission=self._aircraft_last_mission.get(ac_id, {}).get("mission_name"),
                    )
                    continue

            # transit done, maintenance done, or no repair needed
            ac.status = AircraftStatus.AVAILABLE
            del self._busy_until[ac_id]

        return reward

    # ── mission generation ────────────────────────────────────────────

    def _generate_mission(self, slot_id: int) -> Mission:
        if self._mission_manifest:
            if self._next_manifest_mission_idx >= len(self._mission_manifest):
                raise IndexError("No more missions available in mission manifest")
            template = self._mission_manifest[self._next_manifest_mission_idx]
            self._next_manifest_mission_idx += 1
            return replace(
                template,
                id=slot_id,
                assigned_aircraft=[],
                completed=False,
            )

        return sample_mission(
            slot_id=slot_id,
            mission_bounds=self.mission_bounds,
            mission_profiles=self.mission_profiles,
            rng=self.np_random,
        )

    def _fill_mission_queue(self) -> None:
        remaining = self.total_missions - (self.missions_completed + len(self.missions))
        for slot in range(1, self.max_active_missions + 1):
            if remaining <= 0:
                break
            if slot not in self.missions:
                self.missions[slot] = self._generate_mission(slot)
                mission = self.missions[slot]
                mission_offer_id = self._next_mission_offer_id
                self._next_mission_offer_id += 1
                self._mission_offer_ids[slot] = mission_offer_id
                offer = {
                    "mission_offer_id": mission_offer_id,
                    "mission_slot": slot,
                    "mission_name": mission.name,
                    "mission_description": mission.description,
                    "flight_hours": float(mission.flight_hours),
                    "fuel_cost": float(mission.fuel_cost),
                    "weapon_requirements": dict(mission.weapon_requirements),
                    "recommended_equipment": list(mission.recommended_equipment),
                }
                self._mission_offer_log.append(offer)
                self._record_event("mission_offered", **offer)
                remaining -= 1

    # ── repair dice roll ──────────────────────────────────────────────

    def _roll_repair(self, accumulated_hours: float) -> RepairType:
        cat = self._find_repair_category(accumulated_hours)
        if cat is None:
            return self.repair_types[1]
        dice_ids = sorted(cat.distribution.keys())
        probs = np.array([cat.distribution[d] for d in dice_ids], dtype=np.float64)
        probs /= probs.sum()  # safety normalisation
        chosen = int(self.np_random.choice(dice_ids, p=probs))
        return self.repair_types[chosen]

    def _find_repair_category(self, hours: float) -> Optional[RepairTimeCategory]:
        for cat in self.repair_categories.values():
            lo, hi = cat.flight_hours_range
            if lo <= hours < hi:
                return cat
        return None

    # ── periodic base resupply ────────────────────────────────────────

    def _process_deliveries(self) -> None:
        db = self.delivery_bounds
        for base in self.bases.values():
            base.fuel = min(base.fuel + db.max_fuel_per_delivery, base.fuel_max)
            for wid, mx_del in db.max_weapons_per_delivery.items():
                cur = base.weapons.get(wid, 0)
                cap = base.weapons_max.get(wid, 0)
                base.weapons[wid] = min(cur + mx_del, cap)
            base.spare_parts = min(
                base.spare_parts + db.max_spare_parts_per_delivery,
                base.spare_parts_max,
            )

    # ── observations / info ───────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return a flat boolean action mask for MaskablePPO-style agents."""
        return self.masker.flat_mask_matrix(
            self.fleet, self.bases, self.missions,
        ).astype(bool).flatten()

    def _get_obs(self) -> dict:
        ac_ids = sorted(self.fleet)
        base_ids = sorted(self.bases)

        ac_feats = np.stack(
            [self.fleet[aid].get_features(self.bounds) for aid in ac_ids]
        )
        base_feats = np.stack(
            [self.bases[bid].get_features() for bid in base_ids]
        )

        msn_feats = np.zeros(
            (self.max_active_missions, 4 + NUM_WEAPON_TYPES),
            dtype=np.float32,
        )
        for slot, mission in sorted(self.missions.items()):
            idx = slot - 1
            if 0 <= idx < self.max_active_missions and not mission.completed:
                msn_feats[idx] = mission.get_features(self.mission_bounds)

        return {
            "aircraft": ac_feats,
            "bases": base_feats,
            "missions": msn_feats,
            "action_mask": self.action_masks().astype(np.int8),
        }

    def _get_info(self) -> dict:
        return {
            "time_hours": self.current_time,
            "step_index": self.step_index,
            "missions_completed": self.missions_completed,
            "missions_remaining": self.total_missions - self.missions_completed,
            "active_missions": len(self.missions),
            "fleet_status": {
                ac_id: ac.status.name for ac_id, ac in self.fleet.items()
            },
            "episode_events": deepcopy(self.episode_events) if self.record_events else None,
        }

    def _record_event(self, event_type: str, **payload: Any) -> None:
        if not self.record_events:
            return
        self.episode_events.append(
            {
                "event_type": event_type,
                "time_hours": float(self.current_time),
                "step_index": self.step_index,
                **payload,
            }
        )

    def _aircraft_loadout_snapshot(self, ac: Aircraft) -> Dict[str, Any]:
        return {
            "weapons": [int(wid) for wid in ac.weapons],
            "equipment": [
                eid
                for eid in sorted(self.equipment_types)
                if ac.equipment[eid - 1] > 0.5
            ],
        }

    def _hardpoint_group_allows_weapon(
        self,
        hardpoints: tuple[int, ...],
        weapon_id: int,
    ) -> bool:
        return all(
            weapon_id in self.bounds.allowed_weapons.get(hardpoint + 1, [])
            for hardpoint in hardpoints
        )

    def _unload_hardpoint(self, ac: Aircraft, base: Airbase, hardpoint: int) -> None:
        current = int(ac.weapons[hardpoint])
        if current <= 0:
            return
        base.weapons[current] = base.weapons.get(current, 0) + 1
        ac.weapons[hardpoint] = 0.0

    def _apply_weapon_group_action(
        self,
        ac: Aircraft,
        base: Airbase,
        action_idx: int,
        hardpoints: tuple[int, ...],
    ) -> None:
        if action_idx == WEAPON_ACTION_KEEP:
            return

        if action_idx == WEAPON_ACTION_UNLOAD:
            for hardpoint in hardpoints:
                self._unload_hardpoint(ac, base, hardpoint)
            return

        weapon_offset = action_idx - WEAPON_ACTION_WEAPON_OFFSET
        if weapon_offset < 0 or weapon_offset >= len(self.masker.weapon_ids):
            return

        desired_weapon_id = self.masker.weapon_ids[weapon_offset]
        if not self._hardpoint_group_allows_weapon(hardpoints, desired_weapon_id):
            return

        required_stock = sum(
            1
            for hardpoint in hardpoints
            if int(ac.weapons[hardpoint]) != desired_weapon_id
        )
        if base.weapons.get(desired_weapon_id, 0) < required_stock:
            return

        for hardpoint in hardpoints:
            current = int(ac.weapons[hardpoint])
            if current == desired_weapon_id:
                continue
            if current > 0:
                base.weapons[current] = base.weapons.get(current, 0) + 1
            base.weapons[desired_weapon_id] -= 1
            ac.weapons[hardpoint] = float(desired_weapon_id)

    def _record_unfinished_missions(self, outcome: str) -> None:
        completed_offer_ids = {
            event.get("mission_offer_id")
            for event in self.episode_events
            if event.get("event_type") == "mission_execution"
        }
        for offer in self._mission_offer_log:
            if offer.get("mission_offer_id") in completed_offer_ids:
                continue
            self._record_event("mission_unresolved", outcome=outcome, **offer)
