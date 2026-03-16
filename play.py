#!/usr/bin/env python3
"""Interactive terminal game for the Fleet Management environment.

Play the same decisions the RL model makes — transfer aircraft between bases,
load weapons, toggle equipment, and assign missions. The action space is
identical to training so scores are directly comparable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from action_masking import (
    WEAPON_ACTION_KEEP,
    WEAPON_ACTION_UNLOAD,
    WEAPON_ACTION_WEAPON_OFFSET,
)
from fleet_env import FleetEnv
from aircraft import (
    AircraftStatus,
    HARDPOINT_GROUPS,
    HARDPOINT_GROUP_LABELS,
    NUM_EQUIPMENT_SLOTS,
    NUM_HARDPOINTS,
    NUM_WEAPON_GROUPS,
)

# ── ANSI helpers ─────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"

STATUS_COLORS = {
    AircraftStatus.AVAILABLE: GREEN,
    AircraftStatus.IN_TRANSIT: YELLOW,
    AircraftStatus.MAINTENANCE: RED,
    AircraftStatus.ON_MISSION: CYAN,
}

DEFAULT_MODEL_PATH = "models/best_model.zip"


def c(text, *styles):
    return "".join(styles) + str(text) + RESET


def clear():
    os.system("cls" if os.name == "nt" else "clear")


# ── Main game ────────────────────────────────────────────────────────────────

class FleetGame:
    def __init__(
        self,
        config_path: str = "config.yml",
        missions_file: Optional[str] = None,
        model_path: Optional[str] = DEFAULT_MODEL_PATH,
    ):
        self.project_root = Path(__file__).resolve().parent
        self.env = FleetEnv(config_path, mission_manifest_path=missions_file)
        self.obs: dict = {}
        self.info: dict = {}
        self.turn = 0
        self.total_reward = 0.0
        self.advisor_model = None
        self.advisor_model_path: Optional[Path] = None
        self.advisor_status = "OFF"
        self.ai_suggestions: Dict[int, List[int]] = {}
        self.ai_error: Optional[str] = None

        self.weapon_names: Dict[int, str] = {0: "Empty"}
        for wid, wt in self.env.weapon_types.items():
            self.weapon_names[wid] = wt.name

        self.equipment_names: Dict[int, str] = {}
        for eid, et in self.env.equipment_types.items():
            self.equipment_names[eid] = et.name

        self.base_names: Dict[int, str] = {
            bid: b.name for bid, b in self.env.bases.items()
        }

        self._load_advisor(model_path)

    def _refresh_base_names(self) -> None:
        self.base_names = {
            bid: base.name for bid, base in self.env.bases.items()
        }

    # ── advisor helpers ────────────────────────────────────────────────

    def _pick_default_model(self) -> Optional[Path]:
        direct_best_model = self.project_root / "models" / "best_model.zip"
        if direct_best_model.exists():
            return direct_best_model
        return None

    def _display_path(self, path: Optional[Path]) -> str:
        if path is None:
            return "none"
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[Path]:
        if model_path is None:
            return None
        if model_path == "auto":
            return self._pick_default_model()

        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = (self.project_root / resolved).resolve()
        return resolved

    def _load_advisor(self, model_path: Optional[str]) -> None:
        if model_path is None:
            self.advisor_status = "OFF (disabled)"
            return

        resolved_path = self._resolve_model_path(model_path)
        if resolved_path is None:
            self.advisor_status = "OFF (no model checkpoint found)"
            return

        if not resolved_path.exists():
            raise SystemExit(f"Model file not found: {resolved_path}")

        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            if model_path == "auto":
                self.advisor_status = "OFF (sb3-contrib not installed)"
                self.ai_error = str(exc)
                return
            raise SystemExit(
                "Could not import sb3_contrib.MaskablePPO. "
                "Install the project requirements first."
            ) from exc

        self.advisor_model = MaskablePPO.load(str(resolved_path))
        self.advisor_model_path = resolved_path
        self.advisor_status = f"ON ({self._display_path(resolved_path)})"

    def _split_fleet_action(self, action: np.ndarray) -> Dict[int, List[int]]:
        flat = np.asarray(action, dtype=np.int64).reshape(-1)
        expected_size = int(self.env.action_space.nvec.size)
        if flat.size != expected_size:
            raise ValueError(
                f"Model returned {flat.size} actions but environment expects "
                f"{expected_size}."
            )

        split: Dict[int, List[int]] = {}
        for idx, ac_id in enumerate(sorted(self.env.fleet)):
            offset = idx * self.env._n_sub
            split[ac_id] = flat[offset: offset + self.env._n_sub].astype(int).tolist()
        return split

    def _refresh_ai_suggestions(self) -> None:
        self.ai_suggestions = {}
        if self.advisor_model is None or not self.obs:
            return

        self.ai_error = None
        try:
            action, _ = self.advisor_model.predict(
                self.obs,
                deterministic=True,
                action_masks=self.env.action_masks(),
            )
            self.ai_suggestions = self._split_fleet_action(action)
        except Exception as exc:  # pragma: no cover - defensive fallback for CLI play
            self.ai_error = str(exc)

    def _option_default(
        self,
        options: Sequence[Sequence[object]],
        suggested_raw: Optional[int],
        fallback: int = 0,
    ) -> int:
        if suggested_raw is None:
            return fallback
        for idx, option in enumerate(options):
            if int(option[0]) == int(suggested_raw):
                return idx
        return fallback

    def _ai_marker(self, raw_value: int, suggested_raw: Optional[int]) -> str:
        if suggested_raw is None or int(raw_value) != int(suggested_raw):
            return ""
        return f" {c('<- AI', CYAN)}"

    def _infer_mode_from_action(
        self,
        ac_id: int,
        suggested_action: Optional[List[int]],
    ) -> Optional[int]:
        if suggested_action is None:
            return None

        default_action = self._default_action(ac_id)
        if suggested_action == default_action:
            return 0

        base_idx = suggested_action[0]
        if 0 <= base_idx < len(self.env.masker.base_ids):
            base_id = self.env.masker.base_ids[base_idx]
            if base_id != self.env.fleet[ac_id].base_id:
                return 2

        if (
            suggested_action[1:1 + NUM_WEAPON_GROUPS]
            == default_action[1:1 + NUM_WEAPON_GROUPS]
            and suggested_action[1 + NUM_WEAPON_GROUPS] == 0
            and suggested_action[-1] != 0
        ):
            return 1

        return 3

    def _mission_label(self, mission_idx: int) -> str:
        if mission_idx == 0:
            return "idle"
        mission = self.env.missions.get(mission_idx)
        if mission is None or mission.completed:
            return f"mission slot {mission_idx}"
        return mission.name

    def _describe_ai_plan(self, ac_id: int, suggested_action: Optional[List[int]]) -> str:
        if suggested_action is None:
            return "no suggestion"

        ac = self.env.fleet[ac_id]
        default_action = self._default_action(ac_id)
        base_idx = suggested_action[0]
        if 0 <= base_idx < len(self.env.masker.base_ids):
            base_id = self.env.masker.base_ids[base_idx]
        else:
            base_id = ac.base_id

        if suggested_action == default_action:
            return "stay idle and keep current loadout"

        if base_id != ac.base_id:
            base_name = self.base_names.get(base_id, f"Base {base_id}")
            return f"transfer to {base_name}"

        mission_name = self._mission_label(suggested_action[-1])
        weapon_changes = sum(
            int(suggested_action[1 + group_idx] != WEAPON_ACTION_KEEP)
            for group_idx in range(NUM_WEAPON_GROUPS)
        )
        equipment_change = suggested_action[1 + NUM_WEAPON_GROUPS] != 0

        if suggested_action[-1] != 0 and weapon_changes == 0 and not equipment_change:
            return f"keep loadout and fly {mission_name}"

        details: List[str] = []
        if weapon_changes:
            suffix = "s" if weapon_changes != 1 else ""
            details.append(f"{weapon_changes} weapon-group change{suffix}")
        if equipment_change:
            details.append("equipment toggle")
        details.append(f"mission {mission_name}")
        return "stay and " + ", ".join(details)

    # ── display ──────────────────────────────────────────────────────────

    def _header(self):
        env = self.env
        bar = c("=" * 72, BLUE)
        print(bar)
        print(c(
            f"   FLEET COMMAND   |   Turn {self.turn}   |   "
            f"Time: {env.current_time:.0f}h   |   "
            f"Missions: {env.missions_completed}/{env.total_missions}   |   "
            f"Score: {self.total_reward:.1f}",
            BOLD, WHITE,
        ))
        if self.ai_error:
            print(c(f"   AI Advisor error: {self.ai_error}", RED))
        elif self.advisor_model is not None:
            print(c(f"   AI Advisor: {self.advisor_status}", CYAN))
        print(bar)

    def _show_bases(self):
        env = self.env
        print()
        print(c("  AIRBASES", BOLD, YELLOW))
        print(c("  " + "-" * 68, DIM))
        for bid in sorted(env.bases):
            b = env.bases[bid]
            docked = [env.fleet[aid].name for aid in b.aircraft_docked]
            docked_str = ", ".join(docked) if docked else "none"
            wpn_parts = []
            for wid in sorted(b.weapons):
                wname = self.weapon_names.get(wid, f"W{wid}")
                short = wname[:10]
                wpn_parts.append(f"{short}: {b.weapons[wid]}")
            wpn_str = "  |  ".join(wpn_parts)
            print(
                f"  [{bid}] {c(b.name, BOLD):<30s}  "
                f"Fuel: {c(f'{b.fuel:.0f}/{b.fuel_max:.0f}', CYAN)}  "
                f"Spares: {b.spare_parts}/{b.spare_parts_max}  "
                f"Parking: {len(b.aircraft_docked)}/{b.parking_slots}"
            )
            print(f"      Weapons  {wpn_str}")
            print(f"      Docked   {c(docked_str, DIM)}")

    def _show_missions(self):
        env = self.env
        print()
        print(c("  ACTIVE MISSIONS", BOLD, MAGENTA))
        print(c("  " + "-" * 68, DIM))
        any_shown = False
        for mid in sorted(env.missions):
            m = env.missions[mid]
            if m.completed:
                continue
            any_shown = True
            reqs = []
            for wid, qty in sorted(m.weapon_requirements.items()):
                reqs.append(f"{self.weapon_names.get(wid, f'W{wid}')} x{qty}")
            reqs_str = ", ".join(reqs) if reqs else "none"
            eqp_str = ", ".join(
                self.equipment_names.get(eid, f"E{eid}")
                for eid in m.recommended_equipment
            ) or "none"
            print(
                f"  [Slot {mid}] {c(m.name, BOLD)}  |  {c(f'{m.flight_hours:.1f}h', CYAN)} flight  |  "
                f"Fuel cost: {c(f'{m.fuel_cost:.0f}', YELLOW)}  |  "
                f"Weapons needed: {reqs_str}"
            )
            print(f"           Recommended equipment: {eqp_str}")
        if not any_shown:
            print("  (no active missions)")

    def _show_fleet(self):
        env = self.env
        print()
        print(c("  FLEET", BOLD, GREEN))
        print(c("  " + "-" * 68, DIM))
        for aid in sorted(env.fleet):
            ac = env.fleet[aid]
            sc = STATUS_COLORS.get(ac.status, WHITE)
            status_str = c(ac.status.name, sc, BOLD)
            base_name = self.base_names.get(ac.base_id, f"Base {ac.base_id}")

            wpn_display = []
            for hp in range(NUM_HARDPOINTS):
                wid = int(ac.weapons[hp])
                if wid == 0:
                    wpn_display.append(c("-", DIM))
                else:
                    short = self.weapon_names.get(wid, f"W{wid}")
                    abbr = "".join(w[0] for w in short.split())
                    wpn_display.append(c(abbr, CYAN))

            eqp_display = []
            for slot in range(NUM_EQUIPMENT_SLOTS):
                eid = slot + 1
                if ac.equipment[slot] > 0.5:
                    short = self.equipment_names.get(eid, f"E{eid}")
                    abbr = "".join(w[0] for w in short.split())
                    eqp_display.append(c(abbr, GREEN))
                else:
                    eqp_display.append(c("-", DIM))

            print(
                f"  [{aid}] {c(ac.name, BOLD):<24s} @ {base_name:<16s}  "
                f"{status_str:<22s}  Fuel: {c(f'{ac.fuel_level:.0f}', CYAN)}"
            )
            print(
                f"       Hrs: {ac.total_flight_hours:.0f} total, "
                f"{ac.flight_hours_since_last_mission:.0f} since maint  |  "
                f"Wpn: [{' '.join(wpn_display)}]  Eqp: [{' '.join(eqp_display)}]"
            )

    def display_state(self):
        clear()
        self._header()
        self._show_bases()
        self._show_missions()
        self._show_fleet()
        print()

    # ── input helpers ────────────────────────────────────────────────────

    def _ask(self, prompt: str, n_options: int, default: int = 0) -> int:
        while True:
            try:
                raw = input(c(f"    {prompt} [0-{n_options-1}, Enter={default}]: ", YELLOW)).strip()
                if raw == "":
                    return default
                val = int(raw)
                if 0 <= val < n_options:
                    return val
                print(c(f"    Choose 0 to {n_options-1}", RED))
            except ValueError:
                print(c("    Enter a number", RED))
            except (EOFError, KeyboardInterrupt):
                print()
                sys.exit(0)

    # ── per-aircraft action builder ──────────────────────────────────────

    def _default_action(self, ac_id: int) -> List[int]:
        """Stay, keep paired weapon groups unchanged, idle."""
        ac = self.env.fleet[ac_id]
        stay_idx = self.env.masker._base_to_idx.get(ac.base_id, 0)
        return [stay_idx] + [WEAPON_ACTION_KEEP] * NUM_WEAPON_GROUPS + [0] + [0]

    def _group_loadout_label(self, ac_id: int, group_idx: int) -> str:
        ac = self.env.fleet[ac_id]
        hardpoints = HARDPOINT_GROUPS[group_idx]
        labels = []
        for hardpoint in hardpoints:
            wid = int(ac.weapons[hardpoint])
            labels.append(self.weapon_names.get(wid, "Empty") if wid != 0 else "Empty")
        if len(set(labels)) == 1:
            return labels[0]
        return " / ".join(
            f"H{hardpoint + 1} {label}"
            for hardpoint, label in zip(hardpoints, labels)
        )

    def _get_action_for_aircraft(self, ac_id: int) -> List[int]:
        ac = self.env.fleet[ac_id]
        env = self.env
        masker = env.masker
        suggested_action = self.ai_suggestions.get(ac_id)

        if ac.status != AircraftStatus.AVAILABLE:
            return self._default_action(ac_id)

        masks = masker.mask_for_aircraft(ac, env.bases, env.missions)
        base_mask = masks[0]
        wpn_masks = masks[1:1 + NUM_WEAPON_GROUPS]
        eqp_mask = masks[1 + NUM_WEAPON_GROUPS]
        msn_mask = masks[2 + NUM_WEAPON_GROUPS]

        base_name = self.base_names.get(ac.base_id, f"Base {ac.base_id}")
        print(c(f"  === {ac.name} @ {base_name}  |  Fuel: {ac.fuel_level:.0f} ===", BOLD, WHITE))
        if suggested_action is not None:
            print(c(f"  AI suggests: {self._describe_ai_plan(ac_id, suggested_action)}", CYAN))
        print()

        # quick-select menu
        suggested_mode = self._infer_mode_from_action(ac_id, suggested_action)
        print(c("  Choose action mode:", BOLD))
        print(
            f"    [0] Skip (idle — keep everything as-is)"
            f"{self._ai_marker(0, suggested_mode)}"
        )
        print(
            f"    [1] Quick Mission (pick a mission, keep current loadout)"
            f"{self._ai_marker(1, suggested_mode)}"
        )
        print(
            f"    [2] Transfer to another base"
            f"{self._ai_marker(2, suggested_mode)}"
        )
        print(
            f"    [3] Detailed (set base, weapons, equipment, mission)"
            f"{self._ai_marker(3, suggested_mode)}"
        )

        mode = self._ask("Mode", 4, default=suggested_mode or 0)

        if mode == 0:
            return self._default_action(ac_id)

        if mode == 1:
            return self._quick_mission(ac_id, msn_mask, suggested_action)

        if mode == 2:
            return self._quick_transfer(ac_id, base_mask, suggested_action)

        return self._detailed(
            ac_id,
            base_mask,
            wpn_masks,
            eqp_mask,
            msn_mask,
            suggested_action,
        )

    # ── quick mission ────────────────────────────────────────────────────

    def _quick_mission(
        self,
        ac_id: int,
        msn_mask: np.ndarray,
        suggested_action: Optional[List[int]] = None,
    ) -> List[int]:
        env = self.env

        valid: List[tuple] = []
        for midx in range(len(msn_mask)):
            if not msn_mask[midx] or midx == 0:
                continue
            m = env.missions.get(midx)
            if m is None or m.completed:
                continue
            reqs = []
            for wid, qty in sorted(m.weapon_requirements.items()):
                reqs.append(f"{self.weapon_names.get(wid, f'W{wid}')} x{qty}")
            reqs_str = ", ".join(reqs) if reqs else "none"
            eqp_str = ", ".join(
                self.equipment_names.get(eid, f"E{eid}")
                for eid in m.recommended_equipment
            ) or "none"
            label = (
                f"{m.name}: {m.flight_hours:.1f}h, fuel {m.fuel_cost:.0f}, "
                f"weapons [{reqs_str}], required eqp [{eqp_str}]"
            )
            valid.append((midx, label))

        if not valid:
            print(c("    No missions available! Defaulting to idle.", RED))
            return self._default_action(ac_id)

        print(c("  Pick a mission:", BOLD))
        for i, (_, label) in enumerate(valid):
            ai_raw = suggested_action[-1] if suggested_action is not None else None
            print(f"    [{i}] {label}{self._ai_marker(valid[i][0], ai_raw)}")

        choice = self._ask(
            "Mission",
            len(valid),
            default=self._option_default(
                valid,
                suggested_action[-1] if suggested_action is not None else None,
                fallback=0,
            ),
        )
        msn_action = valid[choice][0]

        base = self._default_action(ac_id)
        base[-1] = msn_action
        return base

    # ── quick transfer ───────────────────────────────────────────────────

    def _quick_transfer(
        self,
        ac_id: int,
        base_mask: np.ndarray,
        suggested_action: Optional[List[int]] = None,
    ) -> List[int]:
        ac = self.env.fleet[ac_id]
        masker = self.env.masker

        valid: List[tuple] = []
        for idx, bid in enumerate(masker.base_ids):
            if base_mask[idx] and bid != ac.base_id:
                label = self.base_names.get(bid, f"Base {bid}")
                valid.append((idx, label))

        if not valid:
            print(c("    No transfer destinations available! Defaulting to idle.", RED))
            return self._default_action(ac_id)

        print(c("  Transfer to:", BOLD))
        for i, (_, label) in enumerate(valid):
            ai_raw = suggested_action[0] if suggested_action is not None else None
            print(f"    [{i}] {label}{self._ai_marker(valid[i][0], ai_raw)}")

        choice = self._ask(
            "Destination",
            len(valid),
            default=self._option_default(
                valid,
                suggested_action[0] if suggested_action is not None else None,
                fallback=0,
            ),
        )
        base_idx = valid[choice][0]
        return [base_idx] + [WEAPON_ACTION_KEEP] * NUM_WEAPON_GROUPS + [0] + [0]

    # ── detailed mode ────────────────────────────────────────────────────

    def _detailed(
        self,
        ac_id: int,
        base_mask: np.ndarray,
        wpn_masks: List[np.ndarray],
        eqp_mask: np.ndarray,
        msn_mask: np.ndarray,
        suggested_action: Optional[List[int]] = None,
    ) -> List[int]:
        ac = self.env.fleet[ac_id]
        env = self.env
        masker = env.masker
        action: List[int] = []

        # -- base --
        print(c("  Base:", BOLD))
        valid_bases: List[tuple] = []
        default_base = 0
        for idx, bid in enumerate(masker.base_ids):
            if not base_mask[idx]:
                continue
            label = self.base_names.get(bid, f"Base {bid}")
            if bid == ac.base_id:
                label += " (stay)"
                default_base = len(valid_bases)
            valid_bases.append((idx, bid, label))
            ai_raw = suggested_action[0] if suggested_action is not None else None
            print(
                f"    [{len(valid_bases) - 1}] {label}"
                f"{self._ai_marker(idx, ai_raw)}"
            )

        choice = self._ask(
            "Base",
            len(valid_bases),
            default=self._option_default(
                valid_bases,
                suggested_action[0] if suggested_action is not None else None,
                fallback=default_base,
            ),
        )
        chosen_base_idx, chosen_bid, _ = valid_bases[choice]
        action.append(chosen_base_idx)

        if chosen_bid != ac.base_id:
            action += [WEAPON_ACTION_KEEP] * NUM_WEAPON_GROUPS + [0] + [0]
            print(c(f"    -> Transferring! Other actions skipped.", YELLOW))
            print()
            return action

        # -- weapons per paired hardpoint group --
        for group_idx, group_label in enumerate(HARDPOINT_GROUP_LABELS):
            wmask = wpn_masks[group_idx]
            current_name = self._group_loadout_label(ac_id, group_idx)

            valid_wpns: List[tuple] = []
            default_wpn = 0
            for widx in range(len(wmask)):
                if not wmask[widx]:
                    continue
                if widx == WEAPON_ACTION_KEEP:
                    label = "No change"
                elif widx == WEAPON_ACTION_UNLOAD:
                    label = "Unload both"
                else:
                    wid = masker.weapon_ids[widx - WEAPON_ACTION_WEAPON_OFFSET]
                    label = f"Set both to {self.weapon_names.get(wid, f'Weapon {wid}')}"
                valid_wpns.append((widx, label))

            if len(valid_wpns) == 1:
                action.append(valid_wpns[0][0])
                continue

            print(c(f"  {group_label} (current: {current_name}):", BOLD))
            for i, (_, label) in enumerate(valid_wpns):
                ai_raw = (
                    suggested_action[1 + group_idx] if suggested_action is not None else None
                )
                print(f"    [{i}] {label}{self._ai_marker(valid_wpns[i][0], ai_raw)}")
            choice = self._ask(
                group_label,
                len(valid_wpns),
                default=self._option_default(
                    valid_wpns,
                    suggested_action[1 + group_idx] if suggested_action is not None else None,
                    fallback=default_wpn,
                ),
            )
            action.append(valid_wpns[choice][0])

        # -- equipment --
        print(c("  Equipment toggle:", BOLD))
        valid_eqp: List[tuple] = []
        for eidx in range(len(eqp_mask)):
            if not eqp_mask[eidx]:
                continue
            if eidx == 0:
                label = "No change"
            else:
                eid = masker.equipment_ids[eidx - 1]
                slot = eid - 1
                on = (
                    ac.equipment[slot] > 0.5
                    if 0 <= slot < NUM_EQUIPMENT_SLOTS
                    else False
                )
                name = self.equipment_names.get(eid, f"Eqp {eid}")
                label = f"Toggle {name} ({'ON->OFF' if on else 'OFF->ON'})"
            valid_eqp.append((eidx, label))
            ai_raw = (
                suggested_action[1 + NUM_WEAPON_GROUPS]
                if suggested_action is not None
                else None
            )
            print(
                f"    [{len(valid_eqp) - 1}] {label}"
                f"{self._ai_marker(eidx, ai_raw)}"
            )

        choice = self._ask(
            "Equipment",
            len(valid_eqp),
            default=self._option_default(
                valid_eqp,
                suggested_action[1 + NUM_WEAPON_GROUPS]
                if suggested_action is not None
                else None,
                fallback=0,
            ),
        )
        action.append(valid_eqp[choice][0])

        # -- mission --
        print(c("  Mission:", BOLD))
        valid_msn: List[tuple] = []
        for midx in range(len(msn_mask)):
            if not msn_mask[midx]:
                continue
            if midx == 0:
                label = "Idle (no mission)"
            else:
                m = env.missions.get(midx)
                if m and not m.completed:
                    reqs = []
                    for wid, qty in sorted(m.weapon_requirements.items()):
                        reqs.append(
                            f"{self.weapon_names.get(wid, f'W{wid}')} x{qty}"
                        )
                    reqs_str = ", ".join(reqs) if reqs else "none"
                    eqp_str = ", ".join(
                        self.equipment_names.get(eid, f"E{eid}")
                        for eid in m.recommended_equipment
                    ) or "none"
                    label = (
                        f"{m.name}: {m.flight_hours:.1f}h, fuel {m.fuel_cost:.0f}, "
                        f"weapons [{reqs_str}], required eqp [{eqp_str}]"
                    )
                else:
                    continue
            valid_msn.append((midx, label))
            ai_raw = suggested_action[-1] if suggested_action is not None else None
            print(
                f"    [{len(valid_msn) - 1}] {label}"
                f"{self._ai_marker(midx, ai_raw)}"
            )

        choice = self._ask(
            "Mission",
            len(valid_msn),
            default=self._option_default(
                valid_msn,
                suggested_action[-1] if suggested_action is not None else None,
                fallback=0,
            ),
        )
        action.append(valid_msn[choice][0])

        print()
        return action

    # ── turn loop ────────────────────────────────────────────────────────

    def play_turn(self) -> bool:
        self._refresh_ai_suggestions()
        self.display_state()
        ac_ids = sorted(self.env.fleet.keys())
        full_action: List[int] = []

        for ac_id in ac_ids:
            ac = self.env.fleet[ac_id]
            if ac.status != AircraftStatus.AVAILABLE:
                sc = STATUS_COLORS.get(ac.status, WHITE)
                print(
                    c(f"  {ac.name}: {ac.status.name} — auto-skipped", DIM, sc)
                )
                full_action.extend(self._default_action(ac_id))
            else:
                sub = self._get_action_for_aircraft(ac_id)
                full_action.extend(sub)
            print()

        action = np.array(full_action, dtype=np.int64)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.info = info
        self.total_reward += reward
        self.turn += 1

        print(c("=" * 72, BLUE))
        sign = "+" if reward >= 0 else ""
        rc = GREEN if reward >= 0 else RED
        print(c(
            f"  Turn reward: {sign}{reward:.2f}  |  "
            f"Total: {self.total_reward:.1f}  |  "
            f"Missions done: {info['missions_completed']}/{self.env.total_missions}",
            rc,
        ))
        print(c("=" * 72, BLUE))

        if terminated:
            print()
            print(c("  ALL MISSIONS COMPLETE! VICTORY!", BOLD, GREEN))
            print(c(
                f"  Final score: {self.total_reward:.1f}  |  "
                f"Time: {self.env.current_time:.0f}h  |  "
                f"Turns: {self.turn}",
                BOLD, WHITE,
            ))
            return False

        if truncated:
            print()
            print(c("  TIME'S UP! Episode truncated.", BOLD, RED))
            print(c(
                f"  Final score: {self.total_reward:.1f}  |  "
                f"Missions: {info['missions_completed']}/{self.env.total_missions}",
                BOLD, WHITE,
            ))
            return False

        input(c("\n  Press Enter for next turn...", DIM))
        return True

    # ── entry point ──────────────────────────────────────────────────────

    def run(self):
        self.obs, self.info = self.env.reset(seed=42)
        self._refresh_base_names()
        self._refresh_ai_suggestions()

        clear()
        print(c("=" * 72, BLUE))
        print(c("   FLEET COMMAND — Interactive Fleet Management Game", BOLD, WHITE))
        print(c("=" * 72, BLUE))
        print()
        print("  You are the fleet commander. Each turn, for every available")
        print("  aircraft you decide:")
        print(f"    {c('1.', BOLD)} Where to go  (stay or transfer to another base)")
        print(f"    {c('2.', BOLD)} Weapons      (set paired groups H1/H2, H3/H4, H5/H6)")
        print(f"    {c('3.', BOLD)} Equipment    (toggle 4 equipment slots)")
        print(f"    {c('4.', BOLD)} Mission      (assign to a mission or stay idle)")
        print()
        print("  These are the EXACT same choices the RL model makes during training.")
        print()
        print(c("  AI advisor:", BOLD), self.advisor_status)
        if self.advisor_model is not None:
            print("  Press Enter to accept the AI's suggested default at each prompt,")
            print("  or type another number whenever you want to override it.")
        elif self.ai_error:
            print(c(f"  Advisor warning: {self.ai_error}", YELLOW))
        print()
        print(c("  Goal:", BOLD), f"Complete all {self.env.total_missions} missions. "
              "Faster = higher score.")
        print()
        print(c("  Controls:", BOLD))
        print("    Type a number to pick an option.")
        print("    Press Enter alone to accept the default (shown in brackets).")
        print()
        print(c("  Bases:", BOLD))
        for bid in sorted(self.base_names):
            print(f"    [{bid}] {self.base_names[bid]}")
        print()
        print(c("  Aircraft:", BOLD), f"{len(self.env.fleet)} Viggens")
        print(c("  Weapons:", BOLD))
        for wid in sorted(self.weapon_names):
            if wid == 0:
                continue
            print(f"    [{wid}] {self.weapon_names[wid]}")
        print(c("  Equipment:", BOLD))
        for eid in sorted(self.equipment_names):
            print(f"    [{eid}] {self.equipment_names[eid]}")
        print()
        input(c("  Press Enter to start...", DIM))

        while self.play_turn():
            pass

        print()
        input(c("  Press Enter to exit...", DIM))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play Fleet Command manually, with optional AI suggestions from a "
            "trained MaskablePPO checkpoint."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        help="Path to the scenario config YAML.",
    )
    parser.add_argument(
        "--missions-file",
        help="Optional fixed mission manifest JSON to replay.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to a trained model checkpoint. Defaults to models/best_model.zip."
        ),
    )
    parser.add_argument(
        "--no-advisor",
        action="store_true",
        help="Disable AI suggestions and play fully manually.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FleetGame(
        config_path=args.config,
        missions_file=args.missions_file,
        model_path=None if args.no_advisor else args.model,
    ).run()
