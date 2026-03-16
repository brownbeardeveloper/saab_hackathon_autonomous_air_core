#!/usr/bin/env python3
"""Thin JSON bridge between the Next.js frontend and the Python fleet game."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action_masking import (
    WEAPON_ACTION_KEEP,
    WEAPON_ACTION_UNLOAD,
    WEAPON_ACTION_WEAPON_OFFSET,
)
from aircraft import (
    AircraftStatus,
    HARDPOINT_GROUPS,
    HARDPOINT_GROUP_LABELS,
    NUM_EQUIPMENT_SLOTS,
    NUM_HARDPOINTS,
    NUM_WEAPON_GROUPS,
)
from play import FleetGame

DEFAULT_CONFIG = os.getenv("FLEET_CONFIG_PATH", "config.yml")
DEFAULT_MISSIONS_FILE = os.getenv("FLEET_MISSIONS_FILE", "generated_missions_100.json")
DEFAULT_MODEL = os.getenv(
    "FLEET_MODEL_PATH",
    "models/best_model.zip",
)
SESSIONS_DIR = Path(
    os.getenv("FLEET_SESSIONS_DIR", str(PROJECT_ROOT / ".fleet_web_sessions"))
)
SESSION_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def _read_payload() -> dict:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    return json.loads(raw)


def _session_path(session_id: str) -> Path:
    if not SESSION_ID_RE.match(session_id):
        raise ValueError("Invalid session id.")
    return SESSIONS_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_session(session: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session["session_id"])
    with path.open("w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=2)


def _choice(raw: int, label: str, **extra: Any) -> dict:
    data = {"raw": int(raw), "label": label}
    for key, value in extra.items():
        if value is not None:
            data[key] = value
    return data


def _pick_raw(
    options: List[dict],
    suggested_raw: Optional[int],
    fallback_index: int = 0,
) -> Optional[int]:
    if not options:
        return None
    if suggested_raw is not None:
        for option in options:
            if int(option["raw"]) == int(suggested_raw):
                return int(option["raw"])
    fallback_index = max(0, min(fallback_index, len(options) - 1))
    return int(options[fallback_index]["raw"])


def _mission_requirements(game: FleetGame, mission) -> List[dict]:
    return [
        {
            "weaponId": int(wid),
            "weaponName": game.weapon_names.get(wid, f"W{wid}"),
            "quantity": int(qty),
        }
        for wid, qty in sorted(mission.weapon_requirements.items())
    ]


def _required_equipment(game: FleetGame, mission) -> List[dict]:
    return [
        {
            "id": int(eid),
            "name": game.equipment_names.get(eid, f"E{eid}"),
        }
        for eid in mission.recommended_equipment
    ]


def _mission_label(game: FleetGame, mission) -> str:
    reqs = _mission_requirements(game, mission)
    reqs_text = ", ".join(
        f"{item['weaponName']} x{item['quantity']}" for item in reqs
    ) or "none"
    eqp_text = ", ".join(
        item["name"] for item in _required_equipment(game, mission)
    ) or "none"
    return (
        f"{mission.name}: {mission.flight_hours:.1f}h, fuel {mission.fuel_cost:.0f}, "
        f"weapons [{reqs_text}], required eqp [{eqp_text}]"
    )


def _weapon_group_loadout_label(game: FleetGame, ac, group_idx: int) -> str:
    hardpoints = HARDPOINT_GROUPS[group_idx]
    labels = []
    for hardpoint in hardpoints:
        wid = int(ac.weapons[hardpoint])
        labels.append(game.weapon_names.get(wid, "Empty") if wid != 0 else "Empty")
    if len(set(labels)) == 1:
        return labels[0]
    return " / ".join(
        f"H{hardpoint + 1} {label}"
        for hardpoint, label in zip(hardpoints, labels)
    )


def _initialize_status_tracker(game: FleetGame) -> dict:
    current_time = float(game.env.current_time)
    tracker = {}
    for ac_id, ac in game.env.fleet.items():
        tracker[ac_id] = {
            "current_status": ac.status.name,
            "last_change_time": current_time,
            "segments": [
                {
                    "status": ac.status.name,
                    "start_hours": current_time,
                    "end_hours": None,
                }
            ],
        }
    return tracker


def _capture_status_changes(game: FleetGame, tracker: dict) -> None:
    current_time = float(game.env.current_time)
    for ac_id, ac in game.env.fleet.items():
        entry = tracker[ac_id]
        current_status = ac.status.name
        if entry["current_status"] == current_status:
            continue

        entry["segments"][-1]["end_hours"] = current_time
        entry["segments"].append(
            {
                "status": current_status,
                "start_hours": current_time,
                "end_hours": None,
            }
        )
        entry["current_status"] = current_status
        entry["last_change_time"] = current_time


def _build_aircraft_form(game: FleetGame, ac_id: int) -> dict:
    env = game.env
    ac = env.fleet[ac_id]
    if ac.status != AircraftStatus.AVAILABLE:
        return {
            "available": False,
            "lockedReason": ac.status.name,
            "modeOptions": [],
            "defaults": {
                "mode": 0,
                "quickMissionRaw": None,
                "transferBaseRaw": None,
                "detailedBaseRaw": 0,
                "detailedWeaponRaws": [0] * NUM_WEAPON_GROUPS,
                "detailedEquipmentRaw": 0,
                "detailedMissionRaw": 0,
            },
            "quickMissionOptions": [],
            "transferOptions": [],
            "detailed": {
                "baseOptions": [],
                "weaponOptions": [],
                "equipmentOptions": [],
                "missionOptions": [],
            },
        }

    masker = env.masker
    suggested_action = game.ai_suggestions.get(ac_id)
    masks = masker.mask_for_aircraft(ac, env.bases, env.missions)
    base_mask = masks[0]
    weapon_masks = masks[1:1 + NUM_WEAPON_GROUPS]
    equipment_mask = masks[1 + NUM_WEAPON_GROUPS]
    mission_mask = masks[2 + NUM_WEAPON_GROUPS]

    base_options: List[dict] = []
    transfer_options: List[dict] = []
    stay_index = 0
    for idx, base_id in enumerate(masker.base_ids):
        if not base_mask[idx]:
            continue
        is_current = int(base_id) == int(ac.base_id)
        label = game.base_names.get(base_id, f"Base {base_id}")
        if is_current:
            label += " (stay)"
            stay_index = len(base_options)
        option = _choice(idx, label, baseId=int(base_id), isCurrent=is_current)
        base_options.append(option)
        if not is_current:
            transfer_options.append(
                _choice(
                    idx,
                    game.base_names.get(base_id, f"Base {base_id}"),
                    baseId=int(base_id),
                )
            )

    weapon_options: List[List[dict]] = []
    weapon_default_raws: List[int] = []
    for group_idx, group_label in enumerate(HARDPOINT_GROUP_LABELS):
        valid: List[dict] = []
        default_index = 0
        current_label = _weapon_group_loadout_label(game, ac, group_idx)
        for weapon_idx in range(len(weapon_masks[group_idx])):
            if not weapon_masks[group_idx][weapon_idx]:
                continue
            if weapon_idx == WEAPON_ACTION_KEEP:
                label = f"No change ({current_label})"
                weapon_id = 0
            elif weapon_idx == WEAPON_ACTION_UNLOAD:
                label = "Unload both"
                weapon_id = 0
            else:
                weapon_id = masker.weapon_ids[weapon_idx - WEAPON_ACTION_WEAPON_OFFSET]
                label = f"Set both to {game.weapon_names.get(weapon_id, f'Weapon {weapon_id}')}"
            valid.append(
                _choice(
                    weapon_idx,
                    label,
                    weaponId=int(weapon_id),
                    groupLabel=group_label,
                )
            )
        weapon_options.append(valid)
        weapon_default_raws.append(
            int(
                _pick_raw(
                    valid,
                    suggested_action[1 + group_idx] if suggested_action else None,
                    fallback_index=default_index,
                )
                or 0
            )
        )

    equipment_options: List[dict] = []
    for equipment_idx in range(len(equipment_mask)):
        if not equipment_mask[equipment_idx]:
            continue
        if equipment_idx == 0:
            label = "No change"
            equipment_id = 0
        else:
            equipment_id = masker.equipment_ids[equipment_idx - 1]
            slot_idx = equipment_id - 1
            is_active = (
                0 <= slot_idx < NUM_EQUIPMENT_SLOTS
                and ac.equipment[slot_idx] > 0.5
            )
            name = game.equipment_names.get(equipment_id, f"Eqp {equipment_id}")
            label = f"Toggle {name} ({'ON->OFF' if is_active else 'OFF->ON'})"
        equipment_options.append(
            _choice(equipment_idx, label, equipmentId=int(equipment_id))
        )

    mission_options: List[dict] = [_choice(0, "Idle (no mission)", missionSlot=0)]
    quick_mission_options: List[dict] = []
    for mission_idx in range(len(mission_mask)):
        if not mission_mask[mission_idx] or mission_idx == 0:
            continue
        mission = env.missions.get(mission_idx)
        if mission is None or mission.completed:
            continue
        label = _mission_label(game, mission)
        option = _choice(mission_idx, label, missionSlot=int(mission_idx))
        mission_options.append(option)
        quick_mission_options.append(option)

    quick_enabled = len(quick_mission_options) > 0
    transfer_enabled = len(transfer_options) > 0

    suggested_mode = game._infer_mode_from_action(ac_id, suggested_action)
    mode_options = [
        {
            "value": 0,
            "label": "Skip",
            "description": "Idle and keep everything as-is",
            "disabled": False,
        },
        {
            "value": 1,
            "label": "Quick Mission",
            "description": "Pick a mission and keep the current loadout",
            "disabled": not quick_enabled,
        },
        {
            "value": 2,
            "label": "Transfer",
            "description": "Move to another base",
            "disabled": not transfer_enabled,
        },
        {
            "value": 3,
            "label": "Detailed",
            "description": "Set base, weapons, equipment, and mission",
            "disabled": False,
        },
    ]

    default_mode = 0
    if suggested_mode is not None:
        matching_mode = next(
            (option for option in mode_options if option["value"] == suggested_mode),
            None,
        )
        if matching_mode is not None and not matching_mode["disabled"]:
            default_mode = suggested_mode

    return {
        "available": True,
        "lockedReason": None,
        "modeOptions": mode_options,
        "defaults": {
            "mode": int(default_mode),
            "quickMissionRaw": _pick_raw(
                quick_mission_options,
                suggested_action[-1] if suggested_action else None,
                fallback_index=0,
            ),
            "transferBaseRaw": _pick_raw(
                transfer_options,
                suggested_action[0] if suggested_action else None,
                fallback_index=0,
            ),
            "detailedBaseRaw": int(
                _pick_raw(
                    base_options,
                    suggested_action[0] if suggested_action else None,
                    fallback_index=stay_index,
                )
                or 0
            ),
            "detailedWeaponRaws": weapon_default_raws,
            "detailedEquipmentRaw": int(
                _pick_raw(
                    equipment_options,
                    suggested_action[1 + NUM_WEAPON_GROUPS] if suggested_action else None,
                    fallback_index=0,
                )
                or 0
            ),
            "detailedMissionRaw": int(
                _pick_raw(
                    mission_options,
                    suggested_action[-1] if suggested_action else None,
                    fallback_index=0,
                )
                or 0
            ),
        },
        "quickMissionOptions": quick_mission_options,
        "transferOptions": transfer_options,
        "detailed": {
            "baseOptions": base_options,
            "weaponOptions": weapon_options,
            "equipmentOptions": equipment_options,
            "missionOptions": mission_options,
        },
    }


def _serialize_game(
    session: dict,
    game: FleetGame,
    last_turn: Optional[dict] = None,
    status_tracker: Optional[dict] = None,
) -> dict:
    env = game.env

    bases = []
    for base_id in sorted(env.bases):
        base = env.bases[base_id]
        bases.append(
            {
                "id": int(base_id),
                "name": base.name,
                "fuel": float(base.fuel),
                "fuelMax": float(base.fuel_max),
                "spares": int(base.spare_parts),
                "sparesMax": int(base.spare_parts_max),
                "parkingUsed": len(base.aircraft_docked),
                "parkingSlots": int(base.parking_slots),
                "weapons": [
                    {
                        "id": int(weapon_id),
                        "name": game.weapon_names.get(weapon_id, f"W{weapon_id}"),
                        "count": int(base.weapons[weapon_id]),
                    }
                    for weapon_id in sorted(base.weapons)
                ],
                "dockedAircraft": [
                    env.fleet[aircraft_id].name for aircraft_id in base.aircraft_docked
                ],
            }
        )

    missions = []
    for mission_idx in sorted(env.missions):
        mission = env.missions[mission_idx]
        if mission.completed:
            continue
        missions.append(
            {
                "slot": int(mission_idx),
                "name": mission.name,
                "description": mission.description,
                "flightHours": float(mission.flight_hours),
                "fuelCost": float(mission.fuel_cost),
                "weaponRequirements": _mission_requirements(game, mission),
                "requiredEquipment": _required_equipment(game, mission),
            }
        )

    aircraft = []
    readiness = []
    for aircraft_id in sorted(env.fleet):
        ac = env.fleet[aircraft_id]
        form = _build_aircraft_form(game, aircraft_id)
        suggested_action = game.ai_suggestions.get(aircraft_id)
        tracker_entry = (status_tracker or {}).get(aircraft_id, {})
        full_service_interval_hours = float(env.maintenance_cfg.full_service_interval_hours)
        hours_since_full_service = float(
            max(
                0.0,
                float(ac.total_flight_hours)
                - float(env._hours_at_last_service.get(aircraft_id, ac.total_flight_hours)),
            )
        )
        hours_until_full_service = float(
            max(0.0, full_service_interval_hours - hours_since_full_service)
        )
        full_service_due = hours_until_full_service <= 0.0
        status_since_hours = float(
            max(
                0.0,
                float(env.current_time) - float(tracker_entry.get("last_change_time", 0.0)),
            )
        )
        next_ready_at = env._busy_until.get(aircraft_id)
        ready_now = ac.status == AircraftStatus.AVAILABLE
        timeline = []
        for segment in tracker_entry.get("segments", [])[-4:]:
            end_hours = segment.get("end_hours")
            effective_end = float(env.current_time) if end_hours is None else float(end_hours)
            timeline.append(
                {
                    "status": segment["status"],
                    "startHours": float(segment["start_hours"]),
                    "endHours": None if end_hours is None else float(end_hours),
                    "durationHours": float(max(0.0, effective_end - float(segment["start_hours"]))),
                    "isCurrent": end_hours is None,
                }
            )
        aircraft.append(
            {
                "id": int(aircraft_id),
                "name": ac.name,
                "baseId": int(ac.base_id),
                "baseName": game.base_names.get(ac.base_id, f"Base {ac.base_id}"),
                "status": ac.status.name,
                "fuelLevel": float(ac.fuel_level),
                "fuelMax": float(env.bounds.max_fuel),
                "fuelPercent": float(
                    max(
                        0.0,
                        min(
                            100.0,
                            (float(ac.fuel_level) / float(env.bounds.max_fuel)) * 100.0,
                        ),
                    )
                ) if float(env.bounds.max_fuel) > 0 else 0.0,
                "totalFlightHours": float(ac.total_flight_hours),
                "flightHoursSinceLastMission": float(
                    ac.flight_hours_since_last_mission
                ),
                "weapons": [
                    {
                        "slot": hardpoint_idx + 1,
                        "weaponId": int(ac.weapons[hardpoint_idx]),
                        "label": (
                            game.weapon_names.get(int(ac.weapons[hardpoint_idx]), "Empty")
                            if int(ac.weapons[hardpoint_idx]) != 0
                            else "Empty"
                        ),
                    }
                    for hardpoint_idx in range(NUM_HARDPOINTS)
                ],
                "equipment": [
                    {
                        "slot": equipment_idx + 1,
                        "equipmentId": int(equipment_idx + 1),
                        "name": game.equipment_names.get(
                            equipment_idx + 1,
                            f"E{equipment_idx + 1}",
                        ),
                        "active": bool(ac.equipment[equipment_idx] > 0.5),
                    }
                    for equipment_idx in range(NUM_EQUIPMENT_SLOTS)
                ],
                "advisor": {
                    "summary": game._describe_ai_plan(aircraft_id, suggested_action),
                    "suggestedMode": game._infer_mode_from_action(
                        aircraft_id,
                        suggested_action,
                    ),
                    "suggestedAction": suggested_action,
                },
                "statusTiming": {
                    "statusSinceHours": status_since_hours,
                    "readyNow": ready_now,
                    "readyForHours": status_since_hours if ready_now else None,
                    "readyInHours": (
                        float(max(0.0, float(next_ready_at) - float(env.current_time)))
                        if next_ready_at is not None and not ready_now
                        else None
                    ),
                    "nextReadyAtHours": (
                        float(next_ready_at) if next_ready_at is not None else None
                    ),
                    "timeline": timeline,
                },
                "form": form,
            }
        )
        readiness.append(
            {
                "aircraftId": int(aircraft_id),
                "aircraftName": ac.name,
                "baseName": game.base_names.get(ac.base_id, f"Base {ac.base_id}"),
                "status": ac.status.name,
                "totalFlightHours": float(ac.total_flight_hours),
                "hoursSinceFullService": hours_since_full_service,
                "hoursUntilFullService": hours_until_full_service,
                "fullServiceIntervalHours": full_service_interval_hours,
                "fullServiceDue": bool(full_service_due),
                "statusSinceHours": status_since_hours,
                "readyNow": ready_now,
                "readyForHours": status_since_hours if ready_now else None,
                "readyInHours": (
                    float(max(0.0, float(next_ready_at) - float(env.current_time)))
                    if next_ready_at is not None and not ready_now
                    else None
                ),
                "nextReadyAtHours": (
                    float(next_ready_at) if next_ready_at is not None else None
                ),
                "timeline": timeline,
            }
        )

    readiness.sort(
        key=lambda item: (
            0 if item["readyNow"] else 1,
            -(item["readyForHours"] or 0.0) if item["readyNow"] else (item["readyInHours"] or 0.0),
            item["aircraftName"],
        )
    )

    terminated = bool(last_turn and last_turn.get("terminated"))
    truncated = bool(last_turn and last_turn.get("truncated"))
    result_label = None
    if terminated:
        result_label = "All missions complete"
    elif truncated:
        result_label = "Episode truncated"

    return {
        "sessionId": session["session_id"],
        "meta": {
            "turn": int(game.turn),
            "timeHours": float(env.current_time),
            "score": float(game.total_reward),
            "missionsCompleted": int(env.missions_completed),
            "totalMissions": int(env.total_missions),
            "advisorStatus": game.advisor_status,
            "aiError": game.ai_error,
            "seed": int(session["seed"]),
            "configPath": session["config_path"],
            "missionsFile": session["missions_file"],
            "modelPath": session["model_path"],
            "finished": terminated or truncated,
            "terminated": terminated,
            "truncated": truncated,
            "resultLabel": result_label,
            "lastTurnReward": (
                float(last_turn["reward"]) if last_turn and "reward" in last_turn else None
            ),
        },
        "bases": bases,
        "missions": missions,
        "readiness": readiness,
        "aircraft": aircraft,
    }


def _replay_session(session: dict) -> tuple[FleetGame, Optional[dict], dict]:
    game = FleetGame(
        config_path=session["config_path"],
        missions_file=session["missions_file"],
        model_path=session["model_path"],
    )
    game.obs, game.info = game.env.reset(seed=int(session["seed"]))
    game._refresh_base_names()
    game._refresh_ai_suggestions()
    status_tracker = _initialize_status_tracker(game)

    last_turn: Optional[dict] = None
    for turn_idx, flat_action in enumerate(session.get("history", []), start=1):
        obs, reward, terminated, truncated, info = game.env.step(
            np.asarray(flat_action, dtype=np.int64)
        )
        game.obs = obs
        game.info = info
        game.total_reward += float(reward)
        game.turn = turn_idx
        last_turn = {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        _capture_status_changes(game, status_tracker)
        if terminated or truncated:
            game.ai_suggestions = {}
            break
        game._refresh_ai_suggestions()

    return game, last_turn, status_tracker


def _as_int(value: Any, fallback: Optional[int]) -> Optional[int]:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _build_turn_action(game: FleetGame, aircraft_actions: dict) -> np.ndarray:
    if not isinstance(aircraft_actions, dict):
        aircraft_actions = {}

    full_action: List[int] = []

    for aircraft_id in sorted(game.env.fleet):
        ac = game.env.fleet[aircraft_id]
        if ac.status != AircraftStatus.AVAILABLE:
            full_action.extend(game._default_action(aircraft_id))
            continue

        form = _build_aircraft_form(game, aircraft_id)
        selection = aircraft_actions.get(str(aircraft_id), {})
        if not isinstance(selection, dict):
            selection = {}
        defaults = form["defaults"]

        mode = _as_int(selection.get("mode"), defaults["mode"])
        if mode is None:
            mode = 0

        if mode == 0:
            full_action.extend(game._default_action(aircraft_id))
            continue

        if mode == 1:
            valid = {option["raw"] for option in form["quickMissionOptions"]}
            mission_raw = _as_int(selection.get("quickMissionRaw"), defaults["quickMissionRaw"])
            if mission_raw not in valid:
                full_action.extend(game._default_action(aircraft_id))
                continue
            sub_action = game._default_action(aircraft_id)
            sub_action[-1] = int(mission_raw)
            full_action.extend(sub_action)
            continue

        if mode == 2:
            valid = {option["raw"] for option in form["transferOptions"]}
            base_raw = _as_int(selection.get("transferBaseRaw"), defaults["transferBaseRaw"])
            if base_raw not in valid:
                full_action.extend(game._default_action(aircraft_id))
                continue
            full_action.extend(
                [int(base_raw)] + [WEAPON_ACTION_KEEP] * NUM_WEAPON_GROUPS + [0] + [0]
            )
            continue

        base_raw = _as_int(selection.get("detailedBaseRaw"), defaults["detailedBaseRaw"])
        base_options = {
            option["raw"]: option for option in form["detailed"]["baseOptions"]
        }
        if base_raw not in base_options:
            base_raw = defaults["detailedBaseRaw"]

        chosen_base = base_options[int(base_raw)]
        if not chosen_base.get("isCurrent", False):
            full_action.extend(
                [int(base_raw)] + [WEAPON_ACTION_KEEP] * NUM_WEAPON_GROUPS + [0] + [0]
            )
            continue

        sub_action = [int(base_raw)]

        incoming_weapons = selection.get("detailedWeaponRaws")
        default_weapon_raws = defaults["detailedWeaponRaws"]
        if not isinstance(incoming_weapons, list):
            incoming_weapons = default_weapon_raws

        for group_idx in range(NUM_WEAPON_GROUPS):
            valid = {
                option["raw"] for option in form["detailed"]["weaponOptions"][group_idx]
            }
            desired_raw = None
            if group_idx < len(incoming_weapons):
                desired_raw = _as_int(incoming_weapons[group_idx], None)
            if desired_raw not in valid:
                desired_raw = int(default_weapon_raws[group_idx])
            sub_action.append(int(desired_raw))

        valid_equipment = {
            option["raw"] for option in form["detailed"]["equipmentOptions"]
        }
        equipment_raw = _as_int(
            selection.get("detailedEquipmentRaw"),
            defaults["detailedEquipmentRaw"],
        )
        if equipment_raw not in valid_equipment:
            equipment_raw = defaults["detailedEquipmentRaw"]
        sub_action.append(int(equipment_raw))

        valid_missions = {
            option["raw"] for option in form["detailed"]["missionOptions"]
        }
        mission_raw = _as_int(
            selection.get("detailedMissionRaw"),
            defaults["detailedMissionRaw"],
        )
        if mission_raw not in valid_missions:
            mission_raw = defaults["detailedMissionRaw"]
        sub_action.append(int(mission_raw))

        full_action.extend(sub_action)

    return np.asarray(full_action, dtype=np.int64)


def _new_session(payload: dict) -> dict:
    if not isinstance(payload, dict):
        payload = {}

    return {
        "session_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(payload.get("seed", 42)),
        "config_path": payload.get("configPath", DEFAULT_CONFIG),
        "missions_file": payload.get("missionsFile", DEFAULT_MISSIONS_FILE),
        "model_path": DEFAULT_MODEL,
        "history": [],
    }


def handle_start(payload: dict) -> dict:
    session = _new_session(payload)
    _save_session(session)
    game, last_turn, status_tracker = _replay_session(session)
    return _serialize_game(session, game, last_turn, status_tracker)


def handle_state(session_id: str) -> dict:
    session = _load_session(session_id)
    if session.get("model_path") != DEFAULT_MODEL:
        session["model_path"] = DEFAULT_MODEL
        _save_session(session)
    game, last_turn, status_tracker = _replay_session(session)
    return _serialize_game(session, game, last_turn, status_tracker)


def handle_step(session_id: str, payload: dict) -> dict:
    session = _load_session(session_id)
    if session.get("model_path") != DEFAULT_MODEL:
        session["model_path"] = DEFAULT_MODEL
        _save_session(session)
    game, last_turn, status_tracker = _replay_session(session)
    if last_turn and (last_turn.get("terminated") or last_turn.get("truncated")):
        return _serialize_game(session, game, last_turn, status_tracker)

    if not isinstance(payload, dict):
        payload = {}
    aircraft_actions = payload.get("aircraftActions", {})
    action = _build_turn_action(game, aircraft_actions)
    obs, reward, terminated, truncated, info = game.env.step(action)
    game.obs = obs
    game.info = info
    game.total_reward += float(reward)
    game.turn += 1
    _capture_status_changes(game, status_tracker)
    if terminated or truncated:
        game.ai_suggestions = {}
    else:
        game._refresh_ai_suggestions()

    session["history"].append(action.astype(int).tolist())
    _save_session(session)

    current_turn = {
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    return _serialize_game(session, game, current_turn, status_tracker)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "state", "step"])
    parser.add_argument("session_id", nargs="?")
    args = parser.parse_args()

    try:
        payload = _read_payload()
        if args.command == "start":
            result = handle_start(payload)
        elif args.command == "state":
            if not args.session_id:
                raise ValueError("state requires a session id")
            result = handle_state(args.session_id)
        else:
            if not args.session_id:
                raise ValueError("step requires a session id")
            result = handle_step(args.session_id, payload)

        print(json.dumps(result))
        return 0
    except Exception as exc:  # pragma: no cover - bridge error path
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
