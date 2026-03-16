"""Action masking for the fleet-management RL environment.

Computes per-aircraft boolean validity masks for every sub-action in the
MultiDiscrete space defined in ``FleetEnv``::

    [target_base, weapon_group_1 … weapon_group_3, equipment, mission]

Invalid logits are set to ``-inf`` so the policy can never sample them.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from aircraft import (
    Aircraft,
    AircraftBounds,
    AircraftStatus,
    HARDPOINT_GROUPS,
    NUM_WEAPON_GROUPS,
)
from airbase import Airbase, TransferConfig
from missions import Mission, MissionBounds

_NEG_INF = float("-inf")
WEAPON_ACTION_KEEP = 0
WEAPON_ACTION_UNLOAD = 1
WEAPON_ACTION_WEAPON_OFFSET = 2


# ---------------------------------------------------------------------------
# ActionMasker – holds static config, produces masks every step
# ---------------------------------------------------------------------------

class ActionMasker:
    """Builds boolean validity masks for the fleet MultiDiscrete action space.

    Instantiate once with the environment's static config, then call
    :meth:`mask_for_aircraft` (single aircraft) or :meth:`mask_for_fleet`
    (whole fleet) on every environment step.

    Parameters
    ----------
    base_ids : ordered collection of airbase IDs present in the scenario.
    weapon_type_ids : ordered collection of weapon-type IDs (1-indexed from config).
    equipment_type_ids : ordered collection of equipment-type IDs.
    num_mission_slots : size of the mission sub-action (including the idle slot at 0).
    bounds : aircraft capability bounds parsed from config.
    transfer : transfer duration / fuel cost config.
    mission_bounds : upper limits on mission parameters.
    """

    def __init__(
        self,
        base_ids: Sequence[int],
        weapon_type_ids: Sequence[int],
        equipment_type_ids: Sequence[int],
        num_mission_slots: int,
        bounds: AircraftBounds,
        transfer: TransferConfig,
        mission_bounds: MissionBounds,
    ) -> None:
        self.base_ids: List[int] = sorted(base_ids)
        self.weapon_ids: List[int] = sorted(weapon_type_ids)
        self.equipment_ids: List[int] = sorted(equipment_type_ids)
        self.num_mission_slots = num_mission_slots
        self.bounds = bounds
        self.transfer = transfer
        self.mission_bounds = mission_bounds

        self.n_bases = len(self.base_ids)
        self.n_wpn = len(self.weapon_ids) + WEAPON_ACTION_WEAPON_OFFSET
        self.n_eqp = len(self.equipment_ids) + 1     # idx 0 = none
        self.n_msn = num_mission_slots

        self.flat_width = (
            self.n_bases
            + NUM_WEAPON_GROUPS * self.n_wpn
            + self.n_eqp
            + self.n_msn
        )

        self._base_to_idx: Dict[int, int] = {
            bid: i for i, bid in enumerate(self.base_ids)
        }
        self._wpn_to_idx: Dict[int, int] = {
            wid: i + WEAPON_ACTION_WEAPON_OFFSET for i, wid in enumerate(self.weapon_ids)
        }

    # ------------------------------------------------------------------
    # Per-dimension mask builders
    # ------------------------------------------------------------------

    def _base_mask(
        self,
        ac: Aircraft,
        bases: Dict[int, Airbase],
    ) -> np.ndarray:
        mask = np.zeros(self.n_bases, dtype=bool)

        cur_idx = self._base_to_idx.get(ac.base_id)
        if cur_idx is not None:
            mask[cur_idx] = True                     # "stay" is always legal

        if ac.status != AircraftStatus.AVAILABLE:
            return mask

        for idx, bid in enumerate(self.base_ids):
            if bid == ac.base_id:
                continue
            dest = bases.get(bid)
            if dest is None:
                continue
            if len(dest.aircraft_docked) >= dest.parking_slots:
                continue
            if ac.fuel_level < self.transfer.fuel_cost:
                continue
            mask[idx] = True

        return mask

    def _weapon_group_mask(
        self,
        ac: Aircraft,
        base: Airbase | None,
        hardpoints: Sequence[int],
    ) -> np.ndarray:
        """Mask for one paired hardpoint action.

        The grouped weapon action supports:
        - keep current pair loadout
        - unload both hardpoints in the pair
        - set both hardpoints in the pair to the same weapon type
        """
        mask = np.zeros(self.n_wpn, dtype=bool)
        mask[WEAPON_ACTION_KEEP] = True

        if ac.status != AircraftStatus.AVAILABLE or base is None:
            return mask

        mask[WEAPON_ACTION_UNLOAD] = True

        allowed = None
        for hardpoint in hardpoints:
            hardpoint_allowed = set(self.bounds.allowed_weapons.get(hardpoint + 1, []))
            allowed = hardpoint_allowed if allowed is None else (allowed & hardpoint_allowed)
        allowed = allowed or set()

        for wid in self.weapon_ids:
            action_idx = self._wpn_to_idx[wid]
            if wid not in allowed:
                continue
            required_stock = sum(
                1
                for hardpoint in hardpoints
                if int(ac.weapons[hardpoint]) != wid
            )
            if required_stock == 0:
                mask[action_idx] = True
                continue
            if base.weapons.get(wid, 0) >= required_stock:
                mask[action_idx] = True

        return mask

    def _equipment_mask(self, ac: Aircraft) -> np.ndarray:
        mask = np.zeros(self.n_eqp, dtype=bool)
        mask[0] = True                               # "none" always legal

        if ac.status != AircraftStatus.AVAILABLE:
            return mask

        mask[1:] = True
        return mask

    def _mission_mask(
        self,
        ac: Aircraft,
        missions: Dict[int, Mission],
    ) -> np.ndarray:
        mask = np.zeros(self.n_msn, dtype=bool)
        mask[0] = True                               # idle

        if ac.status != AircraftStatus.AVAILABLE:
            return mask

        for mid, mission in missions.items():
            if mid <= 0 or mid >= self.n_msn:
                continue
            if mission.completed:
                continue
            if len(mission.assigned_aircraft) >= self.mission_bounds.max_aircraft_per_mission:
                continue
            if ac.fuel_level < mission.fuel_cost:
                continue
            mask[mid] = True

        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask_for_aircraft(
        self,
        ac: Aircraft,
        bases: Dict[int, Airbase],
        missions: Dict[int, Mission],
    ) -> List[np.ndarray]:
        """Compute one boolean mask per sub-action dimension for *ac*.

        Returns a list whose length equals the number of sub-actions in the
        ``MultiDiscrete`` space (1 + NUM_WEAPON_GROUPS + 1 + 1 = 6).
        """
        base = bases.get(ac.base_id)

        masks: List[np.ndarray] = [self._base_mask(ac, bases)]
        for hardpoints in HARDPOINT_GROUPS:
            masks.append(self._weapon_group_mask(ac, base, hardpoints))
        masks.append(self._equipment_mask(ac))
        masks.append(self._mission_mask(ac, missions))
        return masks

    def flat_mask_for_aircraft(
        self,
        ac: Aircraft,
        bases: Dict[int, Airbase],
        missions: Dict[int, Mission],
    ) -> np.ndarray:
        """Same as :meth:`mask_for_aircraft` but concatenated into one flat vector."""
        return np.concatenate(self.mask_for_aircraft(ac, bases, missions))

    def mask_for_fleet(
        self,
        fleet: Dict[int, Aircraft],
        bases: Dict[int, Airbase],
        missions: Dict[int, Mission],
    ) -> Dict[int, List[np.ndarray]]:
        """Compute masks for every aircraft in *fleet*."""
        return {
            ac_id: self.mask_for_aircraft(ac, bases, missions)
            for ac_id, ac in fleet.items()
        }

    def flat_mask_matrix(
        self,
        fleet: Dict[int, Aircraft],
        bases: Dict[int, Airbase],
        missions: Dict[int, Mission],
    ) -> np.ndarray:
        """Return a ``(n_aircraft, flat_width)`` boolean matrix over the whole fleet."""
        rows = [
            self.flat_mask_for_aircraft(ac, bases, missions)
            for ac in fleet.values()
        ]
        return np.stack(rows)


# ---------------------------------------------------------------------------
# Logit-masking utilities
# ---------------------------------------------------------------------------

def apply_mask_to_logits(
    logits: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Set logits to ``-inf`` wherever *mask* is ``False``.

    Works on any shape as long as ``logits`` and ``mask`` broadcast together.
    """
    masked = logits.copy()
    masked[~mask] = _NEG_INF
    return masked


def apply_mask_to_logits_torch(logits, mask):
    """Torch variant – returns a new tensor with invalid positions set to ``-inf``.

    Parameters
    ----------
    logits : torch.Tensor
    mask : torch.BoolTensor  (same shape or broadcastable)
    """
    import torch  # noqa: delayed import keeps numpy-only envs lightweight
    return logits.masked_fill(~mask, _NEG_INF)


def split_and_mask_logits(
    flat_logits: np.ndarray,
    masks: List[np.ndarray],
) -> List[np.ndarray]:
    """Split a flat logit vector by sub-action sizes, mask each, and return the list.

    Useful when the policy head outputs a single flat vector that must be
    chunked into per-sub-action logits before applying softmax.
    """
    sections: List[np.ndarray] = []
    offset = 0
    for m in masks:
        size = m.shape[0]
        chunk = flat_logits[offset : offset + size].copy()
        chunk[~m] = _NEG_INF
        sections.append(chunk)
        offset += size
    return sections


def split_and_mask_logits_torch(flat_logits, masks):
    """Torch variant of :func:`split_and_mask_logits`."""
    import torch  # noqa: delayed import
    sections = []
    offset = 0
    for m in masks:
        size = m.shape[0]
        chunk = flat_logits[offset : offset + size]
        sections.append(chunk.masked_fill(~m, _NEG_INF))
        offset += size
    return sections
