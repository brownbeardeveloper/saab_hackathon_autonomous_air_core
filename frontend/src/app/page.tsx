"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import type { GameState } from "@/lib/game-types";

const SESSION_STORAGE_KEY = "fleet-command-web-session";
const AIRCRAFT_LANE_ORDER = ["ready", "transit", "mission", "repair"] as const;

type AircraftLaneKey = (typeof AIRCRAFT_LANE_ORDER)[number];

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function statusTone(status: string) {
  if (status === "AVAILABLE") return "available";
  if (status === "IN_TRANSIT") return "transit";
  if (status === "ON_MISSION") return "mission";
  return "maintenance";
}

function formatSigned(value: number | null) {
  if (value === null) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function formatHours(value: number | null | undefined) {
  if (value === null || value === undefined) return "0.0h";
  return `${value.toFixed(1)}h`;
}

function formatWholeHours(value: number | null | undefined) {
  if (value === null || value === undefined) return "0h";
  return `${Math.round(value)}h`;
}

function formatRatio(current: number, maximum: number) {
  if (maximum <= 0) return 0;
  return Math.round((current / maximum) * 100);
}

function humanizeStatus(status: string) {
  return status.toLowerCase().replaceAll("_", " ");
}

function aircraftLane(status: string): AircraftLaneKey {
  if (status === "AVAILABLE") return "ready";
  if (status === "IN_TRANSIT") return "transit";
  if (status === "ON_MISSION") return "mission";
  return "repair";
}

function plannedActionText(aircraft: GameState["aircraft"][number]) {
  if (aircraft.form.available) {
    return aircraft.advisor.summary;
  }

  if (aircraft.status === "ON_MISSION") {
    return aircraft.statusTiming.readyInHours !== null
      ? `on mission now · returns in ${formatHours(aircraft.statusTiming.readyInHours)}`
      : "on mission now";
  }

  if (aircraft.status === "IN_TRANSIT") {
    return aircraft.statusTiming.readyInHours !== null
      ? `in transit now · arrives in ${formatHours(aircraft.statusTiming.readyInHours)}`
      : "in transit now";
  }

  if (aircraft.status === "MAINTENANCE") {
    return aircraft.statusTiming.readyInHours !== null
      ? `in repair now · ready in ${formatHours(aircraft.statusTiming.readyInHours)}`
      : "in repair now";
  }

  if (aircraft.statusTiming.readyInHours !== null) {
    return `${humanizeStatus(aircraft.status)} · ready in ${formatHours(aircraft.statusTiming.readyInHours)}`;
  }

  return humanizeStatus(aircraft.status);
}

function laneMeta(key: AircraftLaneKey) {
  if (key === "ready") {
    return {
      label: "Ready",
      hint: "Open for tasking",
      headerClass: "text-status-available",
      panelClass: "border-status-available/20 bg-status-available/10",
      badgeClass: "border-status-available/30 bg-status-available/10 text-status-available",
    };
  }

  if (key === "transit") {
    return {
      label: "Transit",
      hint: "Moving between bases",
      headerClass: "text-status-transit",
      panelClass: "border-status-transit/20 bg-status-transit/10",
      badgeClass: "border-status-transit/30 bg-status-transit/10 text-status-transit",
    };
  }

  if (key === "mission") {
    return {
      label: "Mission",
      hint: "Currently deployed",
      headerClass: "text-status-mission",
      panelClass: "border-status-mission/20 bg-status-mission/10",
      badgeClass: "border-status-mission/30 bg-status-mission/10 text-status-mission",
    };
  }

  return {
    label: "Repair",
    hint: "Maintenance queue",
    headerClass: "text-status-maintenance",
    panelClass: "border-status-maintenance/20 bg-status-maintenance/10",
    badgeClass: "border-status-maintenance/30 bg-status-maintenance/10 text-status-maintenance",
  };
}

function statusDotClass(status: string) {
  const tone = statusTone(status);

  if (tone === "available") return "bg-status-available";
  if (tone === "transit") return "bg-status-transit";
  if (tone === "mission") return "bg-status-mission";
  return "bg-status-maintenance";
}

function fuelState(percent: number) {
  if (percent >= 80) {
    return {
      toneClass: "text-status-available",
      chipClass: "bg-status-available/10",
      fillClass: "bg-status-available",
    };
  }

  if (percent >= 40) {
    return {
      toneClass: "text-status-transit",
      chipClass: "bg-status-transit/10",
      fillClass: "bg-status-transit",
    };
  }

  return {
    toneClass: "text-destructive",
    chipClass: "bg-destructive/10",
    fillClass: "bg-destructive",
  };
}

function sortAircraftForLane(
  aircraft: GameState["aircraft"],
  key: AircraftLaneKey,
) {
  return [...aircraft].sort((left, right) => {
    if (key !== "ready") {
      const leftReadyIn = left.statusTiming.readyInHours ?? Number.POSITIVE_INFINITY;
      const rightReadyIn = right.statusTiming.readyInHours ?? Number.POSITIVE_INFINITY;

      if (leftReadyIn !== rightReadyIn) {
        return leftReadyIn - rightReadyIn;
      }
    }

    return left.name.localeCompare(right.name);
  });
}

function GripenSlotIcon({
  busy,
}: {
  busy: boolean;
}) {
  return (
    <span
      aria-hidden="true"
      className={cx(
        "inline-block h-4 w-6",
        busy ? "bg-destructive" : "bg-status-available",
      )}
      style={{
        WebkitMaskImage: "url('/gripen.png')",
        maskImage: "url('/gripen.png')",
        WebkitMaskPosition: "center",
        maskPosition: "center",
        WebkitMaskRepeat: "no-repeat",
        maskRepeat: "no-repeat",
        WebkitMaskSize: "contain",
        maskSize: "contain",
      }}
    />
  );
}

function readinessTimingText(item: GameState["readiness"][number]) {
  if (item.readyNow) {
    return `ready for ${formatHours(item.readyForHours)}`;
  }

  if (item.status === "ON_MISSION") {
    return `back in ${formatHours(item.readyInHours)}`;
  }

  if (item.status === "IN_TRANSIT") {
    return `arrives in ${formatHours(item.readyInHours)}`;
  }

  return `service for ${formatHours(item.readyInHours)}`;
}

function readinessServiceState(item: GameState["readiness"][number]) {
  if (item.fullServiceDue) {
    return {
      label: "service due",
      detail: `${formatWholeHours(item.hoursSinceFullService)} / ${formatWholeHours(item.fullServiceIntervalHours)} cycle`,
      toneClass: "text-destructive",
    };
  }

  if (item.hoursUntilFullService <= item.fullServiceIntervalHours * 0.2) {
    return {
      label: `${formatWholeHours(item.hoursUntilFullService)} to service`,
      detail: `${formatWholeHours(item.hoursSinceFullService)} / ${formatWholeHours(item.fullServiceIntervalHours)} cycle`,
      toneClass: "text-status-transit",
    };
  }

  return {
    label: `${formatWholeHours(item.hoursUntilFullService)} to service`,
    detail: `${formatWholeHours(item.totalFlightHours)} total`,
    toneClass: "text-muted-foreground",
  };
}

// Status badge component
function StatusBadge({ status }: { status: string }) {
  const tone = statusTone(status);
  const colors = {
    available: "bg-status-available/20 text-status-available border-status-available/30",
    mission: "bg-status-mission/20 text-status-mission border-status-mission/30",
    transit: "bg-status-transit/20 text-status-transit border-status-transit/30",
    maintenance: "bg-status-maintenance/20 text-status-maintenance border-status-maintenance/30",
  };

  return (
    <span className={cx(
      "px-1.5 py-0.5 text-[10px] font-mono uppercase tracking-wider border rounded",
      colors[tone as keyof typeof colors]
    )}>
      {humanizeStatus(status)}
    </span>
  );
}

// Stat display component
function StatCell({ label, value, subValue, accent }: {
  label: string;
  value: string | number;
  subValue?: string;
  accent?: boolean;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">{label}</span>
      <span className={cx(
        "text-lg font-mono font-semibold tabular-nums",
        accent ? "text-primary" : "text-foreground"
      )}>{value}</span>
      {subValue && <span className="text-[10px] text-muted-foreground">{subValue}</span>}
    </div>
  );
}

function CompactResourceBar({
  label,
  current,
  maximum,
}: {
  label: string;
  current: number;
  maximum: number;
}) {
  const value = formatRatio(current, maximum);
  const tone = fuelState(value);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-mono text-muted-foreground">{label}</span>
        <span className={cx("text-[10px] font-mono", tone.toneClass)}>
          {value}%
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted/50">
        <div
          className={cx("h-full rounded-full transition-all duration-500", tone.fillClass)}
          style={{ width: `${value}%` }}
        />
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] text-muted-foreground">
          {current.toFixed(0)} / {maximum.toFixed(0)}
        </span>
      </div>
    </div>
  );
}

function ResourceSection({
  title,
  meta,
  parkingUsed,
  parkingSlots,
  fuelCurrent,
  fuelMaximum,
  sparesCurrent,
  sparesMaximum,
}: {
  title: string;
  meta: string;
  parkingUsed?: number;
  parkingSlots?: number;
  fuelCurrent: number;
  fuelMaximum: number;
  sparesCurrent: number;
  sparesMaximum: number;
}) {
  const showParking = parkingUsed !== undefined && parkingSlots !== undefined;
  const iconCount = Math.max(parkingSlots ?? 0, parkingUsed ?? 0, 1);

  return (
    <section className="space-y-3 py-3">
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-xs text-foreground">{title}</span>
          <span className="text-[10px] text-muted-foreground">{meta}</span>
        </div>
        {showParking && (
          <div className="flex items-center gap-1.5">
            {Array.from({ length: iconCount }, (_, index) => {
              const isBusy = index < (parkingUsed ?? 0);

              return (
                <GripenSlotIcon
                  key={`${title}-slot-${index}`}
                  busy={isBusy}
                />
              );
            })}
          </div>
        )}
      </div>
      <CompactResourceBar
        label="Fuel"
        current={fuelCurrent}
        maximum={fuelMaximum}
      />
      <CompactResourceBar
        label="Spares"
        current={sparesCurrent}
        maximum={sparesMaximum}
      />
    </section>
  );
}

function AircraftLaneCard({ aircraft }: {
  aircraft: GameState["aircraft"][number];
}) {
  const fuelPercent = Math.round(aircraft.fuelPercent);
  const fuelTone = fuelState(fuelPercent);

  return (
    <article className="rounded-md border border-border bg-card px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className={cx("h-2 w-2 rounded-full", statusDotClass(aircraft.status))} />
            <span className="truncate font-mono text-xs font-medium text-foreground">{aircraft.name}</span>
          </div>
          <p className="mt-1 truncate text-[10px] text-muted-foreground">
            {aircraft.baseName}
          </p>
        </div>
        <div className={cx("rounded px-1.5 py-1 text-right", fuelTone.chipClass)}>
          <div className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">Fuel</div>
          <div className={cx("text-[10px] font-mono", fuelTone.toneClass)}>{fuelPercent}%</div>
        </div>
      </div>
      <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground">
        {plannedActionText(aircraft)}
      </p>
    </article>
  );
}

// Readiness row component
function ReadinessRow({ item }: {
  item: GameState["readiness"][number];
}) {
  const serviceState = readinessServiceState(item);

  return (
    <div className="grid grid-cols-[auto_minmax(0,5rem)_auto_minmax(0,1fr)_auto] items-center gap-3 px-3 py-2 rounded hover:bg-muted/30 transition-colors">
      <div className={cx(
        "w-1.5 h-1.5 rounded-full",
        item.readyNow ? "bg-status-available" : "bg-status-transit"
      )} />
      <span className="truncate font-mono text-xs text-foreground">{item.aircraftName}</span>
      <StatusBadge status={item.status} />
      <div className="min-w-0">
        <div className="truncate text-[10px] text-muted-foreground">{item.baseName}</div>
        <div className="text-[10px] font-mono text-muted-foreground">
          {readinessTimingText(item)}
        </div>
      </div>
      <div className="text-right">
        <div className={cx("text-[10px] font-mono", serviceState.toneClass)}>
          {serviceState.label}
        </div>
        <div className="text-[10px] text-muted-foreground">
          {serviceState.detail}
        </div>
      </div>
    </div>
  );
}

function MissionRequirementChip({
  label,
  tone = "default",
}: {
  label: string;
  tone?: "default" | "weapon" | "equipment";
}) {
  const toneClass = {
    default: "border-border bg-muted/30 text-foreground",
    weapon: "border-status-mission/20 bg-status-mission/10 text-status-mission",
    equipment: "border-status-available/20 bg-status-available/10 text-status-available",
  };

  return (
    <span className={cx(
      "rounded-full border px-2 py-0.5 text-[10px] font-mono",
      toneClass[tone],
    )}>
      {label}
    </span>
  );
}

function MissionQueueCard({ mission }: {
  mission: GameState["missions"][number];
}) {
  const hasWeapons = mission.weaponRequirements.length > 0;
  const hasEquipment = mission.requiredEquipment.length > 0;

  return (
    <div className="rounded-lg border border-border bg-muted/20 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="font-mono text-xs text-foreground">{mission.name}</div>
          <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
            {mission.description}
          </p>
        </div>
        <div className="rounded-full border border-border bg-card px-2 py-1 text-[10px] font-mono text-muted-foreground">
          #{mission.slot}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <div className="rounded-md bg-card px-2 py-1.5">
          <div className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">Flight</div>
          <div className="mt-0.5 text-[10px] font-mono text-foreground">
            {formatHours(mission.flightHours)}
          </div>
        </div>
        <div className="rounded-md bg-card px-2 py-1.5">
          <div className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">Fuel</div>
          <div className="mt-0.5 text-[10px] font-mono text-foreground">
            {mission.fuelCost.toFixed(0)} units
          </div>
        </div>
      </div>

      <div className="mt-3 space-y-2">
        <div>
          <div className="mb-1 text-[9px] font-mono uppercase tracking-widest text-muted-foreground">
            Weapons Required
          </div>
          <div className="flex flex-wrap gap-1">
            {hasWeapons ? mission.weaponRequirements.map((weapon) => (
              <MissionRequirementChip
                key={`${mission.slot}-weapon-${weapon.weaponId}`}
                label={`${weapon.quantity}x ${weapon.weaponName}`}
                tone="weapon"
              />
            )) : (
              <MissionRequirementChip label="No special weapons" />
            )}
          </div>
        </div>

        <div>
          <div className="mb-1 text-[9px] font-mono uppercase tracking-widest text-muted-foreground">
            Required Equipment
          </div>
          <div className="flex flex-wrap gap-1">
            {hasEquipment ? mission.requiredEquipment.map((equipment) => (
              <MissionRequirementChip
                key={`${mission.slot}-equipment-${equipment.id}`}
                label={equipment.name}
                tone="equipment"
              />
            )) : (
              <MissionRequirementChip label="No required equipment" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [game, setGame] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState("Loading");
  const [error, setError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(true);

  useEffect(() => {
    const savedSessionId = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (!savedSessionId) {
      setRestoring(false);
      return;
    }

    void (async () => {
      setLoading(true);
      setLoadingLabel("Restoring");
      setError(null);

      try {
        const response = await fetch(`/api/game/${savedSessionId}`, { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error ?? "Could not load game.");
        applyGameState(payload as GameState);
      } catch (nextError) {
        window.localStorage.removeItem(SESSION_STORAGE_KEY);
        setGame(null);
        setError(nextError instanceof Error ? nextError.message : "Could not restore session.");
      } finally {
        setLoading(false);
        setRestoring(false);
      }
    })();
  }, []);

  function applyGameState(nextGame: GameState) {
    setGame(nextGame);
    window.localStorage.setItem(SESSION_STORAGE_KEY, nextGame.sessionId);
  }

  async function openRun() {
    setLoading(true);
    setLoadingLabel("Opening");
    setError(null);

    try {
      const response = await fetch("/api/game", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Could not start game.");
      applyGameState(payload as GameState);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not start session.");
    } finally {
      setLoading(false);
      setRestoring(false);
    }
  }

  async function nextRound() {
    if (!game) return;
    setLoading(true);
    setLoadingLabel("Advancing");
    setError(null);

    try {
      const response = await fetch(`/api/game/${game.sessionId}/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Could not play turn.");
      applyGameState(payload as GameState);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not submit turn.");
    } finally {
      setLoading(false);
    }
  }

  const summary = game ? (() => {
    const totalFuel = game.bases.reduce((sum, base) => sum + base.fuel, 0);
    const totalFuelMax = game.bases.reduce((sum, base) => sum + base.fuelMax, 0);
    const totalSpares = game.bases.reduce((sum, base) => sum + base.spares, 0);
    const totalSparesMax = game.bases.reduce((sum, base) => sum + base.sparesMax, 0);

    return {
      totalFuel,
      totalFuelMax,
      totalSpares,
      totalSparesMax,
    };
  })() : null;

  const aircraftLanes = game ? AIRCRAFT_LANE_ORDER.map((key) => ({
    key,
    ...laneMeta(key),
    items: sortAircraftForLane(
      game.aircraft.filter((aircraft) => aircraftLane(aircraft.status) === key),
      key,
    ),
  })) : [];

  return (
    <div className="min-h-screen bg-background text-foreground p-3">
      {/* Scanline effect overlay */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.02] z-50">
      <div className="w-full h-full" style={{
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(106, 229, 255, 0.05) 2px, rgba(106, 229, 255, 0.05) 4px)"
        }} />
      </div>

      <main className="mx-auto max-w-6xl space-y-3">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 bg-card border border-border rounded-lg">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center overflow-hidden">
                <Image
                  alt="Saab"
                  className="h-5 w-5 object-contain"
                  height={20}
                  src="/saab.png"
                  width={20}
                />
              </div>
              <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Autonomous Air Core</span>
            </div>
          </div>

          <button
            className={cx(
              "px-4 py-2 text-xs font-mono uppercase tracking-wider rounded",
              "border border-[#b366ff]/70 bg-[#8f3dff]/10 text-white",
              "shadow-[0_0_24px_rgba(143,61,255,0.28),inset_0_0_18px_rgba(179,102,255,0.08)]",
              "backdrop-blur-sm transition-all",
              "hover:bg-[#a855f7]/16 hover:text-white hover:shadow-[0_0_32px_rgba(168,85,247,0.38),inset_0_0_18px_rgba(179,102,255,0.12)]",
              "disabled:opacity-40 disabled:cursor-not-allowed"
            )}
            disabled={loading || Boolean(game?.meta.finished)}
            onClick={game ? nextRound : openRun}
            type="button"
          >
            {game ? (game.meta.finished ? "Complete" : "Next Round") : "Open Run"}
          </button>
        </header>

        {/* Loading / Error states */}
        {(loading || restoring) && (
          <div className="px-4 py-2 bg-card border border-border rounded-lg">
            <span className="text-xs font-mono text-muted-foreground animate-pulse">{loadingLabel}...</span>
          </div>
        )}

        {error && (
          <div className="px-4 py-2 bg-destructive/10 border border-destructive/30 rounded-lg">
            <span className="text-xs text-destructive">{error}</span>
          </div>
        )}

        {/* Empty state */}
        {!game && !loading && !restoring && (
          <div className="flex items-center justify-center h-64 bg-card border border-border rounded-lg">
            <div className="text-center">
              <div className="w-8 h-8 mx-auto mb-3 rounded-full border-2 border-dashed border-muted-foreground/30" />
              <p className="text-sm text-muted-foreground">Ready to open a run</p>
            </div>
          </div>
        )}

        {/* Result banner */}
        {game?.meta.resultLabel && (
          <div className="px-4 py-2 bg-primary/10 border border-primary/30 rounded-lg flex items-center justify-between">
            <span className="text-xs font-medium text-primary">{game.meta.resultLabel}</span>
            <span className="text-xs font-mono text-primary">Score: {game.meta.score.toFixed(1)}</span>
          </div>
        )}

        {game && (
          <>
            {/* Stats row */}
            <div className="grid grid-cols-5 gap-3">
              <div className="bg-card border border-border rounded-lg p-3">
                <StatCell label="Turn" value={game.meta.turn} />
              </div>
              <div className="bg-card border border-border rounded-lg p-3">
                <StatCell label="Time" value={`${game.meta.timeHours.toFixed(0)}h`} />
              </div>
              <div className="bg-card border border-border rounded-lg p-3">
                <StatCell label="Score" value={game.meta.score.toFixed(1)} accent />
              </div>
              <div className="bg-card border border-border rounded-lg p-3">
                <StatCell
                  label="Missions"
                  value={`${game.meta.missionsCompleted}/${game.meta.totalMissions}`}
                />
              </div>
              <div className="bg-card border border-border rounded-lg p-3">
                <StatCell
                  label="Last Turn"
                  value={formatSigned(game.meta.lastTurnReward)}
                  accent={(game.meta.lastTurnReward ?? 0) >= 0}
                />
              </div>
            </div>

            {/* Main grid */}
            <div className="grid grid-cols-3 gap-3">
              {/* Aircraft list */}
              <div className="col-span-2 bg-card border border-border rounded-lg">
                <div className="px-4 py-2 border-b border-border flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">Aircraft</span>
                    <span className="text-xs font-mono text-foreground">Status Lanes</span>
                  </div>
                  <span className="text-[10px] text-muted-foreground">{game.aircraft.length} units</span>
                </div>
                <div className="overflow-x-auto p-2">
                  <div className="grid min-w-[760px] grid-cols-4 gap-2">
                    {aircraftLanes.map((lane) => (
                      <section
                        key={lane.key}
                        className={cx("rounded-lg border p-2", lane.panelClass)}
                      >
                        <div className="mb-2 flex items-start justify-between gap-2 border-b border-border/60 pb-2">
                          <div>
                            <div className={cx("text-[10px] font-mono uppercase tracking-widest", lane.headerClass)}>
                              {lane.label}
                            </div>
                            <div className="text-[10px] text-muted-foreground">{lane.hint}</div>
                          </div>
                          <div className={cx(
                            "rounded-full border px-2 py-1 text-[10px] font-mono",
                            lane.badgeClass,
                          )}>
                            {lane.items.length}
                          </div>
                        </div>

                        {lane.items.length > 0 ? (
                          <div className="space-y-2">
                            {lane.items.map((aircraft) => (
                              <AircraftLaneCard key={aircraft.id} aircraft={aircraft} />
                            ))}
                          </div>
                        ) : (
                          <div className="rounded-md border border-dashed border-border/70 px-3 py-6 text-center text-[10px] text-muted-foreground">
                            No aircraft here
                          </div>
                        )}
                      </section>
                    ))}
                  </div>
                </div>

                <div className="mt-12 border-t border-border">
                  <div className="px-4 py-2 border-b border-border flex items-center justify-between">
                    <span className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">Readiness</span>
                    <span className="text-[10px] font-mono text-muted-foreground truncate max-w-48">{game.sessionId}</span>
                  </div>
                  <div className="p-2 space-y-0.5">
                    {game.readiness.map((item) => (
                      <ReadinessRow key={item.aircraftId} item={item} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Summary panels */}
              <div className="space-y-3">
                {/* Resources */}
                <div className="bg-card border border-border rounded-lg p-3">
                  <span className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">Resources</span>
                  <div className="mt-2 divide-y divide-border/70">
                    <ResourceSection
                      title="All Bases"
                      meta="Combined stock"
                      fuelCurrent={summary?.totalFuel ?? 0}
                      fuelMaximum={summary?.totalFuelMax ?? 0}
                      sparesCurrent={summary?.totalSpares ?? 0}
                      sparesMaximum={summary?.totalSparesMax ?? 0}
                    />

                    {game.bases.map((base) => (
                      <ResourceSection
                        key={`resources-${base.id}`}
                        title={base.name}
                        meta={`${base.dockedAircraft.length} docked`}
                        parkingUsed={base.parkingUsed}
                        parkingSlots={base.parkingSlots}
                        fuelCurrent={base.fuel}
                        fuelMaximum={base.fuelMax}
                        sparesCurrent={base.spares}
                        sparesMaximum={base.sparesMax}
                      />
                    ))}
                  </div>
                </div>

                {/* Missions */}
                <div className="bg-card border border-border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">Missions</span>
                    <span className="text-sm font-mono font-semibold text-foreground">{game.missions.length}</span>
                  </div>
                  <div className="space-y-2">
                    {game.missions.length > 0
                      ? game.missions.map((mission) => (
                        <MissionQueueCard key={mission.slot} mission={mission} />
                      ))
                      : <span className="text-[10px] text-muted-foreground">No open missions</span>
                    }
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Footer */}
        <footer className="flex items-center justify-center py-2">
          <span className="text-[9px] font-mono text-muted-foreground/50 uppercase tracking-widest">
            SAAB Fleet Management System
          </span>
        </footer>
      </main>
    </div>
  );
}
