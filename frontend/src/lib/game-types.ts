export type ChoiceOption = {
  raw: number;
  label: string;
  baseId?: number;
  isCurrent?: boolean;
  groupLabel?: string;
  weaponId?: number;
  equipmentId?: number;
  missionSlot?: number;
};

export type ModeOption = {
  value: number;
  label: string;
  description: string;
  disabled: boolean;
};

export type AircraftSelection = {
  mode: number;
  quickMissionRaw?: number | null;
  transferBaseRaw?: number | null;
  detailedBaseRaw: number;
  detailedWeaponRaws: number[];
  detailedEquipmentRaw: number;
  detailedMissionRaw: number;
};

export type AircraftForm = {
  available: boolean;
  lockedReason: string | null;
  modeOptions: ModeOption[];
  defaults: {
    mode: number;
    quickMissionRaw: number | null;
    transferBaseRaw: number | null;
    detailedBaseRaw: number;
    detailedWeaponRaws: number[];
    detailedEquipmentRaw: number;
    detailedMissionRaw: number;
  };
  quickMissionOptions: ChoiceOption[];
  transferOptions: ChoiceOption[];
  detailed: {
    baseOptions: ChoiceOption[];
    weaponOptions: ChoiceOption[][];
    equipmentOptions: ChoiceOption[];
    missionOptions: ChoiceOption[];
  };
};

export type GameState = {
  sessionId: string;
  meta: {
    turn: number;
    timeHours: number;
    score: number;
    missionsCompleted: number;
    totalMissions: number;
    advisorStatus: string;
    aiError: string | null;
    seed: number;
    configPath: string;
    missionsFile: string | null;
    modelPath: string | null;
    finished: boolean;
    terminated: boolean;
    truncated: boolean;
    resultLabel: string | null;
    lastTurnReward: number | null;
  };
  bases: Array<{
    id: number;
    name: string;
    fuel: number;
    fuelMax: number;
    spares: number;
    sparesMax: number;
    parkingUsed: number;
    parkingSlots: number;
    dockedAircraft: string[];
    weapons: Array<{
      id: number;
      name: string;
      count: number;
    }>;
  }>;
  missions: Array<{
    slot: number;
    name: string;
    description: string;
    flightHours: number;
    fuelCost: number;
    weaponRequirements: Array<{
      weaponId: number;
      weaponName: string;
      quantity: number;
    }>;
    requiredEquipment: Array<{
      id: number;
      name: string;
    }>;
  }>;
  readiness: Array<{
    aircraftId: number;
    aircraftName: string;
    baseName: string;
    status: string;
    totalFlightHours: number;
    hoursSinceFullService: number;
    hoursUntilFullService: number;
    fullServiceIntervalHours: number;
    fullServiceDue: boolean;
    statusSinceHours: number;
    readyNow: boolean;
    readyForHours: number | null;
    readyInHours: number | null;
    nextReadyAtHours: number | null;
    timeline: Array<{
      status: string;
      startHours: number;
      endHours: number | null;
      durationHours: number;
      isCurrent: boolean;
    }>;
  }>;
  aircraft: Array<{
    id: number;
    name: string;
    baseId: number;
    baseName: string;
    status: string;
    fuelLevel: number;
    fuelMax: number;
    fuelPercent: number;
    totalFlightHours: number;
    flightHoursSinceLastMission: number;
    weapons: Array<{
      slot: number;
      weaponId: number;
      label: string;
    }>;
    equipment: Array<{
      slot: number;
      equipmentId: number;
      name: string;
      active: boolean;
    }>;
    advisor: {
      summary: string;
      suggestedMode: number | null;
      suggestedAction: number[] | null;
    };
    statusTiming: {
      statusSinceHours: number;
      readyNow: boolean;
      readyForHours: number | null;
      readyInHours: number | null;
      nextReadyAtHours: number | null;
      timeline: Array<{
        status: string;
        startHours: number;
        endHours: number | null;
        durationHours: number;
        isCurrent: boolean;
      }>;
    };
    form: AircraftForm;
  }>;
};
