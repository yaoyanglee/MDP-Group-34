import React, { useEffect, useMemo, useRef, useState } from "react";
import QueryAPI from "./QueryAPI";

const BOARD_LIMIT = 20;
const ROBOT_MIN = 1;
const ROBOT_MAX = BOARD_LIMIT - 2;

const HEADINGS = Object.freeze({
  NORTH: 0,
  EAST: 2,
  SOUTH: 4,
  WEST: 6,
  SKIP: 8,
});

const headingLabels = Object.freeze({
  [HEADINGS.NORTH]: "Up",
  [HEADINGS.EAST]: "Right",
  [HEADINGS.SOUTH]: "Down",
  [HEADINGS.WEST]: "Left",
  [HEADINGS.SKIP]: "None",
});

const headingGlyph = Object.freeze({
  [HEADINGS.NORTH]: "↑",
  [HEADINGS.EAST]: "→",
  [HEADINGS.SOUTH]: "↓",
  [HEADINGS.WEST]: "←",
  [HEADINGS.SKIP]: "·",
});

const orientationOffsets = Object.freeze({
  [HEADINGS.NORTH]: { dx: 0, dy: 1 },
  [HEADINGS.EAST]: { dx: 1, dy: 0 },
  [HEADINGS.SOUTH]: { dx: 0, dy: -1 },
  [HEADINGS.WEST]: { dx: -1, dy: 0 },
  [HEADINGS.SKIP]: { dx: 0, dy: 0 },
});

const defaultRobotState = { x: ROBOT_MIN, y: ROBOT_MIN, d: HEADINGS.NORTH, s: -1 };
const defaultRobotForm = { x: String(ROBOT_MIN), y: String(ROBOT_MIN), d: HEADINGS.NORTH };
const defaultObstacleForm = { x: "", y: "", d: HEADINGS.NORTH };

const robotHeadingOptions = [
  { value: HEADINGS.NORTH, label: "↑ Up" },
  { value: HEADINGS.SOUTH, label: "↓ Down" },
  { value: HEADINGS.WEST, label: "← Left" },
  { value: HEADINGS.EAST, label: "→ Right" },
];

const obstacleHeadingOptions = [
  { value: HEADINGS.NORTH, label: "↑ North" },
  { value: HEADINGS.SOUTH, label: "↓ South" },
  { value: HEADINGS.WEST, label: "← West" },
  { value: HEADINGS.EAST, label: "→ East" },
  { value: HEADINGS.SKIP, label: "· None" },
];

export default function Simulator() {
  const [robotForm, setRobotForm] = useState({ ...defaultRobotForm });
  const [robotState, setRobotState] = useState({ ...defaultRobotState });
  const [obstacleForm, setObstacleForm] = useState({ ...defaultObstacleForm });
  const [obstacles, setObstacles] = useState([]);
  const [simulation, setSimulation] = useState({ status: "idle", data: null, error: null });
  const [stepIndex, setStepIndex] = useState(0);
  const [allocateObstacleId, resetObstacleIds] = useSequentialId(1);

  const boardModel = useMemo(() => buildBoardModel(obstacles, robotState), [obstacles, robotState]);
  const busy = simulation.status === "running";

  useEffect(() => {
    if (simulation.status !== "done" || !simulation.data) {
      return;
    }
    const boundedIndex = Math.max(0, Math.min(stepIndex, simulation.data.path.length - 1));
    const frame = simulation.data.path[boundedIndex];
    if (frame) {
      setRobotState(frame);
    }
  }, [simulation.status, simulation.data, stepIndex]);

  const getManualRobot = () => {
    const x = parseBoundedInt(robotForm.x, ROBOT_MIN, ROBOT_MAX) ?? defaultRobotState.x;
    const y = parseBoundedInt(robotForm.y, ROBOT_MIN, ROBOT_MAX) ?? defaultRobotState.y;
    return { x, y, d: robotForm.d, s: -1 };
  };

  const clearSimulation = (preserveError = false) => {
    setSimulation((prev) => ({
      status: "idle",
      data: null,
      error: preserveError ? prev.error : null,
    }));
    setStepIndex(0);
  };

  const handleRobotChange = (field) => (event) => {
    const value = field === "d" ? Number(event.target.value) : event.target.value;
    setRobotForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleRobotUpdate = () => {
    const next = getManualRobot();
    setRobotForm({ x: String(next.x), y: String(next.y), d: next.d });
    setRobotState(next);
    clearSimulation(false);
  };

  const handleObstacleChange = (field) => (event) => {
    const value = field === "d" ? Number(event.target.value) : event.target.value;
    setObstacleForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleAddObstacle = () => {
    const x = parseBoundedInt(obstacleForm.x, 0, BOARD_LIMIT - 1);
    const y = parseBoundedInt(obstacleForm.y, 0, BOARD_LIMIT - 1);

    if (x === null || y === null) {
      setSimulation({ status: "error", data: null, error: "Obstacle coordinates must be integers between 0 and 19." });
      return;
    }

    const duplicate = obstacles.some((obstacle) => obstacle.x === x && obstacle.y === y);
    if (duplicate) {
      setSimulation({ status: "error", data: null, error: "An obstacle already exists at those coordinates." });
      return;
    }

    const id = allocateObstacleId();
    setObstacles((current) => [...current, { id, x, y, d: obstacleForm.d }]);
    setObstacleForm((prev) => ({ ...prev, x: "", y: "" }));
    clearSimulation(false);
  };

  const handleRemoveObstacle = (id) => {
    if (busy) {
      return;
    }
    setObstacles((current) => current.filter((obstacle) => obstacle.id !== id));
    clearSimulation(false);
  };

  const runSimulation = async () => {
    if (!obstacles.length) {
      setSimulation({ status: "error", data: null, error: "Please add at least one obstacle before running the simulation." });
      return;
    }

    const manualRobot = getManualRobot();
    setRobotState(manualRobot);
    setStepIndex(0);
    setSimulation({ status: "running", data: null, error: null });

    try {
      const payload = {
        obstacles,
        robot_x: manualRobot.x,
        robot_y: manualRobot.y,
        robot_dir: manualRobot.d,
      };

      const response = await QueryAPI.runSimulation(payload);
      const path = Array.isArray(response?.path)
        ? response.path
        : Array.isArray(response?.data?.path)
        ? response.data.path
        : null;
      const commands = Array.isArray(response?.commands)
        ? response.commands
        : Array.isArray(response?.data?.commands)
        ? response.data.commands
        : [];

      if (!path || path.length === 0) {
        throw new Error("Simulation returned no path data.");
      }

      setSimulation({ status: "done", data: { path, commands }, error: null });
      setRobotState(path[0]);
    } catch (error) {
      const message = error?.message ?? "An unknown error occurred during simulation.";
      setSimulation({ status: "error", data: null, error: message });
    }
  };

  const resetPath = () => {
    clearSimulation(false);
    setRobotState(getManualRobot());
  };

  const resetAll = () => {
    clearSimulation(false);
    resetObstacleIds();
    setObstacles([]);
    setObstacleForm({ ...defaultObstacleForm });
    setRobotForm({ ...defaultRobotForm });
    setRobotState({ ...defaultRobotState });
  };

  const goToPreviousStep = () => {
    setStepIndex((index) => Math.max(0, index - 1));
  };

  const goToNextStep = () => {
    if (!simulation.data) {
      return;
    }
    setStepIndex((index) => Math.min(simulation.data.path.length - 1, index + 1));
  };

  return (
    <div className="min-h-screen bg-slate-50 py-8 px-4">
      <div className="mx-auto max-w-6xl space-y-8">
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-slate-800">MDP Path Planning Sandbox</h1>
          <p className="text-base text-slate-500">Configure obstacles, run the solver, and inspect the robot&apos;s journey.</p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
          <aside className="space-y-6">
            <SectionCard title="Robot placement" subtitle="Choose the starting pose" icon="🤖">
              <div className="grid grid-cols-3 gap-3">
                <FormField label="X" htmlFor="robot-x">
                  <input
                    id="robot-x"
                    type="number"
                    min={ROBOT_MIN}
                    max={ROBOT_MAX}
                    value={robotForm.x}
                    onChange={handleRobotChange("x")}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-200"
                  />
                </FormField>
                <FormField label="Y" htmlFor="robot-y">
                  <input
                    id="robot-y"
                    type="number"
                    min={ROBOT_MIN}
                    max={ROBOT_MAX}
                    value={robotForm.y}
                    onChange={handleRobotChange("y")}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-200"
                  />
                </FormField>
                <FormField label="Heading" htmlFor="robot-heading">
                  <select
                    id="robot-heading"
                    value={robotForm.d}
                    onChange={handleRobotChange("d")}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-200"
                  >
                    {robotHeadingOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </FormField>
              </div>
              <button
                type="button"
                onClick={handleRobotUpdate}
                className="w-full rounded-lg bg-gradient-to-r from-sky-500 to-indigo-500 px-4 py-2 text-sm font-semibold text-white shadow-md transition hover:from-sky-600 hover:to-indigo-600 focus:outline-none focus:ring-2 focus:ring-sky-300"
              >
                Apply robot position
              </button>
            </SectionCard>

            <SectionCard title="Obstacles" subtitle="Stage the arena" icon="🧱">
              <div className="grid grid-cols-3 gap-3">
                <FormField label="X" htmlFor="obstacle-x">
                  <input
                    id="obstacle-x"
                    type="number"
                    min={0}
                    max={BOARD_LIMIT - 1}
                    value={obstacleForm.x}
                    onChange={handleObstacleChange("x")}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:border-amber-500 focus:outline-none focus:ring-2 focus:ring-amber-200"
                  />
                </FormField>
                <FormField label="Y" htmlFor="obstacle-y">
                  <input
                    id="obstacle-y"
                    type="number"
                    min={0}
                    max={BOARD_LIMIT - 1}
                    value={obstacleForm.y}
                    onChange={handleObstacleChange("y")}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:border-amber-500 focus:outline-none focus:ring-2 focus:ring-amber-200"
                  />
                </FormField>
                <FormField label="Facing" htmlFor="obstacle-heading">
                  <select
                    id="obstacle-heading"
                    value={obstacleForm.d}
                    onChange={handleObstacleChange("d")}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-slate-900 focus:border-amber-500 focus:outline-none focus:ring-2 focus:ring-amber-200"
                  >
                    {obstacleHeadingOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </FormField>
              </div>
              <button
                type="button"
                onClick={handleAddObstacle}
                disabled={busy}
                className="w-full rounded-lg bg-gradient-to-r from-amber-500 to-rose-500 px-4 py-2 text-sm font-semibold text-white shadow-md transition hover:from-amber-600 hover:to-rose-600 focus:outline-none focus:ring-2 focus:ring-rose-300 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Add obstacle
              </button>
              <ObstacleList obstacles={obstacles} onRemove={handleRemoveObstacle} disabled={busy} />
            </SectionCard>

            <PathPanel
              simulation={simulation}
              stepIndex={stepIndex}
              onPrev={goToPreviousStep}
              onNext={goToNextStep}
              onSelectStep={setStepIndex}
            />
          </aside>

          <main className="space-y-6">
            <BoardView board={boardModel} />
            <ActionBar onRun={runSimulation} onResetPath={resetPath} onResetAll={resetAll} busy={busy} />
            <StatusBanner message={simulation.error} />
          </main>
        </div>
      </div>
    </div>
  );
}

function BoardView({ board }) {
  return (
    <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-lg">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">Arena overview</h2>
          <p className="text-sm text-slate-500">20×20 grid with origin anchored at the bottom-left.</p>
        </div>
        <span className="text-2xl" role="img" aria-label="map">
          🗺️
        </span>
      </div>
      <div className="overflow-auto">
        <table className="border-collapse">
          <tbody>
            {board.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`}>
                <td className="h-8 w-8 border border-slate-200 bg-slate-100 text-center text-xs font-semibold text-slate-600 lg:h-10 lg:w-10">
                  {BOARD_LIMIT - 1 - rowIndex}
                </td>
                {row.map((cell) => (
                  <CellView key={`${cell.row}-${cell.col}`} cell={cell} />
                ))}
              </tr>
            ))}
            <tr>
              <td className="h-8 w-8 border border-slate-200 bg-slate-100 lg:h-10 lg:w-10" />
              {Array.from({ length: BOARD_LIMIT }, (_, idx) => (
                <td
                  key={`axis-${idx}`}
                  className="h-8 w-8 border border-slate-200 bg-slate-100 text-center text-xs font-semibold text-slate-600 lg:h-10 lg:w-10"
                >
                  {idx}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CellView({ cell }) {
  const base = "relative h-8 w-8 border border-slate-200 transition-colors lg:h-10 lg:w-10";
  switch (cell.kind) {
    case "obstacle":
      return (
        <td className={cx(base, "bg-rose-500 text-white border-rose-600")}>
          <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold">
            {headingGlyph[cell.direction] ?? "●"}
          </span>
        </td>
      );
    case "robot-core":
      return (
        <td className={cx(base, "bg-sky-500 text-white border-sky-700")}>
          <span className="absolute inset-0 flex items-center justify-center text-base font-semibold">
            {cell.screenshot !== -1 ? cell.screenshot : "•"}
          </span>
        </td>
      );
    case "robot-marker":
      return (
        <td className={cx(base, "bg-sky-400 text-white border-sky-600")}>
          <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold">
            {headingGlyph[cell.direction] ?? "▲"}
          </span>
        </td>
      );
    case "robot-body":
      return <td className={cx(base, "bg-sky-200/70 border-sky-300")} />;
    default:
      return <td className={cx(base, "bg-white hover:bg-slate-50")} />;
  }
}

function SectionCard({ title, subtitle, icon, children }) {
  return (
    <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-lg">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">{title}</h2>
          {subtitle ? <p className="text-sm text-slate-500">{subtitle}</p> : null}
        </div>
        {icon ? (
          <span className="text-2xl" aria-hidden="true">
            {icon}
          </span>
        ) : null}
      </div>
      <div className="space-y-4">{children}</div>
    </section>
  );
}

function FormField({ label, htmlFor, children }) {
  return (
    <div className="flex flex-col space-y-1">
      <label htmlFor={htmlFor} className="text-sm font-medium text-slate-700">
        {label}
      </label>
      {children}
    </div>
  );
}

function ObstacleList({ obstacles, onRemove, disabled }) {
  if (!obstacles.length) {
    return <p className="text-sm text-slate-500">No obstacles added yet.</p>;
  }

  return (
    <ul className="max-h-48 space-y-2 overflow-y-auto pr-1">
      {obstacles.map((obstacle) => (
        <li
          key={obstacle.id}
          className="flex items-center justify-between rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-700"
        >
          <span className="font-mono">
            #{obstacle.id} · ({obstacle.x}, {obstacle.y}) · {headingLabels[obstacle.d] ?? "None"}
          </span>
          <button
            type="button"
            onClick={() => onRemove(obstacle.id)}
            disabled={disabled}
            className="text-rose-600 transition hover:text-rose-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Remove
          </button>
        </li>
      ))}
    </ul>
  );
}

function PathPanel({ simulation, stepIndex, onPrev, onNext, onSelectStep }) {
  if (simulation.status !== "done" || !simulation.data) {
    return null;
  }

  const steps = simulation.data.path;
  const commands = simulation.data.commands ?? [];

  return (
    <SectionCard title="Path timeline" subtitle="Scrub through the computed route" icon="🧭">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-500">Current frame</p>
          <p className="text-xl font-semibold text-slate-800">
            {stepIndex + 1} / {steps.length}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onPrev}
            disabled={stepIndex === 0}
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Previous
          </button>
          <button
            type="button"
            onClick={onNext}
            disabled={stepIndex >= steps.length - 1}
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
      <div className="max-h-40 space-y-1 overflow-y-auto rounded-xl bg-slate-900 p-3">
        {steps.map((step, index) => (
          <button
            key={`step-${index}`}
            type="button"
            onClick={() => onSelectStep(index)}
            className={cx(
              "w-full rounded-md px-2 py-1 text-left text-xs font-mono transition-colors",
              index === stepIndex ? "bg-emerald-500/20 text-emerald-200" : "text-slate-400 hover:text-slate-200"
            )}
          >
            {`#${index + 1}  x:${step.x}  y:${step.y}  dir:${headingLabels[step.d] ?? step.d}`}
          </button>
        ))}
      </div>
      {commands.length > 0 ? (
        <div className="space-y-1">
          <p className="text-sm font-medium text-slate-700">Command stream</p>
          <p className="break-words text-xs text-slate-500">{commands.join(", ")}</p>
        </div>
      ) : null}
    </SectionCard>
  );
}

function ActionBar({ onRun, onResetPath, onResetAll, busy }) {
  return (
    <SectionCard title="Simulation controls" subtitle="Trigger the planner or reset the arena" icon="⚙️">
      <div className="grid gap-3 sm:grid-cols-3">
        <button
          type="button"
          onClick={onRun}
          disabled={busy}
          className="rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 px-4 py-2 text-sm font-semibold text-white shadow-md transition hover:from-emerald-600 hover:to-teal-600 focus:outline-none focus:ring-2 focus:ring-emerald-300 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy ? "Computing…" : "Run simulation"}
        </button>
        <button
          type="button"
          onClick={onResetPath}
          className="rounded-lg bg-gradient-to-r from-amber-500 to-yellow-500 px-4 py-2 text-sm font-semibold text-white shadow-md transition hover:from-amber-600 hover:to-yellow-600 focus:outline-none focus:ring-2 focus:ring-amber-300"
        >
          Reset path
        </button>
        <button
          type="button"
          onClick={onResetAll}
          className="rounded-lg bg-gradient-to-r from-rose-500 to-pink-500 px-4 py-2 text-sm font-semibold text-white shadow-md transition hover:from-rose-600 hover:to-pink-600 focus:outline-none focus:ring-2 focus:ring-rose-300"
        >
          Reset all
        </button>
      </div>
    </SectionCard>
  );
}

function StatusBanner({ message }) {
  if (!message) {
    return null;
  }

  return (
    <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700 shadow-sm">
      <p className="text-sm font-semibold">Error</p>
      <p className="text-sm">{message}</p>
    </div>
  );
}

function cx(...values) {
  return values.filter(Boolean).join(" ");
}

function parseBoundedInt(value, min, max) {
  const parsed = Number(value);
  if (Number.isInteger(parsed) && parsed >= min && parsed <= max) {
    return parsed;
  }
  return null;
}

function projectToBoard(x, y) {
  return {
    row: BOARD_LIMIT - 1 - y,
    col: x,
  };
}

function withinBoard(row, col) {
  return row >= 0 && row < BOARD_LIMIT && col >= 0 && col < BOARD_LIMIT;
}

function createRobotFootprint(robot) {
  const footprint = [];
  const front = orientationOffsets[robot.d] ?? { dx: 0, dy: 0 };

  for (let dx = -1; dx <= 1; dx += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      const { row, col } = projectToBoard(robot.x + dx, robot.y + dy);
      let variant = "body";
      if (dx === 0 && dy === 0) {
        variant = "core";
      } else if (dx === front.dx && dy === front.dy) {
        variant = "marker";
      }
      footprint.push({ row, col, variant });
    }
  }

  return footprint;
}

function buildBoardModel(obstacles, robot) {
  const matrix = Array.from({ length: BOARD_LIMIT }, (_, row) =>
    Array.from({ length: BOARD_LIMIT }, (_, col) => ({
      kind: "empty",
      row,
      col,
    }))
  );

  obstacles.forEach((obstacle) => {
    const { row, col } = projectToBoard(obstacle.x, obstacle.y);
    if (!withinBoard(row, col)) {
      return;
    }
    matrix[row][col] = {
      kind: "obstacle",
      row,
      col,
      direction: obstacle.d,
      id: obstacle.id,
    };
  });

  createRobotFootprint(robot).forEach((cell) => {
    if (!withinBoard(cell.row, cell.col)) {
      return;
    }
    const existing = matrix[cell.row][cell.col];
    if (existing.kind === "obstacle") {
      return;
    }

    if (cell.variant === "core") {
      matrix[cell.row][cell.col] = {
        kind: "robot-core",
        row: cell.row,
        col: cell.col,
        direction: robot.d,
        screenshot: robot.s,
      };
    } else if (cell.variant === "marker") {
      matrix[cell.row][cell.col] = {
        kind: "robot-marker",
        row: cell.row,
        col: cell.col,
        direction: robot.d,
      };
    } else {
      matrix[cell.row][cell.col] = {
        kind: "robot-body",
        row: cell.row,
        col: cell.col,
      };
    }
  });

  return matrix;
}

function useSequentialId(start = 1) {
  const counter = useRef(start);
  const allocate = () => counter.current++;
  const reset = () => {
    counter.current = start;
  };
  return [allocate, reset];
}

