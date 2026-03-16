import { spawn } from "node:child_process";
import path from "node:path";

const frontendRoot = process.cwd();
const repoRoot = path.resolve(frontendRoot, "..");
const bridgeScript = path.join(frontendRoot, "scripts", "fleet_bridge.py");

function pythonCommandParts() {
  const binary = process.env.GAT_PYTHON_BIN ?? "python3";
  const extraArgs = (process.env.GAT_PYTHON_ARGS ?? "")
    .split(/\s+/)
    .filter(Boolean);
  return { binary, extraArgs };
}

export async function runBridge(
  command: "start" | "state" | "step",
  payload: unknown = {},
  sessionId?: string,
) {
  const { binary, extraArgs } = pythonCommandParts();
  const args = [...extraArgs, bridgeScript, command];
  if (sessionId) {
    args.push(sessionId);
  }

  return new Promise<unknown>((resolve, reject) => {
    const child = spawn(binary, args, {
      cwd: repoRoot,
      env: process.env,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", reject);

    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(stderr.trim() || stdout.trim() || "Python bridge failed."),
        );
        return;
      }

      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(
          new Error(
            `Could not parse Python bridge response: ${String(error)}`,
          ),
        );
      }
    });

    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}
