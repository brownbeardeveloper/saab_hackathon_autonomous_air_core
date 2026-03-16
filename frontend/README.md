# Fleet Command Web

Simple Next.js client for the same turn-based fleet controls exposed in `play.py`.

## What it does

- Shows airbases, missions, aircraft, and AI advisor suggestions in the browser.
- Lets the client issue the same four command modes as the CLI:
  - `Skip`
  - `Quick Mission`
  - `Transfer`
  - `Detailed`
- Keeps Python as the game engine, so the web UI is replaying the real environment instead of a JS rewrite.

## Default model

The bridge uses this model by default:

```bash
models/best_model.zip
```

## Run it

```bash
cd frontend
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

## Python environment

The npm scripts already use the `gat-train` environment, so plain `npm run dev` works.

Under the hood the web app calls Python for the real fleet engine and model inference, and that environment is where the required Python packages live.

If you ever want to override that manually, these env vars are still supported:

```bash
export GAT_PYTHON_BIN=mamba
export GAT_PYTHON_ARGS="run -n gat-train python"
```

## Docker

The root project now includes a Docker setup that runs:

- the Next.js frontend
- the Python bridge
- the trained model from `FLEET_MODEL_PATH` (defaults to `models/best_model.zip`)

From the repo root:

```bash
docker compose up --build -d
docker compose logs -f web
```

## Notes

- Session files are stored in `.fleet_web_sessions/` in the project root.
- The browser session is restored from local storage if you refresh the page.
