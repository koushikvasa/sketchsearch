# FRIDAY.md: swap SketchSearch onto the VAST stack

Build day runbook. Everything that runs on free stand-ins today sits behind the `VideoSource` adapter
(`app/sources/`) or a config flag, so the swap is mostly configuration plus confirming a few formats.
Every guess in the code is marked `# FRIDAY-VERIFY:` (`grep -rn FRIDAY-VERIFY app/`).

Source for every VAST detail below: the `vast-builders-challenge` repo (`README.md`, `ARCHITECTURE_REFERENCE.md`,
`config.example`, `.cursor/skills/**`). The VM already has its env vars set from `/config/<team>.config`.

## What swaps

| Piece | Today (stand-in) | Friday (VAST) | How |
|---|---|---|---|
| Segments + tracks | GT index from `ground_truth.json` (`LocalSource`) | `vss-collection` rows in VastDB, boxes from `perception_json` or the `/videos/detections` sidecar | `SOURCE=vast` |
| Text search | CLIP keyframes (off by default) | `POST /api/v1/search` (hybrid caption + visual) | automatic in `VastSource.text_search` |
| Clips | `data/segments/*.mp4` | `GET /api/v1/videos/stream?source=…&token=…`, proxied by `/api/vast/clip/{id}` (the JWT never reaches the browser) | automatic |
| Verification (`ask`) | Gemini, inline clip | Cosmos3-Reason `POST $COSMOS3_REASON_URL/v1/chat/completions`, clip as a base64 `video_url`; falls back to Gemini if Cosmos fails and `GEMINI_API_KEY` is set | automatic |
| LLM (words → sketch, agent) | Groq `openai/gpt-oss-120b` | W&B Inference `https://api.inference.wandb.ai/v1`, same model | `LLM_PROVIDER=wandb` (tested 2026-10-07 with our key; W&B's JSON mode garbles gpt-oss output, so we send plain text + `reasoning_effort: low`) |
| Hosting | `uvicorn` on localhost | `deploy-app-no-registry`: `python:3.12-slim`, code from a ConfigMap, Ingress `/app` | `scripts/package_k8s.py` |

## 60-minute swap checklist

**0-10 min: get the code running on the VM**
```bash
cd ~ && git clone https://github.com/koushikvasa/sketchsearch && cd sketchsearch   # private: gh auth login first
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra vast            # adds vastdb + pyarrow
cp .env.example .env            # then edit: SOURCE=vast, LLM_PROVIDER=wandb, GEMINI_API_KEY (fallback), VAST_CAMERA_IDS
env | cut -d= -f1 | sort        # check names only; never print values
```
`.env` does not override variables that are already set, so the VM's `/config` values win. If a value is
missing, `source /config/<team>.config`.

**10-25 min: answer the open questions**
```bash
uv run --extra vast python -m scripts.vast_probe --camera sdg_warehouse_cam-2
```
This prints PASS/FAIL for login, dashboard labels, metadata schema, a few VastDB rows (columns, timing
fields, a `perception_json` sample, and what our parser makes of it), search, the detections sidecar, clip
streaming, Cosmos3-Reason (models + one real `ask`), the YOLO OpenAPI paths and W&B inference.
The raw samples go to `data/vast_probe.json`.

Cursor prompts that answer the same questions (from the skills):
- `Using vastdb-read, show 3 rows from vss-collection for camera_id sdg_warehouse_cam-2 with every non-vector column, and print perception_json in full.`
- `Show the dashboard stats objects[] list: which object labels exist and how many segments each has.`
- `For one segment source from a search for "forklift near a person", call GET /api/v1/videos/detections and show the JSON.`
- `Fetch $YOLO_URL/openapi.json and show the /v1/infer response schema (box format, frames).`
- `/ask-cosmos` and `check that everything is working` for the GPU endpoints.

**25-40 min: adapt to what you saw** (most of it is one function each)

| Question | If the probe shows… | Change |
|---|---|---|
| Box format | xywh, or normalized coords under another key | `_to_box` / `BOX_KEYS` in `app/sources/vast.py` |
| Per-frame vs per-segment | only per-segment boxes in `perception_json` | `VAST_DETECTIONS=sidecar` (per-frame boxes from `/videos/detections`), or accept `keyframe_only` (see Fallbacks) |
| Timing columns | real names for segment start/end | `START_KEYS` / `END_KEYS`; `SEGMENT_SECONDS_GUESS` (5 s) |
| Labels | `truck` instead of `forklift`, `person` OK | `VastSource.label_aliases` (e.g. `{"forklift": ["truck"]}`), or `VAST_CAPTION_LABELS=forklift` |
| Frame size | pixel boxes without width/height in the JSON | `FRAME_SIZE_GUESS` |
| Pack | which `camera_id` holds the warehouse pack | `VAST_CAMERA_IDS=sdg_warehouse_cam-2` (Pack C) |
| GPU auth | `GPU_BEARER_TOKEN` set and required | nothing: it is sent when set |

Then `uv run pytest -q` (54 tests, offline). Update `tests/test_vast_source.py` fixtures to the real shapes
you saw so the tests keep guarding the parser.

**40-50 min: run it end to end on the VM**
```bash
SOURCE=vast uv run --extra vast uvicorn app.main:app --port 8000
curl -s localhost:8000/api/health        # source, segments/tracks loaded, LLM + verify models
```
- Draw the "Person approaches forklift" preset, Search, then **AI check top 10**. Cosmos verdicts should stream in.
- Run the agent preset once.
- **After connecting their data, run `make_presets.py`** so the landing page's example chips fit *their* footage:
  ```bash
  SOURCE=vast uv run --extra vast python -m scripts.make_presets
  ```
  It scans every window of the index for four kinds of real moment (something walking toward something
  else, two meeting, an object with nobody near it, a group), uses only labels that exist in the index,
  keeps a candidate only if its search returns at least 5 Good matches (score >= 0.70), writes
  `data/presets_vast.json` (the UI picks it up on reload) and warms the AI-check cache for the top 10 of each.
  `--no-warm` skips the AI checks; delete the file to fall back to the built-in examples.
- Run `uv run python -m scripts.warm_demo` to pre-cache the demo (presets + the recorded agent run). It reads
  the same `SOURCE`. Cache keys include the question and the model, so the verdicts are new for Cosmos.

**50-60 min: deploy** (next section). Smoke-test `https://<app host>/app` and `/app/health`.

## Deploy with deploy-app-no-registry

The skill runs `python main.py` from `/code` in `python:3.12-slim`, pip-installs `requirements.txt` at
start, listens on `$PORT` (8080), probes `GET /health`, and puts an Ingress at `/app(/|$)(.*)` that **strips**
`/app`. A ConfigMap built with `--from-file=<dir>` only takes top-level files, so we ship one zip.

```bash
export KUBECONFIG=/config/kubeconfig
NS="$USERNAME"; TEAM_N="${USERNAME#team-}"; APP_HOST="video-lab-team-${TEAM_N}.cosmos.vastdata.com"
APP_NAME=sketchsearch

uv run python -m scripts.package_k8s          # deploy/k8s/build/: main.py, requirements.txt, sketchsearch.zip (~70 KiB)

kubectl -n "$NS" create configmap "${APP_NAME}-code" --from-file=deploy/k8s/build \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NS" create secret generic "${APP_NAME}-vss-creds" \
  --from-literal=VSS_URL="$INGRESS_URL" --from-literal=VSS_USERNAME="$USERNAME" \
  --from-literal=VSS_PASSWORD="$PASSWORD" --dry-run=client -o yaml | kubectl apply -f -
# The skill's pod only gets PORT + VSS_*; everything else goes in a second secret:
kubectl -n "$NS" create secret generic "${APP_NAME}-env" \
  --from-literal=SOURCE=vast --from-literal=ROOT_PATH=/app --from-literal=LLM_PROVIDER=wandb \
  --from-literal=WANDB_API_KEY="$WANDB_API_KEY" --from-literal=WANDB_MODEL=openai/gpt-oss-120b \
  --from-literal=WEAVE_PROJECT="${WANDB_PROJECT:-sketchsearch}" \
  --from-literal=COSMOS3_REASON_URL="$COSMOS3_REASON_URL" --from-literal=COSMOS3_REASON_MODEL="${COSMOS3_REASON_MODEL:-nvidia/cosmos3-reason}" \
  --from-literal=GPU_BEARER_TOKEN="${GPU_BEARER_TOKEN:-}" \
  --from-literal=VDB_ENDPOINT="${VDB_ENDPOINT:-$S3_ENDPOINT}" --from-literal=ACCESS_KEY="$ACCESS_KEY" \
  --from-literal=SECRET_KEY="$SECRET_KEY" --from-literal=VASTDB_BUCKET="$VASTDB_BUCKET" \
  --from-literal=VDB_SCHEMA="${VDB_SCHEMA:-vss-schema}" --from-literal=VDB_COLLECTION="${VDB_COLLECTION:-vss-collection}" \
  --from-literal=VAST_CAMERA_IDS=sdg_warehouse_cam-2 --from-literal=GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
  --dry-run=client -o yaml | kubectl apply -f -
```
Then tell Cursor: **`/deploy-app-no-registry` with APP_NAME=sketchsearch and the code from
deploy/k8s/build. Add `envFrom: [{secretRef: {name: sketchsearch-env}}]` to the container. Keep the
readinessProbe on /health and give it initialDelaySeconds 60 (the pod pip-installs numpy/scipy/vastdb at start).**

Check:
```bash
kubectl -n "$NS" rollout status deploy/sketchsearch
kubectl -n "$NS" logs -l app=sketchsearch --tail=100     # "VastSource: N segments loaded", "matcher ready"
curl -s "http://${APP_HOST}/app/health"; curl -s "http://${APP_HOST}/app/api/health"
```
People open it from https://workshop.thecosmoslabs.com → **App** (not `$INGRESS_URL`).
To update: rerun `package_k8s` and the configmap command, then `kubectl -n "$NS" rollout restart deploy/sketchsearch`.

The app handles both forms of path: `ROOT_PATH=/app` puts `<base href="/app/">` in the page, and
`EnsureRootPath` normalizes requests whether or not the prefix was stripped (tested locally both ways,
plus Playwright at `/app` without a trailing slash).

## Env vars

| Var | From | Used for |
|---|---|---|
| `INGRESS_URL` / `VSS_URL`, `USERNAME` / `VSS_USERNAME`, `PASSWORD` / `VSS_PASSWORD` | `/config` (VM) / the skill's secret (pod). `VSS_*` win | VSS backend + JWT |
| `VDB_ENDPOINT` (else `S3_ENDPOINT`), `ACCESS_KEY`/`SECRET_KEY` (or `VAST_*`), `VASTDB_BUCKET`, `VDB_SCHEMA`, `VDB_COLLECTION` | `/config` | VastDB rows |
| `COSMOS3_REASON_URL`, `COSMOS3_REASON_MODEL`, `GPU_BEARER_TOKEN` (optional) | `/config` | verification |
| `WANDB_API_KEY`, `WANDB_MODEL` | `/config` + ours | W&B Inference + Weave |
| `SOURCE=vast`, `LLM_PROVIDER=wandb`, `ROOT_PATH=/app` | ours | switches |
| `VAST_CAMERA_IDS`, `VAST_CAPTION_LABELS`, `VAST_DETECTIONS`, `VAST_ROWS_FILE` | ours | pack filter + fallbacks |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | ours | verification fallback |

## Fallbacks

1. **Per-segment boxes only (no per-frame tracks).** The parser marks such segments `keyframe_only`. The
   matcher then scores layout only (position, size, relations, renormalized), and motion is left to the
   video model's verification. Better: `VAST_DETECTIONS=sidecar`, which loads per-frame boxes from
   `/videos/detections` at startup, 8 at a time. Or call `$YOLO_URL/v1/infer` on the candidate subset.
2. **No forklift label** (YOLO11s is COCO: expect `truck` at best). Set `VAST_CAPTION_LABELS=forklift`.
   Forklift sketches then search the segments whose Cosmos captions mention a forklift (`/api/v1/search`
   plus a caption check), and the rest of the sketch is matched geometrically. If `truck` boxes really
   are forklifts in Pack C, map them instead: `label_aliases = {"forklift": ["truck"]}`.
3. **Cosmos3-Reason slow, overloaded or empty.** Empty `content` is the documented overload signal.
   With `GEMINI_API_KEY` set, `ask()` falls back to Gemini on the same clip bytes. Raise
   `VERIFY_TIMEOUT_S` if Cosmos is just slow.
4. **VastDB unreachable from the pod.** On the VM run
   `python -m scripts.vast_probe --snapshot data/rows.json.gz`, ship the file, and set `VAST_ROWS_FILE`.
   Keep it under ~1 MiB for a ConfigMap, or mount it another way.
5. **W&B Inference down or rejecting the key.** `LLM_PROVIDER=groq` (needs `GROQ_API_KEY`). If W&B
   needs a team/project header, add it in `app/llm/__init__.py` (`default_headers`).
6. **Pod can't pip-install** (no PyPI egress). Run the app on the VM for the demo and keep trying the
   deploy, since the rules want a deployed app.

## Demo-day checks

- `/app/api/health` shows `"source": "vast"`, a non-zero segment count, the Cosmos model and `"weave_tracing": true`.
- Turn on **Demo mode** after `warm_demo.py` (⋮ menu, top right, or press `D`; a Demo badge shows it is on).
  Example chips then show all 10 verdicts instantly and the agent replays. New sketches still auto-check only the top 3 live.
- Presenter keys: `←`/`→` previous or next match, `Space` play or pause, `C` check more with AI, `?` quick guide.
- Backup: record a screen capture of the local (GT) demo tonight.
