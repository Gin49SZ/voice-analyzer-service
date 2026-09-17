# AGENTS.md — voice-analyzer-service

## Entry point
- `server.py` — FastAPI app with uvicorn

## Run
```bash
python server.py --port 27000                     # CPU
python server.py --port 27000 --gpu true           # GPU
```

## Docker

One `Dockerfile` (no `-f` needed) orchestrated by `docker-compose.yml`. The build is
**zero-network** — every dependency comes from a local directory.

| service | profile | Role | Network |
|---|---|---|---|
| `deps` | `online` | downloads system debs and python wheels into host `./debhouse/`, `./wheelhouse/` | needs internet |
| `app` | default | builds (`target: runtime`) and runs the image | **none** (`build.network: none`) |

```bash
sh scripts/build.sh                              # one command: fetch if missing, then build + up
# manual equivalent
docker compose --profile online run --rm deps    # only when local deps are incomplete
docker compose up -d --build
```

Stages — `base` (installs libsndfile1 / ffmpeg from local `.deb`) → `pydeps`
(`--no-index` from `wheelhouse/`) → `runtime` (`FROM base` + `COPY --from=pydeps
/opt/venv`). `runtime` is defined exactly once; compile tools only ever exist inside the
throwaway `deps` container. CPU-only torch (`torch==2.3.0+cpu`).

Local deps are **mounted** with `RUN --mount=type=bind`, never `COPY`ed — otherwise the
128 MB of `.deb` plus 710 MB of `.whl` stay in image layers forever, and deleting them in
a later `RUN` does not shrink the image. Any new `COPY` of local material belongs in a
discarded stage.

Gotchas that bite:
- **`torchaudio` has no ffmpeg backend in this image.** `torchaudio 2.3.0+cpu` (the PyPI wheel)
  reports `list_audio_backends() == ['soundfile']`, and `backend="ffmpeg"` raises
  `ValueError: Unsupported backend`. `libtorchaudio.so` contains no `libav*` strings at all.
  So webm/m4a **must** be decoded by shelling out to the `ffmpeg` CLI
  (`decode_with_ffmpeg()` in `server.py`). This is the only reason `ffmpeg` is a system
  dependency — nothing else calls it. Do not "simplify" it away.
- `scripts/fetch_deps.sh` decodes `requirements.txt` as UTF-16 and strips the
  `--extra-index-url` / `--index-url` lines (the file hardcodes `download.pytorch.org`,
  which stalls on large responses). **Never grep that file** — grep treats it as binary,
  emits 0 bytes, and `pip install -r <empty>` exits 0, silently skipping everything.
- apt must use **https** mirrors: the image's default `http://deb.debian.org` returns 502
  on this network. The script rewrites the sources before `apt-get update`.
- The `.deb` closure must be downloaded **before** the build toolchain is installed: apt
  never re-downloads packages that are already installed in the container, and
  `gcc` / `python3-dev` drag in a pile of libraries. Download first=205 debs, download
  after=202, and the missing ones make `dpkg -i` fail inside the build.
- Wheels are fetched in two passes — PyPI for everything, then the PyTorch index with
  `--no-deps` for `torch` / `torchaudio` only (`+cpu` local versions exist nowhere else).
- `wheelhouse/.gitkeep` and `debhouse/.gitkeep` must exist, otherwise the `--mount`
  source paths are missing.
- Mirror availability was verified with **large** files (small ones lie): aliyun PyPI OK;
  ustc / pypi.org stall past ~40 MB; tuna unreachable; SJTU `pytorch-wheels` 18.5 MB/s.
- `HEALTHCHECK --start-period=300s`: loading SV + ASR + VAD takes 75–110 s normally and
  has been measured at 263 s on a busy machine. A too-short start period leaves the
  container stuck at `unhealthy` (it is not restarted for that).

## API endpoints
- `POST /transcribe` — file upload, requires `apiKey` query param
- `WS /ws/transcribe` — streaming audio, requires `apiKey` query param; text frame
  `flush`/`end`/`done`/`finish` triggers tail flush (replies `code=3` ack)
- `GET /health`

## Responses
`TranscriptionResponse = {code, info, data, sentences}`.
- `data` — plain text, unchanged from legacy (whole-file ASR for HTTP; per-segment for WS)
- `sentences` — `[{start_ms, end_ms, text}]`, ms relative to audio start, from VAD boundaries
- `timestamps.py` — pure helpers (punctuation split + intra-segment interpolation), no model deps
- HTTP upload: `audio/wav` → `soundfile`; `audio/webm` → ffmpeg CLI → `soundfile`; anything else → 400
- VAD calls **must** go through `vad_offline()` / `vad_stream()`: funasr's
  `AutoModel.inference` writes this call's kwargs into the shared model object, so mixing
  HTTP and WS calls without explicit `chunk_size` / `is_streaming_input` / `is_final`
  corrupts the offline path (see `docs/API.md` §7.5)

## Models (offline, gitignored via `models/`)
Loaded with `local_files_only=True`; `MODELSCOPE_CACHE` / `FUNASR_CACHE` point at `models/`:
- ASR: `SenseVoiceSmall` (funasr)
- VAD: `speech_fsmn_vad_zh-cn-16k-common-pytorch`
- Speaker Verification: `speech_eres2net_large_sv_zh-cn_3dspeaker_16k`

## Auth
Default API key: `sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c` (`Config.api_key`).
A bad key is rejected **before** `accept()`, so Starlette turns it into an HTTP 403
handshake failure — browsers only ever see close code `1006`, never `1008`.

## Dependencies
- `requirements.txt`, UTF-16 encoded, pinned; first line is `--extra-index-url https://download.pytorch.org/whl/cpu`
- Key deps: fastapi, uvicorn, funasr, modelscope, torch, torchaudio, soundfile, loguru

## Test clients
- `test_client.html` — HTTP upload test (open in browser)
- `test_client_wss.html` — WebSocket streaming test (needs microphone)

## Notes
- No test suite, no linter/formatter config
- Default branch: `master` (remote uses `main`)
- Git proxy configured globally: `http.proxy=http://127.0.0.1:7890`
