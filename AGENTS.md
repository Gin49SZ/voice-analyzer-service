# AGENTS.md — voice-analyzer-service

## Entry point
- `server.py` — FastAPI app with uvicorn

## Run
```bash
python server.py --port 27000                    # CPU
python server.py --port 27000 --gpu true          # GPU
```

## Docker
```bash
docker build -t voice-analyzer .
docker run -p 27000:27000 voice-analyzer
```
Dockerfile uses CPU-only torch (`torch==2.3.0+cpu`).

## API endpoints
- `POST /transcribe` — file upload, requires `apiKey` query param
- `WS /ws/transcribe` — streaming audio, requires `apiKey` query param; text frame `flush`/`end`/`done`/`finish` triggers tail flush (replies `code=3` ack)
- `GET /health`

## Responses
`TranscriptionResponse = {code, info, data, sentences}`.
- `data` — plain text, unchanged from legacy (whole-file ASR for HTTP; per-segment ASR for WS)
- `sentences` — `[{start_ms, end_ms, text}]`, times relative to audio start, derived from VAD segment boundaries
- `timestamps.py` — pure helpers (punctuation split + intra-segment interpolation), no model deps
- HTTP: runs VAD once on the whole audio then ASR per segment; `Config.data_from_full_asr=False` skips the extra whole-file ASR pass (use VAD-segment texts to build `data`)

## Models (offline, gitignored via `models/`)
All models load from `models/` directory with `local_files_only=True`:
- ASR: `SenseVoiceSmall` (funasr)
- VAD: `speech_fsmn_vad_zh-cn-16k-common-pytorch`
- Speaker Verification: `speech_eres2net_large_sv_zh-cn_3dspeaker_16k`

Set env vars `MODELSCOPE_CACHE` and `FUNASR_CACHE` to `models/` before loading.

## Auth
Default API key in code: `sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c` (configurable via `Config.api_key`).

## Test clients
- `test_client.html` — HTTP upload test (open in browser)
- `test_client_wss.html` — WebSocket streaming test (open in browser, uses microphone)
- Example audio files in `examples/`

## Dependencies
- `requirements.txt` with pinned versions
- Uses `--extra-index-url https://download.pytorch.org/whl/cpu` (first line)
- Key deps: fastapi, uvicorn, funasr, modelscope, torch, torchaudio, soundfile, loguru

## Notes
- No test suite, no linter/formatter config
- Default branch: `master` (remote uses `main`)
- Git proxy configured globally: `http.proxy=http://127.0.0.1:7890`
