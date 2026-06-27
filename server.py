from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
import numpy as np
import soundfile as sf
import argparse
import uvicorn
from urllib.parse import parse_qs
from modelscope.pipelines import pipeline
from loguru import logger
import sys
import json
import traceback
import time
import asyncio
import torch
import torchaudio
import io
import os

logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

os.environ['MODELSCOPE_CACHE'] = MODEL_DIR
os.environ['FUNASR_CACHE'] = MODEL_DIR

parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
parser.add_argument('--gpu', type=bool, default=False, help='Enable GPU acceleration')
args = parser.parse_args()


class Config(BaseSettings):
    api_key: str = Field("sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c", description="API密钥")
    sv_thr: float = Field(0.3, description="说话人验证阈值")
    chunk_size_ms: int = Field(300, description="每次处理300毫秒音频")
    sample_rate: int = Field(16000, description="音频采样率16kHz")
    bit_depth: int = Field(16, description="Bit depth")
    channels: int = Field(1, description="Number of audio channels")
    avg_logprob_thr: float = Field(-0.25, description="average logprob threshold")


config = Config()

sv_model_path = os.path.join(MODEL_DIR, 'speech_eres2net_large_sv_zh-cn_3dspeaker_16k')
asr_model_path = os.path.join(MODEL_DIR, 'SenseVoiceSmall')
vad_model_path = os.path.join(MODEL_DIR, 'speech_fsmn_vad_zh-cn-16k-common-pytorch')

logger.info(f"Loading speaker verification model from: {sv_model_path}")
sv_pipeline = pipeline(
    task='speaker-verification',
    model=sv_model_path,
    model_revision='v1.0.0',
    local_files_only=True
)

logger.info(f"Loading ASR model from: {asr_model_path}")
model_asr = AutoModel(
    model=asr_model_path,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0" if args.gpu else "cpu",
    disable_update=True,
    local_files_only=True
)

logger.info(f"Loading VAD model from: {vad_model_path}")
model_vad = AutoModel(
    model=vad_model_path,
    model_revision="v2.0.4",
    disable_pbar=True,
    max_end_silence_time=500,
    speech_noise_thres=0.6,
    disable_update=True,
    local_files_only=True
)


def speaker_verify(audio, sv_thr, reg_spks):
    """
    验证音频是否与已注册的说话人匹配
    @param audio 待检测的音频
    @param sv_thr 认为是相同说话人的得分阈值
    @param reg_spks 注册的说话人音频列表
    @return hit 是否是同一说话人
    """
    hit = False
    speaker = None
    for spk in reg_spks:
        res_sv = sv_pipeline([audio, spk], sv_thr)
        if res_sv["score"] >= sv_thr:
            hit = True
            speaker = spk
        logger.info(f"[speaker_verify] audio_len: {len(audio)}; sv_thr: {sv_thr}; hit: {hit}; {speaker}: {res_sv}")
    return hit, speaker


def asr(audio, lang, cache, use_itn=False):
    start_time = time.time()
    result = model_asr.generate(
        input=audio,
        cache=cache,
        language=lang.strip(),
        use_itn=use_itn,
        batch_size_s=60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result


def transcribe_with_timing(*args, **kwargs):
    start_time = time.time()
    result = model_asr.generate(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Transcription execution time: {elapsed_time:.2f} seconds")
    return result, elapsed_time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""

    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data
        ).model_dump()
    )


class TranscriptionResponse(BaseModel):
    code: int
    info: str
    data: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
        file: UploadFile = File(...),
        apiKey: str = ""
):
    try:
        if not apiKey or apiKey != config.api_key:
            raise HTTPException(
                status_code=403,
                detail="Invalid or missing API key. Please provide apiKey parameter."
            )

        file.file.seek(0)
        file_content = await file.read()

        logger.info(f"[DEBUG] UploadFile Object is {file}")
        if file.content_type.startswith('audio/wav'):
            input_wav, sr = sf.read(io.BytesIO(file_content), dtype=np.int16)
            bit_depth = sf.info(io.BytesIO(file_content)).subtype
            is16 = True if bit_depth == 'PCM_16' else False

        elif file.content_type.startswith('audio/webm'):
            input_wav, sr = torchaudio.load(io.BytesIO(file_content))
            dtype = input_wav.dtype
            is16 = True if dtype == np.int16 else False
            input_wav = input_wav.squeeze().numpy()
        else:
            raise HTTPException(status_code=400, detail="Unsupported audio format")

        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)

        if is16:
            input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max

        if sr != 16000:
            logger.info(f"[DEBUG] Audio data sample rate is {sr}")
            resampler = torchaudio.transforms.Resample(sr, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()

        async def generate_text():
            return await asyncio.to_thread(transcribe_with_timing,
                                           input=input_wav,
                                           cache={},
                                           language="zh",
                                           use_itn=True,
                                           batch_size=64)

        resp, elapsed_time = await generate_text()
        logger.info(f"[DEBUG] Transcribe raw response is {resp}")
        text = resp[0]["text"]
        logger.info(f'[DEBUG] res:{resp} text:{text}')

        response = TranscriptionResponse(
            code=0,
            info=f"success, transcription time: {elapsed_time:.2f} seconds",
            data=text
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Exception occurred", exc_info=True)
        response = TranscriptionResponse(
            code=1,
            info=str(e),
            data=""
        )
    return JSONResponse(content=response.model_dump())


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    cache = {}
    cache_asr = {}
    audio_buffer = np.array([], dtype=np.float32)
    audio_vad = np.array([], dtype=np.float32)

    try:
        query_params = parse_qs(websocket.scope['query_string'].decode())

        api_key_param = query_params.get('apiKey', [''])[0]
        if not api_key_param or api_key_param != config.api_key:
            await websocket.close(code=1008, reason="Invalid or missing API key")
            logger.warning(f"WebSocket connection rejected: invalid API key from {websocket.client}")
            return

        lang = query_params.get('lang', ['zh'])[0]
        reg_spks_param = query_params.get('reg_spks', [''])[0]

        if reg_spks_param:
            import urllib.parse
            decoded_param = urllib.parse.unquote(reg_spks_param)
            reg_spks = [url.strip() for url in decoded_param.split(',') if url.strip()]
        else:
            reg_spks = []

        await websocket.accept()

        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)

        last_vad_beg = last_vad_end = -1
        offset = 0
        hit = False

        buffer = b""
        while True:
            data = await websocket.receive_bytes()
            buffer += data

            if len(buffer) < 2:
                continue
            audio_buffer = np.append(
                audio_buffer,
                np.frombuffer(buffer[:len(buffer) - (len(buffer) % 2)], dtype=np.int16).astype(np.float32) / 32767.0
            )

            buffer = buffer[len(buffer) - (len(buffer) % 2):]

            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)

                if last_vad_beg > 1 and len(reg_spks) != 0 and not hit:
                    hit, speaker = speaker_verify(audio_vad[int((last_vad_beg - offset) * config.sample_rate / 1000):],
                                                  config.sv_thr, reg_spks)
                    if hit:
                        response = TranscriptionResponse(
                            code=2,
                            info="speaker",
                            data=speaker
                        )
                        await websocket.send_json(response.model_dump())

                res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1:
                            last_vad_beg = segment[0]
                        if segment[1] > -1:
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            logger.info(f"[vad segment] audio_len: {end - beg}")

                            result = None if not hit and len(reg_spks) != 0 else asr(audio_vad[beg:end], lang.strip(),
                                                                                     cache_asr, True)
                            logger.info(f"asr response: {result}")

                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            hit = False

                            if result is not None:
                                response = TranscriptionResponse(
                                    code=0,
                                    info=json.dumps(result[0], ensure_ascii=False),
                                    data=result[0]['text']
                                )
                                await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\nCall stack:\n{traceback.format_exc()}")
        await websocket.close()
    finally:
        if 'audio_buffer' in locals():
            audio_buffer = np.array([], dtype=np.float32)
        if 'audio_vad' in locals():
            audio_vad = np.array([], dtype=np.float32)
        if 'cache' in locals():
            cache.clear()
        if 'cache_asr' in locals():
            cache_asr.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
