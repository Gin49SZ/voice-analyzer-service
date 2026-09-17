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
from typing import List
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
import subprocess

from timestamps import build_sentences, clean_text

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
    data_from_full_asr: bool = Field(
        True,
        description="data 字段是否保留「整段一次性转写」文本（与旧版本完全一致）；"
                    "置为 False 时用 VAD 片段文本拼接，可省掉一次整段推理"
    )


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


# ---------------------------------------------------------------------------
# VAD 调用参数：必须【每次显式写全】chunk_size / is_streaming_input / is_final
#
# 原因（已在容器内实测复现）：
#   funasr 的 AutoModel.inference 前两行是
#       kwargs = self.kwargs            # 直接引用模型对象自己的字典
#       deep_update(kwargs, cfg)        # 把本次参数【就地】并进去，永久生效
#   HTTP 整段识别与 WS 流式识别共用同一个 model_vad，于是：
#     1) WS 传 chunk_size=300 / is_final=False → 永久粘在模型对象上；
#     2) 之后 HTTP 只传 fs / batch_size_s → inference 里
#        is_streaming_input = kwargs.get("is_streaming_input", True)  （因 chunk_size=300 < 15000）
#        is_final           = kwargs.get("is_final", False)            ← 读到残留的 False
#     3) forward 于是走流式分支，返回 [beg,-1] / [-1,end] 【事件对】而不是 [beg,end]【区间】，
#        build_sentences 把 22 个片段误读成 44 个 → sentences 数量暴涨、大量 start_ms=0。
#   实测：offline#1 = 22 个区间对；跑完一次 WS 流式会话后 offline#2 = 44 个事件对；
#         显式写全参数后 offline#3 = 22 个区间对。
#   结论：两条路径都显式声明自己的全部三个参数，任何一方都无法污染另一方。
# ---------------------------------------------------------------------------
VAD_OFFLINE_CHUNK_SIZE_MS = 60000  # 整段 VAD 分块大小，与 funasr 默认值一致


def vad_offline(audio):
    """整段（离线）VAD，返回 [beg_ms, end_ms] 区间对。

    @param audio 单声道 float32 音频（16kHz）
    @return model_vad.generate 的原始返回
    """
    return model_vad.generate(
        input=audio,
        fs=config.sample_rate,
        batch_size_s=60,
        chunk_size=VAD_OFFLINE_CHUNK_SIZE_MS,
        is_streaming_input=False,
        is_final=True,
    )


def vad_stream(chunk, cache, is_final=False):
    """流式 VAD（WS 路径用），返回 [beg,-1] / [-1,end] 边界事件。

    @param chunk 单个音频块（16kHz float32）；flush 时传空数组
    @param cache 本 WS 会话持有的持久 cache
    @param is_final 是否为本会话最后一次调用；flush 时必须 True，否则尾部片段丢失
    @return model_vad.generate 的原始返回
    """
    return model_vad.generate(
        input=chunk,
        cache=cache,
        is_final=is_final,
        is_streaming_input=True,
        chunk_size=config.chunk_size_ms,
    )


def vad_segments_of(audio):
    """整段音频跑一次 VAD，得到语音片段边界（毫秒，相对音频起点）。

    @param audio 单声道 float32 音频（16kHz）
    @return [[beg_ms, end_ms], ...]；无语音时为空列表
    """
    result = vad_offline(audio)
    if not result or not result[0].get("value"):
        return []
    raw = result[0]["value"]
    # 兜底探针：离线路径只应产出 [beg,end] 区间对。若出现 -1，说明参数被污染，走成了流式分支
    # （见上方「VAD 调用参数」注释）。这里不静默吞掉，直接把问题喊出来。
    negative = [seg for seg in raw if seg[0] < 0 or seg[1] < 0]
    if negative:
        logger.warning(f"[vad] offline path returned {len(negative)}/{len(raw)} streaming-style "
                       f"event pair(s) {negative[:4]} —— is_streaming_input 疑似被污染，"
                       f"请检查 VAD 调用参数是否写全")
    return [[int(seg[0]), int(seg[1])] for seg in raw]


def transcribe_by_vad_segments(audio):
    """按 VAD 片段逐段转写（SenseVoice 本身不输出时间戳，片段边界即时间来源）。

    @param audio 单声道 float32 音频（16kHz）
    @return (segments_ms, segment_texts) 片段边界，以及一一对应的原始文本（含控制标记）
    """
    start_time = time.time()
    segments_ms = vad_segments_of(audio)
    segment_texts = []
    for beg_ms, end_ms in segments_ms:
        beg = max(int(beg_ms * config.sample_rate / 1000), 0)
        end = min(int(end_ms * config.sample_rate / 1000), len(audio))
        if end <= beg:
            segment_texts.append("")
            continue
        result = asr(audio[beg:end], "zh", {}, True)
        segment_texts.append(result[0]["text"] if result else "")
    logger.info(f"vad segments: {segments_ms}")
    logger.info(f"vad segmentation elapsed: {(time.time() - start_time):.2f} seconds, "
                f"{len(segments_ms)} segment(s)")
    return segments_ms, segment_texts


# ---------------------------------------------------------------------------
# webm 解码：必须走 ffmpeg 命令行，不能用 torchaudio
#
# 实测（容器内）：torchaudio 2.3.0+cpu 的 wheel **没有编译 ffmpeg 后端** ——
#     torchaudio.list_audio_backends() == ['soundfile']
#     torchaudio.load(..., backend='ffmpeg') → ValueError: Unsupported backend 'ffmpeg'
# 而 libsndfile 不认识 webm 容器，于是原来那句 torchaudio.load(BytesIO(webm))
# 必然抛 LibsndfileError: Format not recognised（HTTP 返回 code=1）。
# 所以这里用 ffmpeg 命令行把容器格式转成 16bit PCM wav，再交给 soundfile 读。
# 这也是镜像里必须装 ffmpeg 的唯一原因（代码中没有其它地方调用它）。
# ---------------------------------------------------------------------------
def decode_with_ffmpeg(data):
    """把任意容器格式（webm / m4a / ...）转成 16bit PCM wav 后读出。

    @param data 原始文件字节
    @return (int16 ndarray, sample_rate)
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        # 把 ffmpeg 的 stderr 尾部带回给调用方，便于定位「不是有效音频」还是「缺解码器」
        detail = proc.stderr.decode("utf-8", "replace").strip() or "ffmpeg produced no output"
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {detail[-300:]}")
    return sf.read(io.BytesIO(proc.stdout), dtype=np.int16)


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
            info=message,
            data=data
        ).model_dump()
    )


class SentenceSegment(BaseModel):
    """一句完整的话及其相对音频开始时刻的时间区间。"""
    start_ms: int = Field(..., description="句子起始时刻，相对音频起点，单位毫秒")
    end_ms: int = Field(..., description="句子结束时刻，相对音频起点，单位毫秒")
    text: str = Field(..., description="句子文本（已去除 SenseVoice 的控制标记）")


class TranscriptionResponse(BaseModel):
    code: int
    info: str
    data: str
    sentences: List[SentenceSegment] = Field(
        default_factory=list,
        description="按句切分的结构化结果，时间相对音频起点；data 仍为纯文本，保持兼容"
    )


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
            # torchaudio 2.3.0+cpu 没有 ffmpeg 后端，webm 只能交给 ffmpeg 命令行解（见 decode_with_ffmpeg）
            input_wav, sr = decode_with_ffmpeg(file_content)
            is16 = True
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

        request_start = time.time()

        # 1) VAD 分段 + 逐段转写：为新增的 sentences 字段提供时间信息
        #    （SenseVoice 不输出时间戳，时间只能取自 VAD 片段边界）
        segments_ms, segment_texts = await asyncio.to_thread(transcribe_by_vad_segments, input_wav)
        sentences = build_sentences(segments_ms, segment_texts)
        logger.info(f"[DEBUG] sentences: {sentences}")

        # 2) data 字段：默认仍做一次「整段一次性转写」，与旧版本输出完全一致
        if config.data_from_full_asr:
            resp, elapsed_time = await generate_text()
            logger.info(f"[DEBUG] Transcribe raw response is {resp}")
            text = resp[0]["text"]
            logger.info(f'[DEBUG] res:{resp} text:{text}')
        else:
            text = "".join(clean_text(segment_text) for segment_text in segment_texts)
            elapsed_time = time.time() - request_start
            logger.info(f'[DEBUG] data built from vad segments: {text}')

        response = TranscriptionResponse(
            code=0,
            info=f"success, transcription time: {elapsed_time:.2f} seconds",
            data=text,
            sentences=sentences
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

        async def emit_vad_segment(abs_beg, abs_end):
            """对一个已闭合的 VAD 片段逐段转写，并按句附带时间区间外发。

            时间直接取自 VAD 片段的 beg/end 边界（相对音频流起点，单位毫秒）。
            @param abs_beg 片段起始，相对音频流起点
            @param abs_end 片段结束，相对音频流起点
            """
            nonlocal audio_vad, offset, hit

            rel_beg = max(int((abs_beg - offset) * config.sample_rate / 1000), 0)
            rel_end = int((abs_end - offset) * config.sample_rate / 1000)
            beg = min(rel_beg, len(audio_vad))
            end = min(max(rel_end, beg), len(audio_vad))
            logger.info(f"[vad segment] stream=[{abs_beg},{abs_end}]ms audio_len: {end - beg}")

            segment_audio = audio_vad[beg:end]
            audio_vad = audio_vad[end:]
            # offset 始终指向 audio_vad 缓冲区起点的绝对毫秒位置（与实际裁掉的样本严格一致）
            offset += int(round(end * 1000 / config.sample_rate))
            if end <= beg:
                logger.warning(f"[vad segment] empty slice skipped: stream=[{abs_beg},{abs_end}]ms")
                hit = False
                return

            result = None if not hit and len(reg_spks) != 0 else asr(segment_audio, lang.strip(),
                                                                     cache_asr, True)
            logger.info(f"asr response: {result}")

            hit = False

            if result is not None:
                # 片段内若有多句，句内边界按字符比例在片段内插值，首尾锚定 VAD 边界
                sentences = build_sentences([[abs_beg, abs_end]], [result[0]['text']])
                response = TranscriptionResponse(
                    code=0,
                    info=json.dumps(result[0], ensure_ascii=False),
                    data=result[0]['text'],
                    sentences=sentences
                )
                await websocket.send_json(response.model_dump())

        async def flush_pending_audio():
            """把缓冲区里最后一段语音补吐出来。

            流式 VAD 不发 `is_final=True` 时，流尾片段的结束事件不会出现，
            最后一句话会被丢掉（已在容器内实测复现）。
            """
            nonlocal last_vad_beg, last_vad_end

            res = vad_stream(np.zeros(0, dtype=np.float32), cache, is_final=True)
            for segment in (res[0]["value"] if res and len(res[0]["value"]) else []):
                if segment[0] > -1:
                    last_vad_beg = segment[0]
                if segment[1] > -1:
                    last_vad_end = segment[1]

            buffer_ms = int(len(audio_vad) * 1000 / config.sample_rate)
            if last_vad_beg > -1 and last_vad_end <= -1:
                # 只收到开始事件：用缓冲区末端补齐结束时间
                last_vad_end = offset + buffer_ms
            if last_vad_beg <= -1 and len(audio_vad) >= chunk_size:
                # VAD 始终没吐出片段但仍有未转写音频：整段兜底，避免尾部音频被丢弃
                last_vad_beg = offset
                last_vad_end = offset + buffer_ms

            if last_vad_beg > -1 and last_vad_end > last_vad_beg:
                abs_beg, abs_end = last_vad_beg, last_vad_end
                last_vad_beg = last_vad_end = -1
                await emit_vad_segment(abs_beg, abs_end)

        buffer = b""
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            data = message.get("bytes")
            if not data:
                # 文本控制帧（可选）：客户端停止录音时发送 flush，换取尾部片段
                command = (message.get("text") or "").strip().lower()
                if command in ("flush", "end", "done", "finish"):
                    await flush_pending_audio()
                    await websocket.send_json(
                        TranscriptionResponse(code=3, info="flushed", data="").model_dump()
                    )
                else:
                    logger.info(f"[ws] ignored control frame: {command!r}")
                continue

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

                res = vad_stream(chunk, cache, is_final=False)
                for segment in (res[0]["value"] if res and len(res[0]["value"]) else []):
                    # 流式 VAD 输出的是【边界事件】：[beg,-1] 为语音开始、[-1,end] 为语音结束，
                    # 坐标都是相对整个音频流起点的毫秒，配对后即可得到片段区间
                    if segment[0] > -1:
                        # 开始事件：开启一个新片段，同时清掉可能残留的结束时间
                        last_vad_beg = segment[0]
                        last_vad_end = -1
                    if segment[1] > -1 and last_vad_beg > -1:
                        # 结束事件：闭合当前片段；没有开始事件可配对的结束事件直接丢弃
                        last_vad_end = segment[1]
                    if last_vad_beg > -1 and last_vad_end > -1:
                        abs_beg, abs_end = last_vad_beg, last_vad_end
                        last_vad_beg = last_vad_end = -1
                        await emit_vad_segment(abs_beg, abs_end)
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
