#!/usr/bin/env python3
"""
生成带已知时间边界的测试音频 out/mixed.wav 及真值 out/mixed_ground_truth.json。

真值来源（关键设计）：
- 直接在 examples/test.wav（电影对白，含停顿/背景）上运行全文件 VAD 得到语音区；
- 每个语音区再经两道**独立校验**后才算"已知语音区"：
  1) 能量校验：区内 50ms 帧 RMS 均值不低于阈值（排除低能量杂音）；
  2) 转写校验：区内用 SenseVoice 转写得到非空文本（排除纯音乐/噪声段）；
- 选取 3 个时长 1.5~4s 的语音区，剪出后以 600ms 静音拼接成新文件；
  各语音区在新文件中的起止即真值（毫秒，相对新文件起点）。

说明：真值边界源自 VAD（源文件），但经能量+文本独立验证；测试目标不是"VAD 绝对精度"，
而是"对新文件重现这些语音区边界 + 每区得到可读文本"的端到端可行性。
"""
import json
import os

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "out")
SRC_WAV = os.path.join(PROJECT_ROOT, "examples", "test.wav")
OUT_WAV = os.path.join(OUT_DIR, "mixed.wav")
OUT_GT = os.path.join(OUT_DIR, "mixed_ground_truth.json")
SR = 16000
SILENCE_MS = 600
ENERGY_THR = 0.015       # 50ms 帧 RMS 阈值：低于视为非语音背景
MIN_DUR_MS, MAX_DUR_MS = 1500, 4000
N_PICK = 3

def rms_envelope(audio, frame_ms=50):
    frame = int(frame_ms * SR / 1000)
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0)
    return np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))

def main():
    os.chdir(PROJECT_ROOT)
    os.environ["MODELSCOPE_CACHE"] = os.path.join(PROJECT_ROOT, "models")
    os.environ["FUNASR_CACHE"] = os.path.join(PROJECT_ROOT, "models")
    os.environ["HOME"] = os.path.join(PROJECT_ROOT, ".tools", "hubhome")
    from funasr import AutoModel

    audio, sr = sf.read(SRC_WAV, dtype="float32")
    assert sr == SR
    print(f"[gen] 源 {SRC_WAV}: {len(audio)/SR:.2f}s")

    vad = AutoModel(model=os.path.join(PROJECT_ROOT, "models", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                    model_revision="v2.0.4", device="cpu", disable_pbar=True, disable_update=True,
                    local_files_only=True, max_end_silence_time=500, speech_noise_thres=0.6)
    res = vad.generate(input=SRC_WAV, batch_size_s=60)
    segs = [list(map(int, s)) for s in res[0]["value"]]
    print(f"[gen] 源文件 VAD 语音区: {segs}")

    # 独立校验各语音区的能量
    rms = rms_envelope(audio)
    verified = []
    for b, e in segs:
        dur = e - b
        if not (MIN_DUR_MS <= dur <= MAX_DUR_MS):
            continue
        f0, f1 = int(b / 50), int(e / 50)
        r = float(rms[f0:f1].mean()) if f1 > f0 else 0.0
        if r >= ENERGY_THR:
            verified.append({"seg_ms": [b, e], "mean_rms": round(r, 4)})
    print(f"[gen] 能量通过: {[(v['seg_ms'], v['mean_rms']) for v in verified]}")

    # 转写校验：取前 N_PICK 个，ASR 出非空文本
    asr_model = AutoModel(model=os.path.join(PROJECT_ROOT, "models", "SenseVoiceSmall"),
                          trust_remote_code=True, remote_code="./model.py", device="cpu",
                          disable_update=True, local_files_only=True)
    import re
    TAG = re.compile(r"<\|[^|]*\|>")
    picked = []
    for v in verified[: N_PICK * 2]:
        b, e = v["seg_ms"]
        seg_audio = audio[int(b * SR / 1000): int(e * SR / 1000)]
        rr = asr_model.generate(input=seg_audio, cache={}, language="zh", use_itn=True, batch_size_s=60)
        text = TAG.sub("", rr[0]["text"]).strip()
        if len(text) >= 4:
            v["text"] = text
            picked.append(v)
            print(f"[gen] 语音区 {[b, e]}ms 转写: {text!r}")
        if len(picked) >= N_PICK:
            break
    if len(picked) < N_PICK:
        raise RuntimeError(f"只找到 {len(picked)} 个合格语音区，需要 {N_PICK}")

    # 拼接：语音区 + 静音
    parts, expected, verified_slices = [], [], []
    t = 0
    for v in picked:
        b, e = v["seg_ms"]
        dur = e - b
        parts.append(audio[int(b * SR / 1000): int(e * SR / 1000)])
        expected.append([t, t + dur])
        verified_slices.append({"src": [b, e], "shifted": [t, t + dur], "mean_rms": v["mean_rms"], "text": v["text"]})
        t += dur
        if v is not picked[-1]:
            parts.append(np.zeros(int(SILENCE_MS * SR / 1000), dtype=np.float32))
            t += SILENCE_MS

    mixed = np.concatenate(parts)
    os.makedirs(OUT_DIR, exist_ok=True)
    sf.write(OUT_WAV, mixed, SR, subtype="PCM_16")
    gt = {
        "source_wav": SRC_WAV,
        "silence_ms": SILENCE_MS,
        "expected_segments_ms": expected,
        "verified_slices": verified_slices,
        "total_ms": int(len(mixed) / SR * 1000),
        "tolerance_ms": 600,
        "note": "真值源自源文件 VAD，并经 能量(RMS≥0.015) + ASR非空文本 双重校验",
    }
    with open(OUT_GT, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print(f"\n[gen] wrote {OUT_WAV} ({len(mixed)/SR:.2f}s)")
    print(f"[gen] expected segments (ms): {expected}")
    print(f"[gen] wrote {OUT_GT}")

if __name__ == "__main__":
    main()