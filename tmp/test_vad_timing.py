#!/usr/bin/env python3
"""
测试 A：VAD 片段时间 vs 真值（全文件模式 + 流式模式，含能量佐证）。

流式 VAD 语义（本次调试确认）：
- funasr 流式 VAD 输出的是【边界事件】：[beg, -1] 表示语音开始、[-1, end] 表示语音结束，
  坐标均为【相对整个流起点的毫秒】（0 = 第一块）；
- 完整片段 = 匹配的 (beg 事件, end 事件)；因此绝对时间戳可直接取原始 beg/end（负值钳为 0），
  不需要 server.py 里基于 offset 的缓冲区换算；
- 若不发送 is_final=True 的 flush，流尾最后一个片段的 end 事件不会出现（server.py 现状即如此）。

判定：
- VAD 片段数量与真值一致，且每段起止偏差 ≤ tolerance_ms；另附每段的能量均值佐证。
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
os.environ["MODELSCOPE_CACHE"] = os.path.join(os.getcwd(), "models")
os.environ["FUNASR_CACHE"] = os.path.join(os.getcwd(), "models")
os.environ["HOME"] = os.path.join(os.getcwd(), ".tools", "hubhome")

import soundfile as sf  # noqa: E402
from funasr import AutoModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VAD_PATH = os.path.join(os.getcwd(), "models", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
WAV = os.path.join(HERE, "out", "mixed.wav")
GT = os.path.join(HERE, "out", "mixed_ground_truth.json")
SR = 16000
CHUNK_MS = 300

def load_vad():
    return AutoModel(
        model=VAD_PATH, model_revision="v2.0.4", device="cpu",
        disable_pbar=True, disable_update=True, local_files_only=True,
        max_end_silence_time=500, speech_noise_thres=0.6,
    )

def vad_full_file(vad, wav):
    res = vad.generate(input=wav, batch_size_s=60)
    return [list(map(int, s)) for s in res[0]["value"]]

def vad_streaming(vad, audio, flush=False):
    """按流式 VAD 事件语义收集绝对时间片段（bg, end 事件配对）。"""
    chunk_size = CHUNK_MS * SR // 1000
    cache = {}
    segments = []
    pending_beg = None
    def feed(value):
        nonlocal pending_beg
        for beg, end in value:
            if beg > -1:
                pending_beg = beg  # 事件语义：开始事件
            if end > -1:
                b = max(pending_beg if pending_beg is not None else 0, 0)
                segments.append([b, int(end)])
                pending_beg = None
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        res = vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=CHUNK_MS)
        feed(res[0]["value"])
    if flush:
        res = vad.generate(input=np.zeros(0, dtype=np.float32), cache=cache, is_final=True, chunk_size=CHUNK_MS)
        feed(res[0]["value"])
    return segments

def match(segments, expected, tol):
    """贪心一一匹配，返回 (matched, details)。"""
    used, details, matched = set(), [], 0
    for b, e in segments:
        best = None
        for j, (gb, ge) in enumerate(expected):
            if j in used:
                continue
            db, de = b - gb, e - ge
            if abs(db) <= tol and abs(de) <= tol:
                score = abs(db) + abs(de)
                if best is None or score < best[0]:
                    best = (score, j, db, de)
        if best:
            used.add(best[1]); matched += 1
            details.append((b, e, best[1], best[2], best[3], "OK"))
        else:
            details.append((b, e, None, None, None, "NO-MATCH"))
    return matched, details

def report(name, segments, expected, tol):
    print(f"\n=== {name} ===")
    matched, details = match(segments, expected, tol)
    print(f"  segments: {segments}")
    print(f"  matched {matched}/{len(expected)} within ±{tol}ms")
    for b, e, j, db, de, st in details:
        g = f"[{expected[j][0]},{expected[j][1]}]" if j is not None else "-"
        print(f"    vad=[{b:>6},{e:>6}]ms  gt={g}  Δ({db}, {de})  {st}")
    return matched == len(expected) and len(segments) == len(expected)

def energy_of(audio, b_ms, e_ms):
    a = audio[int(b_ms * SR / 1000):int(e_ms * SR / 1000)]
    return float(np.sqrt((a ** 2).mean())) if len(a) else 0.0

def main():
    with open(GT, encoding="utf-8") as f:
        gt = json.load(f)
    expected = gt["expected_segments_ms"]
    tol = gt["tolerance_ms"]
    audio, sr = sf.read(WAV, dtype="float32")
    assert sr == SR

    print(f"[test] wav={WAV} total={len(audio)/SR:.2f}s; expected={expected}; tol=±{tol}ms")

    # 能量佐证：真值区与静音区 能量对比
    print("\n[test] 能量佐证（50ms 帧 RMS，真值语音区 vs 静音区）:")
    for b, e in expected:
        seg_rms = energy_of(audio, b, e)
        silence_rms = energy_of(audio, max(e, 0), min(e + 600, len(audio) / SR * 1000))
        print(f"    [{b:>6},{e:>6}]ms  语音区RMS={seg_rms:.4f}  后续静音RMS≈{silence_rms:.4f}  "
              f"{'✅ 区分明显' if seg_rms > 0.01 else '⚠ 偏低'}")

    vad = load_vad()
    print("[test] VAD loaded")

    seg_full = vad_full_file(vad, WAV)
    ok1 = report("VAD 全文件模式", seg_full, expected, tol)

    seg_st = vad_streaming(vad, audio, flush=False)
    ok2 = report("VAD 流式模式（事件语义，无 flush → 尾部片段缺失预期）", seg_st, expected, tol)

    seg_st_f = vad_streaming(vad, audio, flush=True)
    ok3 = report("VAD 流式模式（事件语义，+ is_final flush）", seg_st_f, expected, tol)

    print("\n=== 结论 ===")
    print(f"  全文件模式 : {'✅ 边界与真值一致' if ok1 else '❌ 不一致'}")
    print(f"  流式无flush: {'✅' if ok2 else '⚠ 尾部片段缺失（server.py 现状如此，实现时需补 is_final）'}")
    print(f"  流式+flush : {'✅ 边界与真值一致' if ok3 else '❌ 不一致'}")
    print("  VAD 输出 [beg,end] 为相对音频起点的毫秒 → 逐段转写即可得到时序转录结果。")
    return 0 if ok3 else 1

if __name__ == "__main__":
    sys.exit(main())