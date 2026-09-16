#!/usr/bin/env python3
"""
测试 B：VAD 分段 + 逐段 ASR → `[{start_ms, end_ms, text}]` 时序转录结果。

流程（与未来功能一致，不改 server.py）：
1. 全文件 VAD → 每个语音片段 [beg_ms, end_ms]（相对音频起点）；
2. 对每个片段单独跑 SenseVoice（远程代码 model.py，同 server 用法）→ 片段文本；
3. 输出带时间的片段转录列表，保存 out/result_segments.json；
4. 三重校验：
   a) 片段边界 vs 真值（tmp/out/mixed_ground_truth.json，容差 600ms）；
   b) 片段文本 vs 真值中源语音区文本（同段语音应转出近似相同文本）；
   c) 片段拼接 LCS 重叠率 vs 整段一次转写（连续性）。
5. 演示：片段内按标点（。！？；）断句，时间按字符比例分配 → 句子级时间（近似）。
"""
import json
import os
import re
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
os.environ["MODELSCOPE_CACHE"] = os.path.join(os.getcwd(), "models")
os.environ["FUNASR_CACHE"] = os.path.join(os.getcwd(), "models")
os.environ["HOME"] = os.path.join(os.getcwd(), ".tools", "hubhome")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from funasr import AutoModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VAD_PATH = os.path.join(os.getcwd(), "models", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
ASR_PATH = os.path.join(os.getcwd(), "models", "SenseVoiceSmall")
WAV = os.path.join(HERE, "out", "mixed.wav")
GT = os.path.join(HERE, "out", "mixed_ground_truth.json")
SR = 16000
TOL_MS = 600

TAG_RE = re.compile(r"<\|[^|]*\|>")
SENT_END_RE = re.compile(r"[。！？；!?;]+")

def clean(text):
    return TAG_RE.sub("", text).strip()

def split_sentences(text):
    parts, ends = SENT_END_RE.split(text), SENT_END_RE.findall(text)
    out = []
    for i, p in enumerate(parts):
        s = (p + ends[i] if i < len(ends) else p).strip()
        if s:
            out.append(s)
    return out

def lcs2(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]

def main():
    with open(GT, encoding="utf-8") as f:
        gt = json.load(f)
    expected = gt["expected_segments_ms"]
    src_texts = [v["text"] for v in gt["verified_slices"]]
    audio, sr = sf.read(WAV, dtype="float32")
    assert sr == SR

    print("[test] loading VAD ...")
    vad = AutoModel(model=VAD_PATH, model_revision="v2.0.4", device="cpu",
                    disable_pbar=True, disable_update=True, local_files_only=True,
                    max_end_silence_time=500, speech_noise_thres=0.6)
    print("[test] loading ASR (SenseVoiceSmall, remote_code=./model.py) ...")
    asr_model = AutoModel(model=ASR_PATH, trust_remote_code=True, remote_code="./model.py",
                          device="cpu", disable_update=True, local_files_only=True)

    # 1+2) VAD 分段 + 逐段 ASR
    res = vad.generate(input=WAV, batch_size_s=60)
    segments_ms = [list(map(int, s)) for s in res[0]["value"]]
    print(f"[test] VAD segments (ms): {segments_ms}")

    seg_results = []
    for b_ms, e_ms in segments_ms:
        b, e = int(b_ms * SR / 1000), int(e_ms * SR / 1000)
        r = asr_model.generate(input=audio[b:e], cache={}, language="zh", use_itn=True, batch_size_s=60)
        seg_results.append({
            "start_ms": b_ms, "end_ms": e_ms, "duration_ms": e_ms - b_ms,
            "raw_text": r[0]["text"], "text": clean(r[0]["text"]),
            "avg_logprob": round(r[0].get("avg_logprob", 0.0), 4),
        })
        print(f"  [{b_ms:>7},{e_ms:>7}]ms  {seg_results[-1]['text']!r}")

    # 3) 校验
    # a) 边界 vs 真值
    ok_bound = []
    for idx, seg in enumerate(seg_results):
        j, best = None, None
        for k, (gb, ge) in enumerate(expected):
            d = abs(seg["start_ms"] - gb) + abs(seg["end_ms"] - ge)
            if d <= 2 * TOL_MS and (best is None or d < best[0]):
                best, j = (d, k), k
        ok_bound.append(j == idx)
    # b) 文本 vs 源语音区文本（同序比较；允许模型小差异 → 用去标点 LCS 重叠率）
    same_ratio = []
    for idx, seg in enumerate(seg_results):
        if idx < len(src_texts):
            a, b = re.sub(r"\s+", "", seg["text"]), re.sub(r"\s+", "", src_texts[idx])
            same_ratio.append(lcs2(a, b) / max(len(b), 1))
    # c) 拼接连续性 vs 整段一次转写
    r_all = asr_model.generate(input=audio, cache={}, language="zh", use_itn=True, batch_size_s=60)
    whole_text = clean(r_all[0]["text"])
    joined = "".join(s["text"] for s in seg_results)
    contiguity = lcs2(re.sub(r"\s+", "", joined), re.sub(r"\s+", "", whole_text)) / max(
        len(re.sub(r"\s+", "", whole_text)), 1)

    print(f"\n[test] 校验 a) 片段边界顺序匹配真值: {ok_bound}  -> {'✅' if all(ok_bound) and len(seg_results)==len(expected) else '❌'}")
    print(f"[test] 校验 b) 片段文本与源语音区文本重叠率: {[round(x,2) for x in same_ratio]}  -> {'✅' if all(r >= 0.5 for r in same_ratio) else '⚠ 需人工确认'}")
    print(f"[test] 校验 c) 片段拼接 vs 整段 LCS 重叠率: {contiguity:.1%}  -> {'✅ >=60%' if contiguity >= 0.6 else '⚠ <60%'}")
    print(f"[test] 整段一次转写({len(re.sub(r'\\s+','',whole_text))}字): {whole_text[:80]}...")

    # 4) 句子级时间（片段内按标点比例分配，近似）
    sentence_results = []
    for seg in seg_results:
        sents = split_sentences(seg["text"])
        n_chars = sum(len(s) for s in sents) or 1
        t = seg["start_ms"]
        for s in sents:
            span = int(seg["duration_ms"] * len(s) / n_chars)
            sentence_results.append({"start_ms": t, "end_ms": t + span, "text": s})
            t += span
    print("\n[test] 句子级结果（近似时间，相对音频起点）:")
    for s in sentence_results:
        print(f"  [{s['start_ms']:>7},{s['end_ms']:>7}]ms  {s['text']!r}")

    out = {
        "wav": WAV, "segment_results": seg_results, "sentence_results": sentence_results,
        "checks": {
            "boundary_ok": all(ok_bound) and len(seg_results) == len(expected),
            "text_similarity": same_ratio,
            "contiguity_ratio": round(contiguity, 4),
        },
        "whole_text": whole_text,
    }
    out_path = os.path.join(HERE, "out", "result_segments.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[test] saved -> {out_path}")

    ok = out["checks"]["boundary_ok"] and out["checks"]["contiguity_ratio"] >= 0.6
    print(f"\n=== 结论: {'✅ 可实现：VAD 分段 + 逐段 ASR → [start_ms, end_ms, text]' if ok else '❌ 未通过，见校验细节'} ===")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())