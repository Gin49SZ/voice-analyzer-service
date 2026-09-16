#!/usr/bin/env python3
"""
测试 D：funasr 内置 SenseVoice 类是否能为当前模型产出词级时间戳。

关键差异（来自调研）：
- 项目 server.py 用 `trust_remote_code=True, remote_code=./model.py`（仓库自带实现），
  其 inference() 不处理 output_timestamp → 无时间戳；
- funasr 内置类（trust_remote_code=False，即模型目录的 funasr/models/sense_voice/model.py）
  支持 generate(..., output_timestamp=True)，输出 timestamp + words（CTC 强制对齐）。

本脚本将本地 models/SenseVoiceSmall 交给 funasr 内置类加载并开启 output_timestamp，
观察能否得到词级 [start,end]。若加载失败，则打印具体原因（也是结论的一部分）。

注意：加载 SenseVoice 约需 2GB 内存；请确保不与其他重进程并发。
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
os.environ["MODELSCOPE_CACHE"] = os.path.join(os.getcwd(), "models")
os.environ["FUNASR_CACHE"] = os.path.join(os.getcwd(), "models")
os.environ["HOME"] = os.path.join(os.getcwd(), ".tools", "hubhome")

import soundfile as sf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ASR_PATH = os.path.join(os.getcwd(), "models", "SenseVoiceSmall")
WAV = os.path.join(os.getcwd(), "examples", "test_speaker.wav")   # 4.93s，片段短、跑得快

import time  # noqa: E402

def main():
    from funasr import AutoModel

    print("[test] loading SenseVoiceSmall with FUNASR BUILT-IN model class (no remote_code) ...")
    t0 = time.time()
    try:
        model = AutoModel(
            model=ASR_PATH,
            device="cpu",
            disable_pbar=True,
            disable_update=True,
            local_files_only=True,
            # 注意：不传 trust_remote_code / remote_code → 使用 funasr 内置 sense_voice 类
        )
    except Exception as e:
        print(f"[test] ❌ 内置类加载失败: {type(e).__name__}: {e}")
        print("[test] 结论：当前模型目录无法直接使用 funasr 内置类（需检查 config.yaml 的 model_type 或换模型名加载）")
        return 1
    print(f"[test] loaded in {time.time()-t0:.1f}s")

    # 先跑一次不带时间戳，确认基线输出字段
    r0 = model.generate(input=WAV, cache={}, language="zh", use_itn=True, batch_size_s=60)
    print(f"[test] 基线输出 keys: {list(r0[0].keys())}")

    # 开启 output_timestamp
    r1 = model.generate(
        input=WAV, cache={}, language="zh", use_itn=True,
        batch_size_s=60, output_timestamp=True,
    )
    out = r1[0]
    print(f"[test] output_timestamp=True 输出 keys: {list(out.keys())}")
    if "timestamp" not in out:
        print("[test] ❌ 没有 timestamp 字段")
        return 1

    ts = out["timestamp"]
    words = out.get("words", [])
    text = out.get("text", "")
    print(f"[test] text: {text!r}")
    print(f"[test] words 数量: {len(words)}, timestamp 数量: {len(ts)}")
    for i, (w, t) in enumerate(zip(words, ts)):
        print(f"    [{t[0]:.3f}s, {t[1]:.3f}s]  {w}")
        if i >= 24:
            print("    ...")
            break

    # 校验：timestamp 起止均在音频时长内
    info = sf.info(WAV)
    dur_s = info.frames / info.samplerate
    ok = all(0 <= t[0] <= t[1] <= dur_s + 0.05 for t in ts) and len(ts) > 0
    print(f"\n[test] 音频时长 {dur_s:.2f}s; 时间戳范围校验: {'✅ 合法' if ok else '❌ 越界'}")
    print(f"\n=== 结论: {'✅ 可实现（funasr 内置类 + output_timestamp 词级时间戳）' if ok else '❌ 时间戳异常'} ===")
    if ok:
        print("    注意：这是 CTC 对齐的近似词级时间（60ms 帧精度），相对输入片段起点；")
        print("    与当前 server 的 remote_code 实现（无时间戳）不同，需切换加载方式才能使用。")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())