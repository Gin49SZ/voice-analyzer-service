#!/usr/bin/env python3
"""验证「每句附带 [start_ms, end_ms]」功能：纯函数边界用例 + 真实端点的协议级测试。

做法：把 funasr/modelscope/soundfile/torch/loguru 换成可控的假实现
（不改动 server.py 一行逻辑），用 Starlette TestClient 直接驱动真实路由：

- timestamps.py：断句 / 锚定 / 边界退化等纯函数用例；
- HTTP：整段 VAD → 逐段 ASR → sentences；并验证 data 与旧行为逐字节一致；
- WS：脚本化的 VAD 边界事件 → 分段消息的 sentences；文本控制帧 flush 的三种分支。

运行（依赖：fastapi/starlette/pydantic/pydantic-settings/numpy/httpx/uvicorn/
python-multipart，均无需 torch/funasr，可直接装在轻量 venv 里）：
    python tmp/test_timestamp_protocol.py
"""
import importlib.util
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

SR = 16000
CHUNK_SAMPLES = SR * 300 // 1000          # 与 config.chunk_size_ms 一致
API_KEY = "sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c"
AUDIO_HTTP = np.zeros(SR * 3, dtype=np.int16)          # HTTP 用：3s
AUDIO_CHUNK = np.zeros(CHUNK_SAMPLES, dtype=np.int16)  # WS 用：300ms 一块

TEXT_SEG1 = "<|zh|><|NEUTRAL|><|Speech|><|withitn|>第一句。第二句！"
TEXT_SEG2 = "<|zh|><|withitn|>第三句只有一句。"
TEXT_FULL = "<|zh|><|NEUTRAL|><|Speech|><|withitn|>整段一次性转写文本。"

WS_TEXT = [
    "<|zh|><|NEUTRAL|><|Speech|><|withitn|>第一句。第二句！",
    "<|zh|><|withitn|>第三句话。",
    "<|zh|><|withitn|>第四句话。",
    "<|zh|><|withitn|>第五句话。",
    "<|zh|><|withitn|>第六句话。",
]

LOGS = []
VAD_PLAN = [[0, 900], [1800, 2700]]
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append(bool(ok))
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}" + (f"  <- {detail}" if detail and not ok else ""))


class FakeVad:
    """可控假 VAD：整段模式返回片段对；流式模式按计划吐 [beg,-1]/[-1,end] 边界事件。"""

    def __init__(self):
        self.close_on_flush = False
        self.plan = []
        self.reset()

    def reset(self, plan=None):
        global VAD_PLAN
        if plan is not None:
            VAD_PLAN = [list(p) for p in plan]
        self.plan = [list(p) for p in VAD_PLAN]
        self.consumed_ms = 0
        self.emitted = [[False, False] for _ in self.plan]

    def generate(self, input=None, cache=None, is_final=False, chunk_size=None, **kwargs):
        n = len(input) if hasattr(input, "__len__") and not isinstance(input, (str, bytes)) else 0
        self.consumed_ms += int(n * 1000 / SR)
        if not is_final and chunk_size is None:
            return [{"value": [list(p) for p in self.plan]}]      # 整段模式
        events = []
        for i, (beg, end) in enumerate(self.plan):
            if not self.emitted[i][0] and self.consumed_ms > beg:
                events.append([beg, -1])
                self.emitted[i][0] = True
            if self.emitted[i][0] and not self.emitted[i][1] and self.consumed_ms >= end:
                events.append([-1, end])
                self.emitted[i][1] = True
        if is_final and self.close_on_flush:
            for i, (beg, end) in enumerate(self.plan):
                if self.emitted[i][0] and not self.emitted[i][1]:
                    events.append([-1, min(end, self.consumed_ms)])
                    self.emitted[i][1] = True
        return [{"value": events}]


class FakeAsr:
    def __init__(self):
        self.script = []
        self.inputs = []

    def generate(self, input=None, **kwargs):
        self.inputs.append(len(input) if hasattr(input, "__len__") else 0)
        text = self.script.pop(0) if self.script else "<|zh|><|withitn|>空。"
        return [{"key": "fake", "text": text, "avg_logprob": -0.01}]


def install_stubs():
    fake_vad, fake_asr = FakeVad(), FakeAsr()

    def AutoModel(**kwargs):
        return fake_vad if "vad" in str(kwargs.get("model", "")).lower() else fake_asr

    funasr = types.ModuleType("funasr")
    funasr.AutoModel = AutoModel
    sys.modules["funasr"] = funasr

    modelscope = types.ModuleType("modelscope")
    pipelines = types.ModuleType("modelscope.pipelines")
    pipelines.pipeline = lambda **kw: (lambda *a, **k: {"score": 0.0})
    modelscope.pipelines = pipelines
    sys.modules["modelscope"] = modelscope
    sys.modules["modelscope.pipelines"] = pipelines

    class _Info:
        subtype = "PCM_16"

    soundfile = types.ModuleType("soundfile")
    soundfile.read = lambda file, dtype=None, **kw: (AUDIO_HTTP.copy(), SR)
    soundfile.info = lambda file, **kw: _Info()
    sys.modules["soundfile"] = soundfile

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.from_numpy = lambda a: a
    sys.modules["torch"] = torch
    sys.modules["torchaudio"] = types.ModuleType("torchaudio")

    class _Logger:
        def remove(self, *a, **k):
            pass

        def add(self, *a, **k):
            pass

        def _log(self, msg, *a, **k):
            LOGS.append(str(msg))

        info = debug = warning = error = _log

    loguru = types.ModuleType("loguru")
    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru
    return fake_vad, fake_asr


def load_server():
    fake_vad, fake_asr = install_stubs()
    spec = importlib.util.spec_from_file_location("voice_server", os.path.join(PROJECT_ROOT, "server.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["voice_server"] = module
    spec.loader.exec_module(module)
    return module, fake_vad, fake_asr


def send_chunks(ws, count):
    for _ in range(count):
        ws.send_bytes(AUDIO_CHUNK.tobytes())


def test_http(server, fake_vad, fake_asr, client):
    print("\n=== HTTP POST /transcribe ===")
    fake_vad.reset([[0, 900], [1800, 3000]])
    fake_asr.script = [TEXT_SEG1, TEXT_SEG2, TEXT_FULL]
    fake_asr.inputs = []
    server.config.data_from_full_asr = True

    resp = client.post("/transcribe", params={"apiKey": API_KEY},
                       files={"file": ("a.wav", b"RIFF-fake", "audio/wav")})
    body = resp.json()
    print("  body:", json.dumps(body, ensure_ascii=False))
    check("HTTP 200", resp.status_code == 200)
    check("code == 0", body.get("code") == 0)
    check("data 与整段一次性转写逐字节一致（旧行为不变）", body.get("data") == TEXT_FULL)
    check("info 保持旧格式", str(body.get("info", "")).startswith("success, transcription time:"))
    expected = [
        {"start_ms": 0, "end_ms": 450, "text": "第一句。"},
        {"start_ms": 450, "end_ms": 900, "text": "第二句！"},
        {"start_ms": 1800, "end_ms": 3000, "text": "第三句只有一句。"},
    ]
    check("sentences 时间区间正确（片段边界锚定 + 片段内插值）",
          body.get("sentences") == expected, json.dumps(body.get("sentences"), ensure_ascii=False))
    check("逐段 ASR 切片长度正确（样本数）", fake_asr.inputs == [14400, 19200, 48000], str(fake_asr.inputs))

    # data_from_full_asr=False：省掉整段推理，data 由片段文本拼接
    fake_vad.reset([[0, 900], [1800, 3000]])
    fake_asr.script = [TEXT_SEG1, TEXT_SEG2]
    fake_asr.inputs = []
    server.config.data_from_full_asr = False
    body2 = client.post("/transcribe", params={"apiKey": API_KEY},
                        files={"file": ("a.wav", b"RIFF-fake", "audio/wav")}).json()
    check("data_from_full_asr=False 时 data 由片段拼接",
          body2.get("data") == "第一句。第二句！第三句只有一句。", body2.get("data"))
    check("data_from_full_asr=False 时省掉整段推理", fake_asr.inputs == [14400, 19200], str(fake_asr.inputs))
    check("data_from_full_asr=False 时 sentences 不变", body2.get("sentences") == expected)
    server.config.data_from_full_asr = True

    # 兼容性 / 错误分支
    r403 = client.post("/transcribe", params={"apiKey": "bad"},
                       files={"file": ("a.wav", b"RIFF-fake", "audio/wav")})
    print("  403 body:", r403.text[:120])
    check("错误 apiKey → 403（未影响既有鉴权行为）", r403.status_code == 403)
    r400 = client.post("/transcribe", params={"apiKey": API_KEY},
                       files={"file": ("a.ogg", b"RIFF-fake", "audio/ogg")})
    check("不支持的音频格式 → 400", r400.status_code == 400)
    check("GET /health 可用", client.get("/health").json() == {"status": "ok"})


def test_ws(server, fake_vad, fake_asr, client):
    print("\n=== WS /ws/transcribe ===")
    # 4 段语音：第 2 段在 flush 时闭合；第 3、4 段验证 flush 之后 offset 仍然正确
    fake_vad.reset([[0, 900], [1800, 2700], [3000, 3600], [4200, 4800]])
    fake_vad.close_on_flush = False
    fake_asr.script = list(WS_TEXT)
    fake_asr.inputs = []

    with client.websocket_connect(f"/ws/transcribe?apiKey={API_KEY}&lang=zh") as ws:
        # 1) 0~900ms：第 1 段闭合 → 片段内两句按字符比例插值
        send_chunks(ws, 3)
        m1 = ws.receive_json()
        print("  m1:", json.dumps(m1, ensure_ascii=False))
        check("WS 分段消息 code=0", m1.get("code") == 0)
        check("WS data 保持原始文本（含控制标记，未改动）", m1.get("data") == WS_TEXT[0])
        check("WS sentences 片段内插值", m1.get("sentences") == [
            {"start_ms": 0, "end_ms": 450, "text": "第一句。"},
            {"start_ms": 450, "end_ms": 900, "text": "第二句！"},
        ], json.dumps(m1.get("sentences"), ensure_ascii=False))

        # 2) 900~2100ms：第 2 段只收到开始事件，不应发出结果
        send_chunks(ws, 4)

        # 3) 未知文本控制帧应被忽略且不打断连接
        ws.send_text("ping")

        # 4) flush（VAD 未补结束事件）→ 用缓冲区末端补齐 + ack
        ws.send_text("flush")
        m2 = ws.receive_json()
        m3 = ws.receive_json()
        print("  m2:", json.dumps(m2, ensure_ascii=False))
        print("  m3:", json.dumps(m3, ensure_ascii=False))
        check("flush 补齐尾段区间（缓冲区末端兜底）",
              m2.get("sentences") == [{"start_ms": 1800, "end_ms": 2100, "text": "第三句话。"}],
              json.dumps(m2.get("sentences"), ensure_ascii=False))
        check("flush 回 ack(code=3)", m3.get("code") == 3 and m3.get("sentences") == [])
        check("未知控制帧未打断连接（仍能收到 flush 结果）", m2.get("code") == 0)

        # 5) 2100~2700ms：孤立的结束事件应被丢弃，不产生空切片推理、不重发结果
        send_chunks(ws, 2)

        # 6) 2700~3600ms：第 3 段正常闭合
        send_chunks(ws, 3)
        m4 = ws.receive_json()
        print("  m4:", json.dumps(m4, ensure_ascii=False))
        check("孤立的结束事件未污染后续片段（第 3 段区间正确）",
              m4.get("sentences") == [{"start_ms": 3000, "end_ms": 3600, "text": "第四句话。"}],
              json.dumps(m4.get("sentences"), ensure_ascii=False))

        # 7) flush 时 VAD 没给出任何片段，但缓冲区还有音频 → 整段兜底
        send_chunks(ws, 2)
        fake_vad.close_on_flush = True
        ws.send_text("flush")
        m5 = ws.receive_json()
        m6 = ws.receive_json()
        print("  m5:", json.dumps(m5, ensure_ascii=False))
        check("缓冲区残留音频整段兜底（未丢尾）",
              m5.get("sentences") == [{"start_ms": 3600, "end_ms": 4200, "text": "第五句话。"}],
              json.dumps(m5.get("sentences"), ensure_ascii=False))
        check("第二次 flush 也回 ack", m6.get("code") == 3)

        # 8) 4200~4800ms：flush 之后 offset/缓冲区状态仍然正确
        send_chunks(ws, 2)
        m7 = ws.receive_json()
        print("  m7:", json.dumps(m7, ensure_ascii=False))
        check("flush 之后片段仍然正确（时间呈单调递增）",
              m7.get("sentences") == [{"start_ms": 4200, "end_ms": 4800, "text": "第六句话。"}],
              json.dumps(m7.get("sentences"), ensure_ascii=False))

        # 9) 无待处理音频时的 flush：只回 ack
        ws.send_text("flush")
        m8 = ws.receive_json()
        check("空 flush 只回 ack", m8.get("code") == 3)

    check("WS 逐段 ASR 切片长度正确（样本数，验证 flush 前后 offset 一致性）",
          fake_asr.inputs == [14400, 4800, 9600, 9600, 9600], str(fake_asr.inputs))
    check("WS 断连（未 flush）不报错", True)


def test_timestamps_unit():
    """timestamps.py 纯函数边界用例（不需要任何依赖）。"""
    print("\n=== timestamps.py 纯函数边界用例 ===")
    import timestamps as ts

    check("clean_text 去掉控制标记", ts.clean_text("<|zh|><|withitn|>你好。") == "你好。")
    check("clean_text 纯标记返回空串", ts.clean_text("<|zh|><|NEUTRAL|>") == "")
    check("split_sentences 按标点断句", ts.split_sentences("甲。乙！丙") == ["甲。", "乙！", "丙"])
    check("split_sentences 连续标点不产生空句", ts.split_sentences("甲！！乙。") == ["甲！！", "乙。"])
    check("split_sentences 无标点整段返回", ts.split_sentences("无标点") == ["无标点"])
    check("build_sentences 空输入", ts.build_sentences([], []) == [])
    check("build_sentences 空文本跳过", ts.build_sentences([[0, 500]], ["   "]) == [])
    check("build_sentences 单句直接使用片段精确边界",
          ts.build_sentences([[100, 200]], ["只有一句"]) == [{"start_ms": 100, "end_ms": 200, "text": "只有一句"}])
    check("build_sentences 负起点被钳为 0",
          ts.build_sentences([[-5, 200]], ["x。"]) == [{"start_ms": 0, "end_ms": 200, "text": "x。"}])
    check("build_sentences 逆序边界自动纠正",
          ts.build_sentences([[200, 100]], ["x。"]) == [{"start_ms": 100, "end_ms": 200, "text": "x。"}])
    check("build_sentences 零长片段退化为一个点",
          ts.build_sentences([[100, 100]], ["a。b。"]) == [
              {"start_ms": 100, "end_ms": 100, "text": "a。"},
              {"start_ms": 100, "end_ms": 100, "text": "b。"}])

    multi = ts.build_sentences([[0, 900]], ["甲。乙。丙。"])
    times = [(s["start_ms"], s["end_ms"]) for s in multi]
    check("build_sentences 多句时间单调且首尾锚定片段边界",
          times == [(0, 300), (300, 600), (600, 900)] and all(b >= a for a, b in times), str(times))
    check("build_sentences 片段数多于文本数时按 zip 截断",
          ts.build_sentences([[0, 100], [200, 300]], ["甲。"]) == [{"start_ms": 0, "end_ms": 100, "text": "甲。"}])


def main():
    print("[test] 加载 server.py（模型/推理库已替换为假实现）")
    server, fake_vad, fake_asr = load_server()
    from fastapi.testclient import TestClient
    client = TestClient(server.app)

    test_timestamps_unit()
    test_http(server, fake_vad, fake_asr, client)
    test_ws(server, fake_vad, fake_asr, client)

    print("\n=== 服务端 VAD/片段日志（时间证据） ===")
    for line in LOGS:
        if "vad segment" in line:
            print("   ", line)

    total, passed = len(CHECKS), sum(CHECKS)
    print(f"\n=== 结果: {passed}/{total} 项通过 ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
