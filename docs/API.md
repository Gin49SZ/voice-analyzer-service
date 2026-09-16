# voice-analyzer-service API 文档

基于 FunASR（SenseVoiceSmall）+ FSMN-VAD + ERes2Net 说话人验证的音频转录服务。支持**音频文件上传转录**与 **WebSocket 流式转录**，两条链路都会为每句完整的话附带相对音频开始时刻的时间区间 `[start_ms, end_ms]`。

> 本文档对应 `server.py` 当前实现（含 `sentences` 时间区间功能）。
> 文中所有响应报文、状态码、关闭码均来自对真实端点的实测（验证脚本见 `tmp/test_timestamp_protocol.py`）。

---

## 目录

- [1. 快速索引](#1-快速索引)
- [2. 通用约定](#2-通用约定)
  - [2.1 鉴权](#21-鉴权)
  - [2.2 响应包装与 code 码表](#22-响应包装与-code-码表)
  - [2.3 错误响应的两种形态](#23-错误响应的两种形态)
- [3. 数据模型](#3-数据模型)
- [4. `GET /health`](#4-get-health)
- [5. `POST /transcribe`](#5-post-transcribe)
- [6. `WS /ws/transcribe`](#6-ws-ws-transcribe)
- [7. 句子级时间戳语义](#7-句子级时间戳语义)
- [8. 配置项](#8-配置项)
- [9. 兼容性与迁移](#9-兼容性与迁移)
- [10. 常见问题](#10-常见问题)
- [11. 变更记录](#11-变更记录)

---

## 1. 快速索引

| 方法 | 路径 | 说明 | 鉴权位置 | 是否含 `sentences` |
|---|---|---|---|---|
| `GET` | `/health` | 健康检查 | 无 | — |
| `POST` | `/transcribe` | 上传音频文件，整段转录 | Query `apiKey` | ✅ |
| `WS` | `/ws/transcribe` | 流式音频，边录边出结果 | Query `apiKey` | ✅ |

**默认端口**：`27000`（启动参数 `--port` 可改）

```bash
python server.py --port 27000            # CPU
python server.py --port 27000 --gpu true # GPU
```

> 路由挂载在根路径。若服务经反向代理发布，实际基准地址以前缀为准（示例：`https://<host>/<prefix>`）。

---

## 2. 通用约定

### 2.1 鉴权

- **位置**：URL Query 参数 `apiKey`（HTTP 与 WS 一致，**均不放在 Header**）。
- **取值**：默认 `sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c`，由 `Config.api_key` 配置。
- **失败表现**：

| 通道 | 失败响应 |
|---|---|
| HTTP | `403` + `{"detail": "Invalid or missing API key. Please provide apiKey parameter."}` |
| WS | 连接被拒绝，**关闭码 `1008`**，reason `Invalid or missing API key` |

> `apiKey` 缺失与错误**返回完全相同的响应**，不区分。

### 2.2 响应包装与 code 码表

`POST /transcribe` 与 `WS /ws/transcribe` 的**业务成功/失败消息**统一使用如下四字段结构：

```json
{
  "code": 0,
  "info": "success, transcription time: 1.23 seconds",
  "data": "<|zh|><|withitn|>军事动作类型片里。",
  "sentences": [
    {"start_ms": 0, "end_ms": 1970, "text": "军事动作类型片里。"}
  ]
}
```

`code` 码表：

| `code` | 出现位置 | 含义 | `data` 内容 | `sentences` |
|---|---|---|---|---|
| `0` | HTTP / WS | 转写成功 | HTTP：整段纯文本<br>WS：该片段原始文本 | 句子时间区间 |
| `1` | HTTP | 服务端处理异常（未捕获异常） | `""` | `[]` |
| `2` | WS | 说话人验证命中 | 命中的注册说话人音频 URL | `[]` |
| `3` | WS | `flush` 收尾完成确认 | `""` | `[]` |

> `code=1` 时 **HTTP 状态码仍为 `200`**，业务错误需读 `code` 字段判断，`info` 为异常信息字符串。

### 2.3 错误响应的两种形态

这是本服务容易踩坑的一点：**参数校验类错误不走业务包装**。

| 场景 | HTTP 状态码 | 响应体 |
|---|---|---|
| 缺少 / 错误 `apiKey` | `403` | `{"detail": "Invalid or missing API key. Please provide apiKey parameter."}` |
| 不支持的音频格式 | `400` | `{"detail": "Unsupported audio format"}` |
| 缺少必填的 `file` 字段 | `422` | `{"detail": [{"type":"missing","loc":["body","file"],"msg":"Field required","input":null}]}` |
| 推理/IO 等未捕获异常 | `200` | `{"code": 1, "info": "<异常信息>", "data": "", "sentences": []}` |

**判断建议**：先判 HTTP 状态码；`2xx` 时再判 `code == 0`。若 `2xx` 且含 `detail` 字段缺失、`code != 0`，读取 `info` 作为错误描述。

---

## 3. 数据模型

### 3.1 `TranscriptionResponse`

| 字段 | 类型 | 必有 | 说明 |
|---|---|---|---|
| `code` | `int` | ✅ | 业务码，见 [2.2 码表](#22-响应包装与-code-码表) |
| `info` | `str` | ✅ | 成功时为 `"success, transcription time: {秒:.2f} seconds"`；失败时为错误描述 |
| `data` | `str` | ✅ | **纯文本主结果**，字段语义与历史版本完全一致 |
| `sentences` | `SentenceSegment[]` | ✅ | 结构化句子级结果；无内容时为空数组 `[]` |

### 3.2 `SentenceSegment`

| 字段 | 类型 | 说明 |
|---|---|---|
| `start_ms` | `int` | 句首时刻，相对**音频/流起点**，单位毫秒，`>= 0` |
| `end_ms` | `int` | 句尾时刻，相对**音频/流起点**，单位毫秒，`>= start_ms` |
| `text` | `str` | 句子文本，**已去除** SenseVoice 控制标记（如 `<\|zh\|>`、`<\|NEUTRAL\|>`、`<\|withitn\|>`） |

**注意**：

- 区间语义为**闭区间** `[start_ms, end_ms]`，恒有 `start_ms <= end_ms`。
- `sentences` 按时间**升序**排列。
- `data` 保留原始文本（WS 链路**含**控制标记，与旧版逐字节一致）；`sentences[].text` 是清洗后的文本。二者**不要互相替代**。
- 相邻句子之间通常**存在时间间隙**（VAD 片段间的静音），因此 `sentences` 在时间轴上一般**不连续**，请勿假定 `sentences[i].end_ms == sentences[i+1].start_ms`。

---

## 4. `GET /health`

健康检查，无鉴权。

**响应** `200`：

```json
{"status": "ok"}
```

---

## 5. `POST /transcribe`

上传音频文件，返回整段转录文本 + 句子级时间区间。

### 5.1 请求

**Query 参数**

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `apiKey` | `string` | ✅ | API 密钥 |

**Body（`multipart/form-data`）**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | `file` | ✅ | 音频文件 |

**支持的格式**（依据请求的 `Content-Type` 分派）

| `Content-Type` 前缀 | 解码方式 | 说明 |
|---|---|---|
| `audio/wav` | `soundfile` | WAV / PCM；按 `PCM_16` 子类型判定是否需要 int16 归一化 |
| `audio/webm` | `torchaudio` | WebM / Opus 等 |

其他类型（如 `audio/mpeg`、`application/octet-stream`）→ `400 Unsupported audio format`。

> ⚠️ 格式判定**依赖客户端声明的 `Content-Type`**，而非文件真实内容。上传 MP3 却声称 `audio/wav` 会在解码阶段抛异常，最终表现为 `200 / code=1`。

**音频预处理（服务端自动完成）**

1. 解码为 numpy 数组；
2. 若判定为 16-bit PCM，则按 `int16` 满量程归一化到 `[-1, 1]`；
3. 多声道取**声道均值**转为单声道；
4. 采样率非 `16000` 时，用 `torchaudio.transforms.Resample` 重采样到 16 kHz。

> 实践建议：**直接上传 16 kHz / 16-bit / 单声道 WAV**，可跳过重采样与格式转换，延迟最低、结果最稳。

### 5.2 响应

成功 `200`：

```json
{
  "code": 0,
  "info": "success, transcription time: 2.41 seconds",
  "data": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>军事动作类型片里。我有别人没有拥有过的经验。",
  "sentences": [
    {"start_ms": 0, "end_ms": 1970, "text": "军事动作类型片里。"},
    {"start_ms": 2570, "end_ms": 6160, "text": "我有别人没有拥有过的经验。"}
  ]
}
```

- `data`：**整段一次性转写**的原始文本（含控制标记），与历史版本逐字节一致；
- `sentences`：由 VAD 片段边界 + 逐段转写文本生成（见 [第 7 节](#7-句子级时间戳语义)）。

**无语音时**：`sentences` 为 `[]`，`data` 仍为整段转写结果（可能为空串，也可能是被模型识别出的少量文本）。

**耗时说明**：默认配置下服务端会**跑两次 ASR**——一次按 VAD 片段逐段转写（为 `sentences` 提供文本），一次整段转写（为 `data` 保兼容）。因此耗时约为旧版本的 2 倍。通过 `Config.data_from_full_asr = False` 可省掉整段推理（详见 [8. 配置项](#8-配置项)）。

### 5.3 调用示例

**curl**

```bash
curl -X POST "http://127.0.0.1:27000/transcribe?apiKey=sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c" \
  -F "file=@examples/test.wav;type=audio/wav"
```

**Python（requests）**

```python
import requests

resp = requests.post(
    "http://127.0.0.1:27000/transcribe",
    params={"apiKey": "sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c"},
    files={"file": ("test.wav", open("examples/test.wav", "rb"), "audio/wav")},
    timeout=120,
)
resp.raise_for_status()
body = resp.json()

if body.get("code") == 0:
    print("文本:", body["data"])
    for s in body["sentences"]:
        print(f'[{s["start_ms"]:>6} - {s["end_ms"]:>6} ms] {s["text"]}')
else:
    print("失败:", body)
```

**JavaScript（浏览器）**

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const resp = await fetch(
  `/transcribe?apiKey=${encodeURIComponent(API_KEY)}`,
  { method: "POST", body: formData }
);
const data = await resp.json();

// 逐句渲染，并支持按句回放
const audio = document.getElementById("player");
data.sentences.forEach((s) => {
  const btn = document.createElement("button");
  btn.textContent = `[${s.start_ms}-${s.end_ms}] ${s.text}`;
  btn.onclick = () => {
    audio.currentTime = s.start_ms / 1000;
    audio.play();
    setTimeout(() => audio.pause(), s.end_ms - s.start_ms);
  };
  document.body.appendChild(btn);
});
```

> 仓库内的 `test_client.html` 已实现上述「句子时间表 + 按句试听」。

---

## 6. `WS /ws/transcribe`

实时流式转录：客户端持续推流麦克风音频，服务端每检测到**一个完整语音片段**就回一条结果消息。

### 6.1 连接

**URL**

```
ws://<host>:27000/ws/transcribe?apiKey=<KEY>&lang=zh&reg_spks=<URL编码的音频URL列表>
```

**Query 参数**

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `apiKey` | `string` | ✅ | — | API 密钥；缺失或错误 → 关闭码 `1008` |
| `lang` | `string` | ❌ | `zh` | 识别语言，传入 ASR 引擎 |
| `reg_spks` | `string` | ❌ | `""` | 已注册说话人音频 URL，**逗号分隔**，整体需 URL 编码 |

> `reg_spks` 需要 `urllib.parse.quote` 编码，例如
> `reg_spks=https%3A%2F%2Fexample.com%2Fa.wav%2Chttps%3A%2F%2Fexample.com%2Fb.wav`

### 6.2 上行帧（客户端 → 服务端）

| 帧类型 | 内容 | 说明 |
|---|---|---|
| **二进制帧** | 原始 PCM 音频字节 | **16 kHz / 16-bit / 单声道 / 小端**，与 WAV 的 data 段一致（去掉 44 字节头）。建议每帧 300 ms |
| **文本帧** | `flush`（或 `end` / `done` / `finish`） | 可选控制帧，见 [6.5 收尾 flush](#65-收尾-flush) |

**音频帧处理细节**

- 服务端把收到的字节按 `int16` 解析（`/ 32767.0` 归一化），累积到内部缓冲区；
- 每累积满 `chunk_size_ms = 300 ms`（`16000 × 0.3 = 4800` 样本）就送一次流式 VAD；
- 若某帧字节数为**奇数**，末尾 1 字节会被暂存、与下一帧拼接，不会丢失；
- 缓冲区累积不足 2 字节时不解析，等待后续字节。

### 6.3 下行消息（服务端 → 客户端）

统一为 [2.2](#22-响应包装与-code-码表) 的四字段 JSON，按事件顺序可能出现：

**(1) 片段时间区间消息 `code=0`**（主要输出）

```json
{
  "code": 0,
  "info": "{\"key\": \"...\", \"text\": \"<|zh|><|withitn|>军事动作类型片里。\", \"avg_logprob\": -0.01}",
  "data": "<|zh|><|withitn|>军事动作类型片里。",
  "sentences": [{"start_ms": 0, "end_ms": 1970, "text": "军事动作类型片里。"}]
}
```

- `data` 为**该片段的原始文本**（含控制标记，与旧版逐字节一致）；
- `sentences` 的时间为**相对音频流起点**的绝对毫秒（非相对本片段）；
- 一个 VAD 片段若含多句，`sentences` 会有多项，首尾锚定片段边界；
- `info` 是 ASR 原始结果对象的 JSON 字符串（含 `key` / `text` / `avg_logprob`）。

**(2) 说话人命中消息 `code=2`**（仅当传入 `reg_spks` 且验证命中）

```json
{"code": 2, "info": "speaker", "data": "https://example.com/spk_a.wav", "sentences": []}
```

**(3) flush 确认 `code=3`**

```json
{"code": 3, "info": "flushed", "data": "", "sentences": []}
```

### 6.4 时序

```
客户端                                         服务端
  |---- WS 握手 (apiKey) ----------------------->|
  |<--- accept ----------------------------------|
  |---- 二进制帧 (300ms PCM) ------------------->|
  |---- 二进制帧 (300ms PCM) ------------------->|
  |                                    流式 VAD 检测到语音起点
  |<--- {code:0, sentences:[...]} ---------------|   ← 片段闭合即推送
  |---- 二进制帧 ... ---------------------------->|
  |---- 文本帧 "flush" ------------------------->|
  |<--- {code:0, sentences:[...]} ---------------|   ← 尾段结果（若有）
  |<--- {code:3, info:"flushed"} ----------------|   ← 收尾确认
  |---- close ---------------------------------->|
```

> **重要**：流式 VAD 在收到 `is_final=True` 之前**不会发出流尾片段的结束事件**。因此若客户端直接断开而不发 `flush`，**最后一句话会丢失**。请务必在停止录音时先发 `flush` 并等待 `code=3` 确认后再关闭连接。

### 6.5 收尾 flush

发送文本帧（内容大小写不敏感，允许首尾空白）：

```
flush
```

也可用 `end` / `done` / `finish`，行为一致。

服务端处理逻辑：

1. 对 VAD 做一次 `is_final=True` 收尾；
2. 按以下规则补齐尾段边界：
   - 只收到了**开始事件**（有 beg 无 end）→ 用缓冲区末端补齐 `end_ms`；
   - VAD **始终未报出片段**但缓冲区仍有 ≥ 300 ms 未转写音频 → 整段兜底转写（`beg = 缓冲区起点`）；
3. 若补齐出有效片段 → 先推 `code=0` 结果；
4. 无论如何都追加一条 `code=3` 确认。

**幂等性**：`flush` 可重复发送；第二次 `flush` 若缓冲区已空，只回 `code=3`。`flush` 之后连接**仍可继续使用**（可继续推流）。

**其他文本帧**：内容不是上述关键字时，服务端**仅记日志并忽略**，不会中断连接。

### 6.6 关闭与异常

| 场景 | 表现 |
|---|---|
| 客户端主动 `close()` | 正常断开；服务端清理缓存与缓冲区 |
| `apiKey` 缺失/错误 | 关闭码 **`1008`**，reason `Invalid or missing API key` |
| 服务端内部异常 | 记错误日志并关闭连接 |
| `reg_spks` 非空但未命中说话人 | 该片段**不推送** `code=0`（仅可能出现 `code=2`），这是设计行为 |

### 6.7 调用示例

**JavaScript（浏览器麦克风推流，含 flush 收尾）**

```javascript
const API_KEY = "sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c";
const ws = new WebSocket(
  `ws://127.0.0.1:27000/ws/transcribe?apiKey=${encodeURIComponent(API_KEY)}&lang=zh`
);
ws.binaryType = "arraybuffer";

const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const ctx = new AudioContext({ sampleRate: 16000 });
const source = ctx.createMediaStreamSource(stream);
const processor = ctx.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
  if (ws.readyState !== WebSocket.OPEN) return;
  const f32 = e.inputBuffer.getChannelData(0);
  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    i16[i] = Math.max(-1, Math.min(1, f32[i])) * 0x7fff;
  }
  ws.send(i16.buffer); // 二进制帧：16kHz / 16bit / 单声道
};
source.connect(processor);
processor.connect(ctx.destination);

ws.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  if (msg.code === 0) {
    msg.sentences.forEach((s) =>
      console.log(`[${s.start_ms} - ${s.end_ms} ms] ${s.text}`)
    );
  } else if (msg.code === 2) {
    console.log("说话人命中:", msg.data);
  } else if (msg.code === 3) {
    console.log("收尾完成，可以安全关闭连接");
    ws.close();
  }
};

// 停止录音：必须先 flush，再等 code=3 后关闭
function stopRecording() {
  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  ws.send("flush");
}
```

> 仓库内的 `test_client_wss.html` 为可直接运行的完整版本（已按上述时序实现）。

**Python（websockets）**

```python
import asyncio
import json
import wave

import websockets

URL = "ws://127.0.0.1:27000/ws/transcribe?apiKey=sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c"
# 必须为 16kHz / 16-bit / 单声道，否则需要先转码
WAV_PATH = "examples/test.wav"
FRAME_BYTES = 4800 * 2  # 300ms 音频（16000 * 0.3 样本 * 2 字节）


async def main():
    with wave.open(WAV_PATH, "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1, \
            "需要 16kHz / 16-bit / 单声道 WAV"
        pcm = w.readframes(w.getnframes())  # 去掉 WAV 头，取裸 PCM

    async with websockets.connect(URL) as ws:
        for i in range(0, len(pcm), FRAME_BYTES):
            await ws.send(pcm[i:i + FRAME_BYTES])  # 二进制帧

        await ws.send("flush")  # 收尾，避免丢尾句
        async for raw in ws:
            msg = json.loads(raw)
            if msg["code"] == 0:
                for s in msg["sentences"]:
                    print(f'[{s["start_ms"]:>6} - {s["end_ms"]:>6} ms] {s["text"]}')
            elif msg["code"] == 2:
                print("说话人命中:", msg["data"])
            elif msg["code"] == 3:
                break


asyncio.run(main())
```

---

## 7. 句子级时间戳语义

### 7.1 时间来源

SenseVoiceSmall（`model.py`）的 `inference()` **只返回 `key` / `text` / `avg_logprob`，不输出任何时间戳**。因此时间信息**只能**取自 funasr VAD（`speech_fsmn_vad_zh-cn-16k-common-pytorch`）的语音片段边界 `[beg_ms, end_ms]`。

本方案**不修改模型、不改变加载方式**（仍为 `local_files_only=True` 本地离线加载）。

| 环节 | HTTP `/transcribe` | WS `/ws/transcribe` |
|---|---|---|
| 时间来源 | 对整段音频跑一次 VAD 得到 `[beg, end]` | 复用链路内部已有的流式 VAD 片段边界（相对流起点） |
| 片段文本 | 每个 VAD 片段单独转写 | 同左 |
| 时间基准 | 音频起点 | 音频流起点 |

### 7.2 句级切分与插值规则

1. 片段文本按**句末标点**断句：`。！？；!?;…`（连续标点不产生空句，标点保留在句尾）；
2. 片段内**只有一句** → 时间即 VAD 片段的**精确**边界；
3. 片段内**多句** → 首句 `start_ms` 与末句 `end_ms` **严格锚定** VAD 片段边界，中间边界按**累计字符数比例**插值。

设某片段边界为 $[b, e]$，切成 $n$ 句 $s_1, s_2, \dots, s_n$，第 $i$ 句累计字符占比

$$r_i = \frac{\sum_{k=1}^{i} \mathrm{len}(s_k)}{\sum_{k=1}^{n} \mathrm{len}(s_k)}$$

则边界（并强制单调不减、裁剪到 $[b, e]$）

$$\text{stop}_i = \min\left(\max\left(b + \mathrm{round}\big((e-b)\cdot r_i\big),\ \text{cursor}\right),\ e\right),\quad i < n$$

$$\text{stop}_n = e$$

**性质**：$\text{start}_1 = b$，$\text{end}_n = e$，且所有区间单调递增、互不重叠。

**边界加固**：

| 情形 | 处理 |
|---|---|
| 片段边界为负 | 钳为 `0` |
| `end < beg`（逆序） | 自动交换 |
| `end == beg`（零长片段） | 区间退化为一个点 `[b, b]`，文本仍保留 |
| 片段文本清洗后为空 | 跳过，不产生句子 |
| 片段数多于文本数 | 按 `zip` 截断 |

### 7.3 精度与已知限制

| 项目 | 结论 |
|---|---|
| **片段级时间** | VAD **实测边界**，实测与真值偏差多在 **0～310 ms** 内 |
| **片段内多句的中间边界** | **近似值**（按字符比例插值，与实际语速无关） |
| **时间连续性** | `sentences` 之间通常有静音间隙，**不连续** |
| **极端情况** | 长停顿、音乐/噪声、语速剧烈变化时，VAD 边界精度会下降 |

> 若需更高精度，需要在 `server.py` 中调整 VAD 参数 `max_end_silence_time`（默认 `500 ms`）与 `speech_noise_thres`（默认 `0.6`），或升级 funasr 以启用词级时间戳（当前钉版 `1.2.7` 不可用，实验见 `tmp/test_word_ts.py`）。

### 7.4 实验与验证

| 脚本 | 作用 |
|---|---|
| `tmp/test_vad_timing.py` | 全文件 VAD 与「服务端式」流式 VAD 的片段时间 vs 真值 |
| `tmp/test_asr_segments.py` | VAD 分段 + 逐段 ASR，产出 `out/result_segments.json`，做文本连续性校验 |
| `tmp/test_word_ts.py` | 验证 SenseVoice `output_timestamp=True` 是否可用（结论：当前版本不可用） |
| `tmp/test_timestamp_protocol.py` | **协议级回归**：假模型驱动真实路由，覆盖 HTTP/WS 两条链路的 `sentences`、flush、错误分支 |

---

## 8. 配置项

配置在 `server.py` 的 `Config` 类（`pydantic-settings`，支持环境变量覆盖），当前版本无配置文件。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `api_key` | `str` | `sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c` | API 密钥，HTTP Query / WS Query 均校验此值 |
| `sv_thr` | `float` | `0.3` | 说话人验证阈值，`score >= sv_thr` 视为同一说话人 |
| `chunk_size_ms` | `int` | `300` | 流式 VAD 每次处理的音频时长（ms） |
| `sample_rate` | `int` | `16000` | 音频采样率 |
| `bit_depth` | `int` | `16` | 位深 |
| `channels` | `int` | `1` | 声道数 |
| `avg_logprob_thr` | `float` | `-0.25` | 平均 logprob 阈值 |
| `data_from_full_asr` | `bool` | `True` | `data` 字段的来源开关，见下 |

### `data_from_full_asr` 取舍

| 取值 | `data` 内容 | 额外开销 |
|---|---|---|
| `True`（默认） | **整段一次性转写**的原始文本，与历史版本**逐字节一致** | 多跑一次整段 ASR（耗时约翻倍） |
| `False` | 各 VAD 片段文本拼接（`第一句。第二句！…`） | 省掉整段推理，响应更快；但 `data` 不含控制标记，且**与旧版输出不再逐字节相同** |

> 无论该开关为何值，`sentences` 的内容**完全一致**。

### 模型文件（离线，`models/` 目录，已 gitignore）

| 用途 | 模型 |
|---|---|
| ASR | `SenseVoiceSmall` |
| VAD | `speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| 说话人验证 | `speech_eres2net_large_sv_zh-cn_3dspeaker_16k` |

启动前会将环境变量 `MODELSCOPE_CACHE` 与 `FUNASR_CACHE` 指向 `models/`，全部以 `local_files_only=True` 加载，**运行期不联网**。

---

## 9. 兼容性与迁移

### 9.1 向后兼容保证

| 项目 | 状态 |
|---|---|
| `code` / `info` / `data` 三字段 | ✅ **完全不变**（`data` 在默认配置下逐字节一致） |
| `POST /transcribe` 请求格式 | ✅ 不变 |
| WS 二进制帧处理逻辑 | ✅ 逐字节行为不变 |
| WS `code=0` / `code=2` 消息 | ✅ 语义不变，仅**新增** `sentences` 字段 |
| 新增字段 | `sentences`（默认 `[]`） |

**旧客户端无需任何改动**即可继续工作；新客户端读取 `sentences` 获得时间区间。

### 9.2 迁移提示

**1）解析 WS 结果时别忘了 `flush`**（最重要）

旧代码若直接 `close()`，仍能拿到大部分片段，但**尾句会丢**。建议改为：

```javascript
// 旧：直接关闭
// ws.close();

// 新：先 flush，收到 code=3 再关闭
ws.send("flush");
// ws.onmessage 中： if (msg.code === 3) ws.close();
```

**2）HTTP 响应耗时变化**

默认配置下 `POST /transcribe` 耗时约为旧版**2 倍**（多一次整段 ASR）。若端到端延迟敏感且可接受 `data` 变化，可设 `Config.data_from_full_asr = False`。

**3）错误处理需区分两种形态**

见 [2.3](#23-错误响应的两种形态)。旧代码若统一读 `data.msg`，需改为读 `detail`（HTTP 校验错误）或 `info`（业务错误）。

**4）注意 `data` 与 `sentences[].text` 的差别**

- `data`（WS）含 `<|zh|>` 等控制标记，是**原始输出**；
- `sentences[].text` 是**清洗后**的纯文本。

显示用途请优先用 `sentences[].text`；如需完整兼容旧行为请继续用 `data`。

---

## 10. 常见问题

**Q1. `sentences` 为空数组 `[]`，但 `data` 有文本？**

说明 VAD 未检测到语音片段（如纯静音、极低音量、纯音乐），而整段 ASR 仍输出了文本。属正常现象，可结合 `data` 使用。

**Q2. WS 最后一句话没收到？**

未发送 `flush` 控制帧。见 [6.5](#65-收尾-flush) 与 [9.2](#92-迁移提示)。

**Q3. 传入 `reg_spks` 后 WS 完全不出文本？**

设计如此：当 `reg_spks` 非空且未命中说话人时，该片段只发 `code=2`（命中时）或不发结果，**不推送** `code=0` 文本。

**Q4. 句子区间为什么是「近似」的？**

SenseVoice 不输出时间戳，时间只能来自 VAD 片段边界。**片段级**是实测边界，**片段内多句**的中间边界按字符比例插值，因此是近似值。见 [7.3](#73-精度与已知限制)。

**Q5. 上传 MP3 报错？**

仅支持 `audio/wav` 与 `audio/webm`（按 `Content-Type` 判定）。请在客户端转码为 WAV 后上传。

**Q6. 想要词级时间戳？**

当前 funasr 钉版 `1.2.7` 下 SenseVoice 的 `output_timestamp=True` 不可用（见 `tmp/test_word_ts.py`）。需升级依赖后另行评估。

**Q7. 如何本地验证改动没破坏协议？**

```bash
python tmp/test_timestamp_protocol.py     # 38 项协议级断言
```

该脚本用**假模型**（替换 `funasr` / `modelscope` / `soundfile` / `torch` / `loguru`）加载**真实** `server.py`，通过 TestClient 驱动真实路由，**无需 torch 与模型文件**。依赖：`fastapi` `starlette` `pydantic` `pydantic-settings` `numpy` `httpx`。

---

## 11. 变更记录

| 版本 | 变更 |
|---|---|
| 本版（含时间区间） | 响应新增 `sentences` 字段（句子级 `[start_ms, end_ms]`）；HTTP 链路新增整段 VAD + 逐段转写；WS 链路复用内部 VAD 边界并为片段附带 `sentences`；新增 WS 文本控制帧 `flush`（含 `end`/`done`/`finish`）与 `code=3` 确认；新增 `Config.data_from_full_asr`；修复：异常处理器字段名错误、WS 孤立结束事件污染后续片段、WS 缓冲偏移量与实际裁样不一致 |
| 历史版本 | `code` / `info` / `data` 三字段；HTTP 整段转写；WS 流式 VAD + 逐段 ASR；说话人验证 |

---

## 附：完整交互示例

**HTTP：一次完整调用与结果解读**

```bash
$ curl -s -X POST "http://127.0.0.1:27000/transcribe?apiKey=sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c" \
    -F "file=@examples/test.wav;type=audio/wav" | python -m json.tool
{
    "code": 0,
    "info": "success, transcription time: 2.41 seconds",
    "data": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>军事动作类型片里。我有别人没有拥有过的经验。",
    "sentences": [
        {"start_ms": 0,    "end_ms": 1970, "text": "军事动作类型片里。"},
        {"start_ms": 2570, "end_ms": 6160, "text": "我有别人没有拥有过的经验。"}
    ]
}
```

解读：

- 第 1 句落在 `0 – 1970 ms`，第 2 句落在 `2570 – 6160 ms`；
- 两段之间 `1970 – 2570 ms` 的 600 ms 是 VAD 判定的静音间隔，因此区间不连续；
- 每个 VAD 片段各含一句，所以 `start_ms` / `end_ms` 均为**实测精确边界**；
- `data` 中的 `<|zh|><|NEUTRAL|><|Speech|><|withitn|>` 是模型控制标记，`sentences[].text` 已清洗。

**WS：一次完整会话的消息序列**

```
→ 二进制帧 × N   (16kHz/16bit/mono PCM)
← {"code":0,"info":"{\"key\":\"...\",\"text\":\"<|zh|><|withitn|>第一句。\",\"avg_logprob\":-0.01}",
   "data":"<|zh|><|withitn|>第一句。",
   "sentences":[{"start_ms":0,"end_ms":1970,"text":"第一句。"}]}
→ 二进制帧 × N
→ 文本帧 "flush"
← {"code":0,"info":"...","data":"...","sentences":[{"start_ms":6400,"end_ms":8100,"text":"最后一句。"}]}
← {"code":3,"info":"flushed","data":"","sentences":[]}
→ close
```
