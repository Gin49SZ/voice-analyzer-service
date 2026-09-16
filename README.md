# voice-analyzer-service

基于 FunASR (SenseVoiceSmall) 的音频实时转录 Web 服务，支持文件上传和 WebSocket 流式识别，集成 VAD 语音活动检测与说话人验证。

## 功能特性

- **文件转录** — POST 上传 WAV/WebM 音频文件，返回识别文本
- **WebSocket 流式识别** — 实时麦克风输入，低延迟流式转写
- **句子级时间戳** — 每条结果附带每句话相对音频开始的 `[start_ms, end_ms]`
- **VAD 端点检测** — 自动检测语音起止，支持流式缓存
- **说话人验证** — 基于声纹比对，判断音频是否匹配已注册说话人
- **GPU 加速** — 可选 CUDA 推理

## 快速开始

### 环境要求

- Python 3.10+
- 模型文件（离线）：需将模型放入 `models/` 目录

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

CPU 模式（默认）：
```bash
python server.py --port 27000
```

GPU 模式：
```bash
python server.py --port 27000 --gpu true
```

### Docker

```bash
docker build -t voice-analyzer .
docker run -p 27000:27000 voice-analyzer
```

## API 接口

> 完整 API 文档（含字段模型、错误码、调用示例、时序与兼容性说明）：**[`docs/API.md`](docs/API.md)**

### `POST /transcribe` — 音频文件转录

请求参数：
- `file` (form-data): WAV 或 WebM 音频文件
- `apiKey` (query): API 密钥

响应示例：

```json
{
  "code": 0,
  "info": "success, transcription time: 1.23 seconds",
  "data": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>军事动作类型片里。",
  "sentences": [
    {"start_ms": 0, "end_ms": 1970, "text": "军事动作类型片里。"},
    {"start_ms": 2570, "end_ms": 6160, "text": "我有别人没有拥有过的经验。"}
  ]
}
```

- `data`：与历史版本完全一致，仍是**整段一次性转写**的纯文本；
- `sentences`：新增字段，承载结构化结果，时间相对**音频起点**（毫秒）。

### `WS /ws/transcribe` — WebSocket 流式转录

查询参数：
- `apiKey` (required): API 密钥
- `lang` (optional): 语言，默认 `zh`
- `reg_spks` (optional): 已注册说话人音频 URL（逗号分隔）

连接建立后发送 **二进制帧**（16kHz / 16bit / 单声道 PCM，建议每帧 300ms），服务端每检测到一个完整语音片段就回一条消息：

```json
{
  "code": 0,
  "info": "{\"key\": \"...\", \"text\": \"...\"}",
  "data": "<|zh|><|withitn|>军事动作类型片里。",
  "sentences": [{"start_ms": 0, "end_ms": 1970, "text": "军事动作类型片里。"}]
}
```

`code` 含义：`0` 转写结果、`2` 说话人验证命中、`3` flush 完成确认。

**文本控制帧（可选，向后兼容）**：发送文本帧 `flush`（也接受 `end` / `done` / `finish`）后，服务端会对缓冲区里最后一段语音做 `is_final=True` 的 VAD 收尾与转写，先回该片段结果，再回一条 `code=3` 的确认；否则停止录音时**最后一句话会丢失**（流式 VAD 不 flush 时拿不到尾部片段的结束事件）。二进制帧的处理逻辑与旧版本完全一致。

### `GET /health` — 健康检查

## 句子级时间戳

SenseVoice（`model.py`）**不输出任何时间戳**，因此时间信息取自 funasr VAD 的语音片段边界：

| 环节 | HTTP `/transcribe` | WS `/ws/transcribe` |
|---|---|---|
| 时间来源 | 对整段音频跑一次 VAD 得到 `[beg, end]` | 复用链路内部已有的 VAD 片段边界（相对流起点） |
| 文本 | 每个 VAD 片段单独转写 | 同左 |
| 句级处理 | 片段文本按 `。！？；!?;…` 断句；片段内多句时首句 `start_ms` 与末句 `end_ms` 严格锚定 VAD 边界，中间边界按字符数比例插值 | 同左 |

- 精度：**片段级时间为 VAD 实测边界**（实验验证：与真值偏差多在 0～310ms 内），**片段内多句的中间边界为近似值**；
- 不修改模型与加载方式，仍为 `local_files_only=True` 本地加载；
- 实验与验证脚本见 `tmp/`：`test_vad_timing.py`（片段时间 vs 真值）、`test_asr_segments.py`（分段+逐段转写）、`test_timestamp_protocol.py`（两条链路的协议级回归）。

## 配置

在 `server.py` 的 `Config` 类中可配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `api_key` | `sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c` | API 密钥 |
| `sv_thr` | `0.3` | 说话人验证阈值 |
| `chunk_size_ms` | `300` | 每次处理的音频时长 (ms) |
| `data_from_full_asr` | `True` | `data` 是否保留「整段一次性转写」以完全兼容旧版本；置为 `False` 时改用 VAD 片段文本拼接，可省掉一次整段推理（降低响应耗时） |

## 测试客户端

- `test_client.html` — HTTP 上传测试（浏览器打开）
- `test_client_wss.html` — WebSocket 流式测试（需要麦克风权限）
- `examples/` — 示例音频文件

## 致谢 & 衍生声明 (Attribution)

本项目基于 [api4sensevoice](https://github.com/0x5446/api4sensevoice.git) 开发，由 [0x5446/t1ger] 保留原始版权。

在此向原项目的作者表示深深的感谢！
