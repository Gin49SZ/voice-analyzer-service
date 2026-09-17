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

### Docker 部署

依赖全部放在本地目录，**构建期零网络**（compose 给 `app` 设了 `build.network: none`）：

| 本地物料 | 内容 | 来源 |
|---|---|---|
| `debhouse/*.deb` | 系统依赖 libsndfile1 / ffmpeg 及其依赖闭包，约 128 MB | `deps` 服务 |
| `wheelhouse/*.whl` | Python 依赖，111 个 wheel，约 710 MB | `deps` 服务 |
| `models/` | 离线模型，约 1 GB | 单独拷入 |
| `python:3.10-slim` | 基础镜像 | `docker pull`，或从别处 `docker load` |

**有依赖就直接构建启动，没依赖就先下到本地** —— 一条命令：

```bash
sh scripts/build.sh
```

它内部做的事（也可以自己敲）：

```bash
# 只在本地依赖不全时才需要，需要外网
docker compose --profile online run --rm deps    # → 宿主机 ./debhouse + ./wheelhouse

# 每次构建 + 启动，零网络
docker compose up -d --build
```

> - 依赖下载放在 `docker compose run`（容器运行期 + bind mount 到宿主机）而不是 Dockerfile 里：
>   `docker build` 只能**读**构建上下文、**写不回宿主机**，容器的 bind mount 可以。
> - 本地 `.deb` / `.whl` 在 Dockerfile 里用 `RUN --mount=type=bind` **挂载**安装，不 `COPY` 进镜像 ——
>   否则那 128 MB + 710 MB 会永久留在镜像层里，事后 `rm` 也减不掉体积。

**日常操作**

```bash
docker compose logs -f app           # 日志
docker compose restart app           # 重启
docker compose down                  # 停止并删除容器
docker compose ps                    # 状态
curl http://127.0.0.1:27000/health
```

换源、镜像名等可覆盖变量见 [`.env.example`](.env.example)。

产出的镜像**运行期完全离线**：模型内置、`local_files_only=True` 加载，并设置 `HF_HUB_OFFLINE=1` 等变量。

> 镜像为 CPU 版（`torch==2.3.0+cpu`），体积约 6.4 GB（模型约 1 GB），冷启动 75 秒起（机器繁忙时可达 4 分钟，故 `HEALTHCHECK --start-period=300s`），建议容器内存 ≥ 4 GB。已实测：构建期与运行期均可完全断网，`/health`、`/transcribe`、`WS /ws/transcribe` 三条路径全部可用。

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
- 实测（`examples/test.wav`，约 73 s）：`HTTP → HTTP → 完整 WS 会话 → HTTP` 连续四次，`sentences` 完全一致（各 23 条、时间单调、无零长与重叠），说明 WS 会话不会污染后续 HTTP 的 VAD 参数（该陷阱的成因与防护见 [`docs/API.md` §7.5](docs/API.md)）；
- `data` 纯文本字段偶发差一个标点，属 ITN 环节抖动，与时间信息无关（见 [`docs/API.md` §7.6](docs/API.md)）。

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
