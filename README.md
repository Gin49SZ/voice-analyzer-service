# voice-analyzer-service

基于 FunASR (SenseVoiceSmall) 的音频实时转录 Web 服务，支持文件上传和 WebSocket 流式识别，集成 VAD 语音活动检测与说话人验证。

## 功能特性

- **文件转录** — POST 上传 WAV/WebM 音频文件，返回识别文本
- **WebSocket 流式识别** — 实时麦克风输入，低延迟流式转写
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

### `POST /transcribe` — 音频文件转录

请求参数：
- `file` (form-data): WAV 或 WebM 音频文件
- `apiKey` (query): API 密钥

### `WS /ws/transcribe` — WebSocket 流式转录

查询参数：
- `apiKey` (required): API 密钥
- `lang` (optional): 语言，默认 `zh`
- `reg_spks` (optional): 已注册说话人音频 URL（逗号分隔）

### `GET /health` — 健康检查

## 配置

在 `server.py` 的 `Config` 类中可配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `api_key` | `sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c` | API 密钥 |
| `sv_thr` | `0.3` | 说话人验证阈值 |
| `chunk_size_ms` | `300` | 每次处理的音频时长 (ms) |

## 测试客户端

- `test_client.html` — HTTP 上传测试（浏览器打开）
- `test_client_wss.html` — WebSocket 流式测试（需要麦克风权限）
- `examples/` — 示例音频文件

## 致谢 & 衍生声明 (Attribution)

本项目基于 [api4sensevoice](https://github.com/0x5446/api4sensevoice.git) 开发，由 [0x5446/t1ger] 保留原始版权。

在此向原项目的作者表示深深的感谢！
