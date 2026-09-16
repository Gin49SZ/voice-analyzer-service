# voice-analyzer-service — 项目长期备忘

## 接口契约（务必保持）
- `TranscriptionResponse = {code, info, data, sentences}`。
- `data` 必须**始终是纯文本**（不塞结构化对象）；结构化结果一律放 `sentences`，
  每条为 `{start_ms, end_ms, text}`，时间为相对音频起点的毫秒。
- `sentences` 为附加字段，缺失/为空不得影响老客户端；`code` 语义：0 转写结果、2 说话人、3 flush ack。

## 技术约束
- SenseVoice（`model.py` / `remote_code=./model.py`）**不输出时间戳**，时间信息只能来自 VAD 片段边界；
  不要为了时间戳去改模型或更换加载方式（funasr 1.2.7 的内置 `output_timestamp` 时间数值损坏）。
- WS 流式 VAD 输出边界事件（`[beg,-1]` 开始、`[-1,end]` 结束，相对流起点毫秒）；
  不发送 `is_final=True` 的 flush 会丢失尾部片段。
- 本地模型从 `models/` 目录 `local_files_only=True` 加载；`MODELSCOPE_CACHE`/`FUNASR_CACHE` 指向该目录。

## 工程习惯
- 用户习惯：先给结论/方案，再给可落地的具体实现；输出用 Markdown + 表格。
- 实验与验证脚本统一放 `tmp/`，不污染项目源码；`tmp/out/` 放产物。
- 改动源码前先做隔离实验验证可行性，落地后补协议级回归测试。
