# 测试用例：转录结果带每段话起止时间（可行性验证）

## 目标
验证「返回带有标记每一段话起始终止时间（相对音频开始）的转录结果」是否可实现，
以及采用哪种实现路径。**不修改项目源码**，所有测试代码与本目录隔离。

## 背景结论（来自调研）
- SenseVoice（本项目所用远程代码 `model.py`）输出 `key/text/avg_logprob`，**不含时间戳**；
- 文本带标点（`use_itn=True`），可按 `。！？；` 切分句子；
- WS 链路（`server.py`）内部已用 fsmn VAD 计算语音片段边界，但未外发；HTTP 链路无任何时间信息。

## 可行性路径（本测试验证）
A. **VAD 分段计时**：fsmn VAD 给出每个语音片段 `[start_ms, end_ms]`（相对音频起点）→ 片段级时间；
B. **逐段 ASR**：对每个 VAD 片段单独转写，得到 `[{start_ms, end_ms, text}]`；
C. **句内插值**：片段内按标点断开句子，时间按字符数比例分配（近似）；
D. **词级时间戳（备选）**：funasr 内置 SenseVoice 类 `output_timestamp=True`（CTC 对齐）。

## 测试文件
| 文件 | 内容 |
|---|---|
| `gen_test_audio.py` | 从 `examples/test.wav` 截取 3 段已知音频拼成 `out/mixed.wav`，生成真值 `out/mixed_ground_truth.json` |
| `test_vad_timing.py` | 测试 A：全文件 VAD 与「服务端式」流式 VAD 的片段时间 vs 真值（容差判定） |
| `test_asr_segments.py` | 测试 A+B+C：VAD 分段 + 逐段 ASR，产出 `out/result_segments.json`，并做文本连续性校验 |
| `test_word_ts.py` | 测试 D：funasr 内置 SenseVoice `output_timestamp=True` 是否可得到词级时间戳 |
| `test_timestamp_protocol.py` | **功能落地后的回归测试**：`timestamps.py` 纯函数边界用例 + 两条链路的协议级测试（假模型驱动真实端点，校验 `sentences` 时间、`data` 兼容性与 WS flush 控制帧） |

## 运行（需在项目根目录，模型本地加载）
```bash
.venv/bin/python tmp/gen_test_audio.py
.venv/bin/python tmp/test_vad_timing.py
.venv/bin/python tmp/test_asr_segments.py
.venv/bin/python tmp/test_word_ts.py
```
运行前提：`models/` 下已有三个本地模型；内存 ≥ 4GB 可用（`test_asr_segments.py`/`test_word_ts.py` 需加载 SenseVoice，约 2GB）。

## 判定标准
- VAD 片段时间与真值偏差 ≤ 600ms（fsmn VAD 存在默认 padding）视为通过；
- 逐段 ASR 文本拼接与整段 ASR 文本高度一致（边界处允许少量差异）视为时序可用；
- `output_timestamp=True` 能产出词级时间即证明路径 D 可行。

## 实测结果（2026-09-12）

| 测试 | 结果 |
|---|---|
| `gen_test_audio.py` | ✅ 从 test.wav 选出 3 个经「能量 + ASR 非空文本」双重校验的语音区（军事动作类型片里。/ 我有别人没有拥没有拥有过的经验。/ 不是拿票房能够换得来的。），拼接为 out/mixed.wav（9.29s） |
| `test_vad_timing.py` | ✅ 全文件模式 3/3 精确匹配（多数 Δ=0ms）；流式+flush 3/3（尾部 Δ=-310ms 内）；**流式无 flush 丢失尾部片段**（= server.py 现状，实现时需补 `is_final=True`） |
| `test_asr_segments.py` | ✅ VAD 分段 + 逐段 ASR 产出 `[{start_ms,end_ms,text}]`：边界 3/3 匹配、片段文本与源语音区文本重叠率 1.0、拼接 vs 整段 LCS 重叠率 94.6%；句子级结果见 out/result_segments.json |
| `test_word_ts.py` | ⚠ **funasr 1.2.7 内置类可加载并输出 `timestamp`/`words`，但时间数值损坏**（如单个字符被标为 330 秒，远超音频时长）→ 1.2.7 版本此功能不可用，需升级 funasr（新版已重做格式为毫秒对）或仅依赖 VAD 分段路径 |

**结论**：基于 VAD 分段 + 逐段 ASR 的「片段级起止时间 + 文本」已实测可行（路径 A+B）；
路径 D（词级时间戳）在当前 funasr 1.2.7 钉版下不可用，需升级依赖。

## 落地实现（2026-09-15）

按上述结论把时间信息落到两条链路，**不改模型与加载方式**：

| 项 | 实现 |
|---|---|
| 新增字段 | `TranscriptionResponse.sentences = [{start_ms, end_ms, text}]`，时间相对音频起点（毫秒） |
| `data` 字段 | 保持纯文本与原行为不变（HTTP 仍为整段一次性转写；WS 仍为片段原始文本） |
| 工具模块 | `timestamps.py`：控制标记清理、按句末标点断句、以 VAD 片段边界为锚点生成句子区间 |
| HTTP | 整段 VAD → 逐段 ASR → `sentences`；`Config.data_from_full_asr=False` 可省掉整段推理 |
| WS | 复用链路内部 VAD 片段的绝对边界（beg/end 事件）→ `sentences`；新增文本控制帧 `flush` 触发 `is_final` 收尾，避免尾部片段丢失（回 `code=3` ack） |

`test_timestamp_protocol.py` 实测：**38/38 项通过**（纯函数 13 项 + HTTP 11 项 + WS 13 项 + 其他）。
运行方式：装轻量依赖（fastapi/starlette/pydantic/pydantic-settings/numpy/httpx/uvicorn/python-multipart），
无需 torch/funasr，脚本会用假模型替换推理库后直接驱动真实路由：

```bash
python tmp/test_timestamp_protocol.py
```