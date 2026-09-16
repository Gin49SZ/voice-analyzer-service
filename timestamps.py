#!/usr/bin/env python3
"""句子级时间区间工具（纯函数，不依赖模型/推理库，可单独测试）。

背景
----
本项目 ASR 用的 SenseVoice（`remote_code=./model.py`）**不输出任何时间戳**：
`model.py::inference()` 只返回 `key/text/avg_logprob`。因此时间信息只能来自
funasr VAD（`speech_fsmn_vad_zh-cn-16k-common-pytorch`）给出的语音片段边界
`[beg, end]`（毫秒，相对音频起点）。VAD 片段 = 一句完整的话（一个完整语句），
即 tmp/ 实验验证后选定的方案（见 tmp/README.md，「VAD 分段 + 逐段 ASR」）。

本模块只做「VAD 片段边界 + 片段转写文本」→「句子 + [start_ms, end_ms]」的整理：

- 片段文本按标点（。！？；!?;…）断句；
- 片段内只有一句时，时间就是 VAD 片段的**精确**边界；
- 片段内有多句时，首句 `start_ms` 与末句 `end_ms` 仍**严格锚定** VAD 片段边界，
  中间边界在片段内按字符数比例插值（片段级精确、句级内部近似）。

约定
----
- 所有时间均为相对音频（或音频流）起点的整数毫秒；
- 区间语义为闭区间 `[start_ms, end_ms]`，保证 `start_ms <= end_ms`；
- 句子文本已去除 SenseVoice 的 `<|zh|><|NEUTRAL|>...` 控制标记。
"""
from typing import Dict, Iterable, List, Sequence, Union
import re

__all__ = [
    "TAG_RE",
    "SENT_END_RE",
    "clean_text",
    "split_sentences",
    "normalize_segment",
    "build_sentences",
]

# SenseVoice 输出里的控制标记，例如 <|zh|><|NEUTRAL|><|Speech|><|withitn|>
TAG_RE = re.compile(r"<\|[^|]*\|>")

# 句末标点（与 tmp/test_asr_segments.py 的切分规则保持一致）
SENT_END_RE = re.compile(r"[。！？；!?;…]+")

VadSegment = Sequence[Union[int, float]]
Sentence = Dict[str, Union[int, str]]


def clean_text(text: str) -> str:
    """去掉 SenseVoice 控制标记与首尾空白。

    @param text 模型原始输出（可能含 <|zh|> 等标记）
    @return 纯文本
    """
    if not text:
        return ""
    return TAG_RE.sub("", text).strip()


def split_sentences(text: str) -> List[str]:
    """按句末标点把一段文本切成句子，标点保留在句尾。

    @param text 片段文本（可含控制标记）
    @return 句子列表，空白句会被丢弃
    """
    text = clean_text(text)
    if not text:
        return []
    parts = SENT_END_RE.split(text)
    ends = SENT_END_RE.findall(text)
    sentences: List[str] = []
    for i, part in enumerate(parts):
        sentence = (part + ends[i]) if i < len(ends) else part
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def normalize_segment(segment: VadSegment) -> List[int]:
    """把 VAD 片段的 [beg, end] 规整为两个非负整数毫秒，并保证 beg <= end。

    @param segment VAD 输出的片段边界
    @return [beg_ms, end_ms]
    """
    beg = int(segment[0])
    end = int(segment[1])
    if beg < 0:
        beg = 0
    if end < 0:
        end = 0
    if end < beg:
        beg, end = end, beg
    return [beg, end]


def build_sentences(segments_ms: Iterable[VadSegment],
                    segment_texts: Iterable[str]) -> List[Sentence]:
    """把 VAD 片段边界与片段文本整理成「一句完整的话 + 时间区间」。

    @param segments_ms VAD 片段边界列表，每项 [beg_ms, end_ms]，相对音频起点
    @param segment_texts 与片段一一对应的模型原始文本（可含控制标记）
    @return [{"start_ms": int, "end_ms": int, "text": str}, ...]，按时间升序
    """
    sentences: List[Sentence] = []
    for segment, raw_text in zip(segments_ms, segment_texts):
        beg, end = normalize_segment(segment)
        items = split_sentences(raw_text)
        if not items:
            continue
        if end <= beg:
            # 片段边界退化（长度为 0）：保留文本但区间退化为一个点，便于排查
            sentences.extend(
                {"start_ms": beg, "end_ms": beg, "text": item} for item in items
            )
            continue
        total_chars = sum(len(item) for item in items)
        cursor = beg
        for i, item in enumerate(items):
            if i == len(items) - 1:
                stop = end
            else:
                # 片段内多句：按累计字符数比例插值，且保证单调不减
                done_chars = sum(len(x) for x in items[: i + 1])
                stop = beg + int(round((end - beg) * done_chars / total_chars))
                stop = min(max(stop, cursor), end)
            sentences.append({"start_ms": cursor, "end_ms": stop, "text": item})
            cursor = stop
    return sentences
