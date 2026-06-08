from __future__ import annotations

"""Per-clip captioning for the miner pipeline.

The captioner takes the first frame of each clip and asks an OpenAI-compatible
vision model for a short prompt-style description. Captions feed into the
trainer manifest's `prompt` field; if no API key is configured the captioner
returns an empty string and the trainer falls back to its default prompt.
"""
"""
矿工 Caption 生成器。

功能：对每段视频的首帧（JPG）调用 OpenAI/Gemini 视觉模型，生成一句简短的
文本描述（prompt）。这段文本会被写入 trainer manifest 的 `prompt` 字段，
供 LoRA 训练时使用。

如果没有配置 API Key，captioner 会生成空字符串，Trainer 会回退到默认 prompt。
"""

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 发送给 LLM 的 system prompt：要求用一句话描述画面，适合作为文生视频的 prompt
_PROMPT = (
    "Describe this video frame in one short sentence (≤ 20 words) that would "
    "work as a text-to-video generation prompt. Focus on subject, setting, and "
    "motion cues. Do not add commentary."
)


def _b64_image(path: Path) -> str:
    """将图片文件转为 Base64 字符串。"""
    return base64.b64encode(path.read_bytes()).decode("ascii")


@dataclass
class Captioner:
    """
    Caption 生成器。

    使用 OpenAI 兼容 API（OpenAI 官方或 Gemini 代理端点）。
    """
    api_key: str = ""                          # OpenAI 或 Gemini API Key
    model: str = "gpt-4o-mini"                 # 默认模型
    base_url: str | None = None                # 自定义端点（用于 Gemini 代理）
    timeout_sec: int = 30                      # API 调用超时

    def __post_init__(self) -> None:
        """初始化 OpenAI 客户端；如果没有 API Key 则禁用。"""
        self._client = None
        if not self.api_key.strip():
            logger.info("captioner disabled: no API key configured")
            return
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=float(self.timeout_sec),
            )
        except Exception as exc:
            logger.warning("captioner init failed err=%s; will return empty captions", exc)
            self._client = None

    @property
    def enabled(self) -> bool:
        """Captioner 是否可用（有有效的 API Key）。"""
        return self._client is not None

    def caption_frame(self, frame_path: Path) -> str:
        """
        对单张首帧图片生成 caption。

        流程：
        1. 读取图片并转为 Base64 Data URL
        2. 调用 vision model 生成描述
        3. 返回截断到 300 字符的文本

        失败时返回空字符串。
        """
        if self._client is None or not frame_path.exists():
            return ""
        try:
            data_url = f"data:image/jpeg;base64,{_b64_image(frame_path)}"
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=80,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text[:300]
        except Exception as exc:
            logger.warning("caption call failed frame=%s err=%s", frame_path, exc)
            return ""
