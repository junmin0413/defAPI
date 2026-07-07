from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from defapi.models import Report


SYSTEM_MESSAGE = (
    "You are a secure coding assistant. Analyze vulnerable code and provide safe, practical fixes."
)
REVIEW_SYSTEM_MESSAGE = (
    "You are a senior application security reviewer. Review proposed fixes for correctness, "
    "residual risk, and missing verification steps."
)
RESPONSE_FORMAT = """응답은 반드시 다음 네 섹션을 순서대로 포함해야 한다.
1. 취약점 설명
2. 안전한 수정 코드
3. 수정 이유
4. 추가 주의사항"""
REVIEW_FORMAT = """응답은 반드시 다음 네 섹션을 순서대로 포함해야 한다.
1. 패치 검토 결과
2. 남은 위험
3. 검증 방법
4. 배포 전 체크리스트"""


def _load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_project_env()


@dataclass(frozen=True)
class SGLangConfig:
    base_url: str = field(default_factory=lambda: os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("SGLANG_API_KEY", "None"))
    model: str = field(default_factory=lambda: os.getenv("SGLANG_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct"))
    lora_name: str | None = field(default_factory=lambda: os.getenv("SGLANG_LORA_NAME", "defapi"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("SGLANG_MAX_TOKENS", "2048")))
    temperature: float = field(default_factory=lambda: float(os.getenv("SGLANG_TEMPERATURE", "0.1")))

    @property
    def request_model(self) -> str:
        if self.lora_name:
            if self.model.endswith(f":{self.lora_name}"):
                return self.model
            return f"{self.model}:{self.lora_name}"
        return self.model


class LLMClient:
    """OpenAI-compatible client for a separately running SGLang server."""

    def __init__(self, config: SGLangConfig | None = None) -> None:
        self.config = config or SGLangConfig()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def generate_repair(self, report: Report) -> str:
        response = self.client.chat.completions.create(
            model=self.config.request_model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": (
                        "작업:\n"
                        "다음 스캔 결과의 보안 취약점을 분석하고 안전한 코드로 수정하라.\n\n"
                        f"출력 형식:\n{RESPONSE_FORMAT}\n\n"
                        f"분석 대상:\n{report.to_finetuning_input()}"
                    ),
                },
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_review(self, report: Report) -> str:
        response = self.client.chat.completions.create(
            model=self.config.request_model,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": (
                        "작업:\n"
                        "다음 보안 스캔 결과와 수정 제안을 검토하라. "
                        "수정이 취약점을 실제로 완화하는지, 남은 위험과 검증 방법을 제시하라.\n\n"
                        f"출력 형식:\n{REVIEW_FORMAT}\n\n"
                        f"스캔 결과:\n{report.to_finetuning_input()}\n\n"
                        f"수정 제안:\n{report.repair or '없음'}"
                    ),
                },
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""
