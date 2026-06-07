from __future__ import annotations

from typing import Any


SYSTEM_MESSAGE = (
    "You are a secure coding assistant. Analyze vulnerable code and provide "
    "safe, practical fixes."
)

RESPONSE_FORMAT_INSTRUCTION = """응답은 반드시 다음 네 섹션을 포함해야 한다.
1. 취약점 설명
2. 안전한 수정 코드
3. 수정 이유
4. 추가 주의사항"""


def build_user_message(instruction: str, vulnerable_input: str) -> str:
    return (
        f"{instruction.strip()}\n\n"
        f"{RESPONSE_FORMAT_INSTRUCTION}\n\n"
        f"분석 대상 코드 또는 입력:\n{vulnerable_input.strip()}"
    )


def build_chat_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": build_user_message(example["instruction"], example["input"]),
        },
        {"role": "assistant", "content": example["output"].strip()},
    ]


def format_with_chat_template(example: dict[str, Any], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        build_chat_messages(example),
        tokenize=False,
        add_generation_prompt=False,
    )
