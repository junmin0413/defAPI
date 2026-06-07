from __future__ import annotations

from typing import Any


SYSTEM_MESSAGE = (
    "You are a secure coding assistant. Analyze vulnerable code and provide "
    "safe, practical fixes. Return concise, actionable remediation guidance "
    "for security review reports."
)

RESPONSE_SECTIONS = (
    "취약점 설명",
    "안전한 수정 코드",
    "수정 이유",
    "추가 주의사항",
)


def build_response_format_instruction() -> str:
    numbered_sections = "\n".join(
        f"{index}. {section}" for index, section in enumerate(RESPONSE_SECTIONS, start=1)
    )
    return (
        "응답은 반드시 아래 네 섹션을 같은 순서로 포함해야 한다.\n"
        f"{numbered_sections}\n\n"
        "각 섹션 제목은 그대로 사용하고, 수정 코드는 fenced code block으로 작성하라. "
        "CWE, 언어, 취약 코드 정보가 제공되면 이를 근거로 분석하라."
    )


def build_user_message(instruction: str, vulnerable_input: str) -> str:
    return (
        f"작업:\n{instruction.strip()}\n\n"
        f"출력 형식:\n{build_response_format_instruction()}\n\n"
        f"분석 대상:\n{vulnerable_input.strip()}"
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
