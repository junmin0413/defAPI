from __future__ import annotations

from typing import Any, Mapping, Protocol, TypedDict, cast


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[ChatMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        ...


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


def required_text(example: Mapping[str, Any], field: str) -> str:
    value = example.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset sample field must be a non-empty string: {field}")
    return value.strip()


def build_chat_messages(example: Mapping[str, Any]) -> list[ChatMessage]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": build_user_message(
                required_text(example, "instruction"),
                required_text(example, "input"),
            ),
        },
        {"role": "assistant", "content": required_text(example, "output")},
    ]


def format_with_chat_template(example: Mapping[str, Any], tokenizer: ChatTemplateTokenizer) -> str:
    formatted = tokenizer.apply_chat_template(
        build_chat_messages(example),
        tokenize=False,
        add_generation_prompt=False,
    )
    return cast(str, formatted)
