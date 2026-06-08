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
    "safe, practical fixes."
)

RESPONSE_FORMAT_INSTRUCTION = """응답은 반드시 다음 네 섹션을 포함해야 한다.
1. 취약점 설명
2. 안전한 수정 코드
3. 수정 이유
4. 추가 주의사항"""


def safe_strip(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_instruction(example: dict[str, Any]) -> str:
    cwe_id = safe_strip(example.get("cwe_id"))
    cwe_description = safe_strip(example.get("cwe_description"))
    language = safe_strip(example.get("language"))

    return (
        f"다음 {language} 코드의 보안 취약점을 분석하고 안전하게 수정해라.\n"
        f"CWE ID: {cwe_id}\n"
        f"CWE 설명: {cwe_description}"
    ).strip()


def build_user_message(example: dict[str, Any]) -> str:
    vulnerable_code = safe_strip(example.get("vulnerable_code"))
    instruction = build_instruction(example)

    return (
        f"{instruction}\n\n"
        f"{RESPONSE_FORMAT_INSTRUCTION}\n\n"
        f"분석 대상 코드:\n"
        f"```{safe_strip(example.get('language_dir')) or safe_strip(example.get('language'))}\n"
        f"{vulnerable_code}\n"
        f"```"
    ).strip()


def build_assistant_message(example: dict[str, Any]) -> str:
    cwe_id = safe_strip(example.get("cwe_id"))
    cwe_description = safe_strip(example.get("cwe_description"))
    fixed_code = safe_strip(example.get("fixed_code"))
    language = safe_strip(example.get("language_dir")) or safe_strip(example.get("language"))

    return f"1. 취약점 설명\n이 코드는 {cwe_id} 유형의 취약점과 관련이 있다. {cwe_description}\n\n2. 안전한 수정 코드\n{language}\n{fixed_code}"


def required_text(example: Mapping[str, Any], field: str) -> str:
    value = example.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"데이터셋 샘플 필드는 비어 있지 않은 문자열이어야 합니다: {field}")
    return value.strip()


def build_defapi_user_message(instruction: str, vulnerable_input: str) -> str:
    return (
        f"작업:\n{instruction.strip()}\n\n"
        f"출력 형식:\n{RESPONSE_FORMAT_INSTRUCTION}\n\n"
        f"분석 대상:\n{vulnerable_input.strip()}"
    )


def build_chat_messages(example: Mapping[str, Any]) -> list[ChatMessage]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": build_defapi_user_message(
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
