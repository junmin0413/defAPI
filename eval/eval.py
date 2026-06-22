from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langsmith import Client, evaluate, traceable
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defapi.llm import SGLangConfig


CASES = [
    {
        "case_id": "command_injection",
        "prompt": 'Fix this vulnerable Python code and explain briefly:\n\n```python\nimport os\nname = input("name: ")\nos.system("echo " + name)\n```',
        "expected": ["subprocess", "command injection"],
    },
    {
        "case_id": "sql_injection",
        "prompt": "Fix this vulnerable Python code and explain briefly:\n\n```python\ndef find_user(conn, username):\n    query = \"SELECT id FROM users WHERE name = '\" + username + \"'\"\n    return conn.execute(query).fetchall()\n```",
        "expected": ["parameter", "sql injection"],
    },
    {
        "case_id": "eval_user_input",
        "prompt": 'Fix this vulnerable Python code and explain briefly:\n\n```python\nexpr = input("expr: ")\nprint(eval(expr))\n```',
        "expected": ["eval"],
    },
    {
        "case_id": "hardcoded_secret",
        "prompt": 'Fix this vulnerable Python code and explain briefly:\n\n```python\nPAYMENT_API_KEY = "sk_live_1234567890abcdef"\n\ndef headers():\n    return {"Authorization": f"Bearer {PAYMENT_API_KEY}"}\n```',
        "expected": ["secret", "environment"],
    },
    {
        "case_id": "path_traversal",
        "prompt": "Fix this vulnerable Python code and explain briefly:\n\n```python\nfrom pathlib import Path\nBASE_DIR = Path('/srv/uploads')\n\ndef read_file(name: str) -> str:\n    return (BASE_DIR / name).read_text()\n```",
        "expected": ["resolve", "path"],
    },
]

FORBIDDEN_MARKERS = ["작업:", "출력:", "CWE:", "Description:", "Language:", "Vulnerable code:"]
DATASET_NAME = os.getenv("LANGSMITH_DATASET_NAME", "defapi-repair-prompts")


def score_response(text: str, expected_terms: list[str]) -> dict[str, Any]:
    lowered = text.lower()
    expected_hits = [term for term in expected_terms if term.lower() in lowered]
    forbidden_hits = [marker for marker in FORBIDDEN_MARKERS if marker in text]
    has_code_block = "```" in text
    score = len(expected_hits)
    if has_code_block:
        score += 1
    score -= len(forbidden_hits) * 2
    return {
        "score": score,
        "has_code_block": has_code_block,
        "expected_hits": expected_hits,
        "forbidden_hits": forbidden_hits,
    }


def case_examples() -> list[dict[str, Any]]:
    return [
        {
            "inputs": {
                "case_id": case["case_id"],
                "prompt": case["prompt"],
                "expected": case["expected"],
            },
            "outputs": {
                "expected": case["expected"],
            },
            "metadata": {
                "case_id": case["case_id"],
            },
        }
        for case in CASES
    ]


def ensure_dataset(client: Client) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"LangSmith dataset exists: {DATASET_NAME}")
        return

    client.create_dataset(
        dataset_name=DATASET_NAME,
        description="DefAPI base vs LoRA security repair prompts.",
        metadata={"source": "eval/eval.py"},
    )
    client.create_examples(dataset_name=DATASET_NAME, examples=case_examples())
    print(f"Created LangSmith dataset: {DATASET_NAME}")


def make_target(config: SGLangConfig, variant: str):
    openai_client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    @traceable(name=f"defapi_repair_{variant}")
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        response = openai_client.chat.completions.create(
            model=config.request_model,
            messages=[{"role": "user", "content": inputs["prompt"]}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        text = response.choices[0].message.content or ""
        return {
            "response": text,
            "variant": variant,
            "request_model": config.request_model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "checks": score_response(text, inputs["expected"]),
        }

    return target


def repair_checks(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    checks = outputs.get("checks") or score_response(outputs.get("response", ""), inputs["expected"])
    return [
        {
            "key": "repair_score",
            "score": checks["score"],
            "comment": json.dumps(checks, ensure_ascii=False),
        },
        {
            "key": "has_code_block",
            "score": bool(checks["has_code_block"]),
        },
        {
            "key": "no_dataset_field_leak",
            "score": not bool(checks["forbidden_hits"]),
            "comment": ", ".join(checks["forbidden_hits"]),
        },
    ]


def compare_base_lora(
    inputs: dict[str, Any],
    outputs: list[dict[str, Any]],
    reference_outputs: dict[str, Any] | None = None,
    runs: list[Any] | None = None,
) -> dict[str, Any]:
    scores = [int((output.get("checks") or {}).get("score", 0)) for output in outputs]
    if scores[0] > scores[1]:
        pair_scores = [1, 0]
        winner = "A"
    elif scores[1] > scores[0]:
        pair_scores = [0, 1]
        winner = "B"
    else:
        pair_scores = [0.5, 0.5]
        winner = "tie"

    if runs:
        score_payload: dict[str, float] | list[float] = {
            str(runs[0].id): pair_scores[0],
            str(runs[1].id): pair_scores[1],
        }
    else:
        score_payload = pair_scores

    return {
        "key": "base_vs_lora_repair_score",
        "scores": score_payload,
        "comment": json.dumps(
            {"winner": winner, "scores": scores, "case_id": inputs.get("case_id")},
            ensure_ascii=False,
        ),
    }


def experiment_ref(result: Any) -> str:
    return str(getattr(result, "experiment_name", None) or getattr(result, "id", None) or result)


def main() -> int:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is required for eval/eval.py")

    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    client = Client()
    ensure_dataset(client)

    base_config = SGLangConfig(lora_name=None)
    lora_config = SGLangConfig()

    base_result = evaluate(
        make_target(base_config, "base"),
        data=DATASET_NAME,
        evaluators=[repair_checks],
        experiment_prefix=f"{run_label}-base",
        description=f"DefAPI base repair eval: {base_config.request_model}",
        metadata={"variant": "base", "request_model": base_config.request_model},
        client=client,
        blocking=True,
        max_concurrency=1,
    )
    lora_result = evaluate(
        make_target(lora_config, "lora"),
        data=DATASET_NAME,
        evaluators=[repair_checks],
        experiment_prefix=f"{run_label}-lora",
        description=f"DefAPI LoRA repair eval: {lora_config.request_model}",
        metadata={"variant": "lora", "request_model": lora_config.request_model},
        client=client,
        blocking=True,
        max_concurrency=1,
    )

    base_ref = experiment_ref(base_result)
    lora_ref = experiment_ref(lora_result)
    pairwise_result = evaluate(
        (base_ref, lora_ref),
        evaluators=[compare_base_lora],
        experiment_prefix=f"{run_label}-base-vs-lora",
        description="DefAPI base vs LoRA deterministic repair comparison.",
        metadata={"base": base_config.request_model, "lora": lora_config.request_model},
        client=client,
        blocking=True,
        max_concurrency=1,
    )
    pairwise_ref = experiment_ref(pairwise_result)

    output = {
        "run_label": run_label,
        "dataset_name": DATASET_NAME,
        "models": {
            "base": base_config.request_model,
            "lora": lora_config.request_model,
        },
        "experiments": {
            "base": base_ref,
            "lora": lora_ref,
            "base_vs_lora": pairwise_ref,
        },
    }
    output_dir = PROJECT_ROOT / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_label}_langsmith_eval.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
