from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PHOENIX_WORKING_DIR", str(ROOT / ".phoenix"))

from defapi.models import ScanRecord
from defapi.workflow import ScanWorkflow

DEFAULT_BASE_URL = "https://app.phoenix.arize.com/s/wkdwnsals0413"
CASES = ROOT / "eval/cases/scan_cases.jsonl"

JUDGE_PROMPT = """
You are judging a security scan report.

Case: {{case_name}}
Description: {{description}}
Expected scanners: {{expected_scanners}}
Minimum findings: {{min_findings_total}}
Report JSON: {{report_json}}

PASS only when the report satisfies the case expectation. Clean cases need no
findings. Vulnerable cases need at least one relevant finding. Scanner failures,
ambiguous evidence, or irrelevant-only findings are FAIL.
"""


def phoenix_imports():
    try:
        from phoenix.client import Client
        from phoenix.evals import ClassificationEvaluator, LLM
    except Exception as exc:
        raise SystemExit(
            "Install Arize Phoenix packages and remove the old 'phoenix' package:\n"
            "python -m pip uninstall -y phoenix\n"
            "python -m pip install arize-phoenix-client arize-phoenix-evals openai"
        ) from exc
    return Client, ClassificationEvaluator, LLM


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if case := line.strip():
            data = json.loads(case)
            rows.append(
                {
                    "case_name": data["name"],
                    "target": data["target"],
                    "description": data.get("description", ""),
                    "min_findings_total": data.get("min_findings_total", 0),
                    "max_findings_total": data.get("max_findings_total"),
                    "expected_scanners": ",".join(data.get("expected_scanners", [])),
                }
            )
    return rows


def compact_report(report: Any) -> dict[str, Any]:
    data = report.model_dump(mode="json")
    results = data.get("scanner_results", [])
    return {
        "target": data["target"],
        "status": data["status"],
        "summary": data.get("summary", {}),
        "scanner_statuses": {r["scanner"]: r["status"] for r in results},
        "scanner_errors": {r["scanner"]: r["error"] for r in results if r.get("error")},
        "findings": [
            {
                "scanner": f.get("scanner"),
                "rule_id": f.get("rule_id"),
                "severity": f.get("severity"),
                "title": f.get("title"),
                "message": f.get("message"),
                "file_path": f.get("file_path"),
                "start_line": f.get("start_line"),
                "cwe": f.get("cwe", []),
            }
            for r in results
            for f in r.get("findings", [])
        ],
    }


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    report = await ScanWorkflow().run(ScanRecord(target=str((ROOT / case["target"]).resolve())))
    compact = compact_report(report)
    return {**case, **compact, "report_json": json.dumps(compact, ensure_ascii=False)}


def task(input: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(run_case(input))


def finding_count_bounds(output: dict[str, Any], input: dict[str, Any]) -> tuple[bool, str, str]:
    total = output.get("summary", {}).get("findings_total", 0)
    lower = input.get("min_findings_total", 0)
    upper = input.get("max_findings_total")
    ok = total >= lower and (upper is None or total <= upper)
    return ok, "pass" if ok else "fail", f"findings_total={total}, expected={lower}..{upper}"


def expected_scanners_completed(output: dict[str, Any], input: dict[str, Any]) -> tuple[bool, str, str]:
    expected = [s for s in input.get("expected_scanners", "").split(",") if s]
    statuses = output.get("scanner_statuses", {})
    failed = [s for s in expected if statuses.get(s) != "completed"]
    return not failed, "pass" if not failed else "fail", f"failed={failed}, statuses={statuses}"


def llm_judge(model: str):
    _, ClassificationEvaluator, LLM = phoenix_imports()
    evaluator = ClassificationEvaluator(
        name="llm_report_quality",
        prompt_template=JUDGE_PROMPT,
        llm=LLM(provider="openai", model=model),
        choices={"FAIL": 0, "PASS": 1},
        include_explanation=True,
    )

    def judge(output: dict[str, Any]) -> tuple[float | int | None, str | None, str | None]:
        score = evaluator.evaluate(output)[0]
        return score.score, score.label, score.explanation

    judge.__name__ = "llm_report_quality"
    return judge


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--dataset-name", default="defapi-security-scan-cases")
    parser.add_argument("--experiment-name", default="defapi-scan-llm-judge")
    parser.add_argument("--phoenix-base-url", default=os.getenv("PHOENIX_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--phoenix-api-key", default=os.getenv("PHOENIX_API_KEY"))
    parser.add_argument("--judge-model", default=os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini"))
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--dry-run", nargs="?", const=1, type=int)
    return parser.parse_args()


def main() -> None:
    opts = args()
    if not opts.phoenix_api_key:
        raise SystemExit("PHOENIX_API_KEY is required for Phoenix Cloud.")

    Client, _, _ = phoenix_imports()
    client = Client(base_url=opts.phoenix_base_url, api_key=opts.phoenix_api_key)
    try:
        dataset = client.datasets.create_dataset(
            name=opts.dataset_name,
            inputs=load_cases(opts.cases),
            dataset_description="defAPI security scanner fixture eval cases",
        )
    except Exception as exc:
        raise SystemExit(
            "Phoenix dataset upload failed. Check that PHOENIX_API_KEY belongs "
            f"to this space: {opts.phoenix_base_url}"
        ) from exc

    evaluators = [finding_count_bounds, expected_scanners_completed]
    if not opts.no_llm_judge:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required unless --no-llm-judge is set.")
        evaluators.append(llm_judge(opts.judge_model))

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=opts.experiment_name,
        dry_run=opts.dry_run or False,
    )
    print(experiment)


if __name__ == "__main__":
    main()
