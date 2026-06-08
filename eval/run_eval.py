from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from defapi.models import ScanRecord
from defapi.workflow import ScanWorkflow

CASE_PATH = REPO_ROOT / "eval/cases/scan_cases.jsonl"

def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases

def scanner_statuses(report) -> dict[str, str]:
    return {
        result.scanner.value: result.status.value for result in report.scanner_results
    }

def finding_rule_ids(report) -> set[str]:
    return {
        finding.rule_id for result in report.scanner_results for finding in result.findings
    }

def check_case(case: dict[str, Any], report, elapsed_sec: float) -> list[str]:
    errors = []
    findings_total = report.summary.get("findings_total", 0)
    statuses = scanner_statuses(report)
    rule_ids = finding_rule_ids(report)

    if "min_findings_total" in case:
        if findings_total < case["min_findings_total"]:
            errors.append(
                f"findings_total={findings_total}, expected >= {case['min_findings_total']}"
            )

    if "max_findings_total" in case:
        if findings_total > case["max_findings_total"]:
            errors.append(
                f"findings_total={findings_total}, expected <= {case['max_findings_total']}"
            )
    
    for scanner in case.get("expected_scanners", []):
        if scanner not in statuses:
            errors.append(f"missing scanner result: {scanner}")

    for rule_id in case.get("expected_rule_ids", []):
        if rule_id not in rule_ids:
            errors.append(f"missing expected rule_id: {rule_id}")

    if "max_elapsed_sec" in case:
        if elapsed_sec > case["max_elapsed_sec"]:
            errors.append(
                f"elapsed_sec={elapsed_sec:.3f}, expected <= {case['max_elapsed_sec']}"
            )

    return errors


async def run_case(workflow: ScanWorkflow, case: dict[str, Any]) -> dict[str, Any]:
    target = Path(case["target"]).resolve()
    record = ScanRecord(
        target=str(target),
        include_zap=case.get("include_zap", False),
    )

    started = time.perf_counter()
    report = await workflow.run(record)
    elapsed_sec = time.perf_counter() - started

    errors = check_case(case, report, elapsed_sec)

    return {
        "name": case["name"],
        "target": str(target),
        "passed": not errors,
        "errors": errors,
        "elapsed_sec": round(elapsed_sec, 3),
        "findings_total": report.summary.get("findings_total", 0),
        "scanner_statuses": scanner_statuses(report),
    }


async def main() -> int:
    cases = load_cases(CASE_PATH)
    workflow = ScanWorkflow()

    results = []
    for case in cases:
        result = await run_case(workflow, case)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['name']} "
            f"findings={result['findings_total']} "
            f"elapsed={result['elapsed_sec']}s "
            f"scanners={result['scanner_statuses']}"
        )

        for error in result["errors"]:
            print(f"  - {error}")

    passed = sum(1 for item in results if item["passed"])
    total = len(results)

    print(f"\nResult: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
