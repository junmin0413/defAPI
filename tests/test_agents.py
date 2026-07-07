from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from defapi.agents import RepairAgent, ReviewAgent, ScanCoordinatorAgent, ScannerAgent
from defapi.models import (
    Finding,
    FindingSeverity,
    Report,
    ScanStatus,
    ScannerName,
    ScannerResult,
)


class FakeScanner:
    scanner = ScannerName.semgrep
    executable = "fake"

    async def scan(self, target: Path) -> ScannerResult:
        return ScannerResult(
            scanner=self.scanner,
            status="completed",
            findings=[
                Finding(
                    scanner=self.scanner,
                    rule_id="fake.rule",
                    severity=FindingSeverity.high,
                    title="Fake finding",
                    message=f"Found in {target}",
                )
            ],
        )


class FailingLLM:
    def generate_repair(self, report: Report) -> str:
        raise RuntimeError("llm unavailable")

    def generate_review(self, report: Report) -> str:
        raise RuntimeError("llm unavailable")


class StaticLLM:
    def generate_repair(self, report: Report) -> str:
        return "repair text"

    def generate_review(self, report: Report) -> str:
        return "review text"


def _report(findings_total: int = 1, repair: str | None = None) -> Report:
    return Report(
        scan_id="scan-1",
        target="/tmp/project",
        status=ScanStatus.completed,
        created_at=datetime.now(timezone.utc),
        summary={"findings_total": findings_total},
        repair=repair,
    )


def test_scan_coordinator_collects_findings_and_agent_steps(tmp_path: Path) -> None:
    coordinator = ScanCoordinatorAgent(scanner_agents=[ScannerAgent(FakeScanner())])

    scanner_results, findings, steps = asyncio.run(coordinator.run(tmp_path))

    assert len(scanner_results) == 1
    assert len(findings) == 1
    assert [step.name for step in steps] == [
        "scan_coordinator_agent",
        "semgrep_scanner_agent",
    ]


def test_repair_agent_records_failure_without_dropping_report() -> None:
    report = _report()
    repaired_report, step = asyncio.run(RepairAgent(llm=FailingLLM()).run(report))

    assert repaired_report is report
    assert step.status == "failed"
    assert step.error == "llm unavailable"


def test_repair_and_review_agents_attach_outputs() -> None:
    report = _report()

    repaired_report, repair_step = asyncio.run(RepairAgent(llm=StaticLLM()).run(report))
    reviewed_report, review_step = asyncio.run(ReviewAgent(llm=StaticLLM()).run(repaired_report))

    assert repaired_report.repair == "repair text"
    assert reviewed_report.review == "review text"
    assert repair_step.status == "completed"
    assert review_step.status == "completed"
