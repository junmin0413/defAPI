from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from defapi.llm import LLMClient
from defapi.mcp import CodeQLMCP, CommandMCP, SemgrepMCP, TrivyMCP
from defapi.models import AgentStep, Finding, Report, ScannerResult
from defapi.reports import ReportGenerator


class BaseAgent:
    name: str

    def _step(
        self,
        *,
        status: str,
        summary: str,
        started_at: datetime,
        error: str | None = None,
    ) -> AgentStep:
        return AgentStep(
            name=self.name,
            status=status,
            summary=summary,
            error=error,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )


class ScannerAgent(BaseAgent):
    def __init__(self, scanner: CommandMCP) -> None:
        self.scanner = scanner
        self.name = f"{scanner.scanner.value}_scanner_agent"

    async def run(self, target: Path) -> tuple[ScannerResult, AgentStep]:
        started_at = datetime.now(timezone.utc)
        result = await self.scanner.scan(target)
        summary = (
            f"{result.scanner.value} finished with {len(result.findings)} finding(s)"
            if result.status == "completed"
            else f"{result.scanner.value} {result.status.value}"
        )
        return result, self._step(
            status=result.status.value,
            summary=summary,
            started_at=started_at,
            error=result.error,
        )


class ScanCoordinatorAgent(BaseAgent):
    name = "scan_coordinator_agent"

    def __init__(self, scanner_agents: list[ScannerAgent] | None = None) -> None:
        self.scanner_agents = scanner_agents or [
            ScannerAgent(SemgrepMCP()),
            ScannerAgent(TrivyMCP()),
            ScannerAgent(CodeQLMCP()),
        ]

    async def run(self, target: Path) -> tuple[list[ScannerResult], list[Finding], list[AgentStep]]:
        started_at = datetime.now(timezone.utc)
        pairs = await asyncio.gather(*(agent.run(target) for agent in self.scanner_agents))
        scanner_results = [result for result, _step in pairs]
        scanner_steps = [_step for _result, _step in pairs]
        findings = [finding for result in scanner_results for finding in result.findings]
        coordinator_step = self._step(
            status="completed",
            summary=f"coordinated {len(scanner_results)} scanner(s), collected {len(findings)} finding(s)",
            started_at=started_at,
        )
        return scanner_results, findings, [coordinator_step, *scanner_steps]


class ReportAgent(BaseAgent):
    name = "report_agent"

    def __init__(self, report_generator: ReportGenerator | None = None) -> None:
        self.report_generator = report_generator or ReportGenerator()

    async def run(self, record, scanner_results: list[ScannerResult]) -> tuple[Report, AgentStep]:
        started_at = datetime.now(timezone.utc)
        report = self.report_generator.build(record, scanner_results)
        return report, self._step(
            status="completed",
            summary=f"built report with {report.summary.get('findings_total', 0)} finding(s)",
            started_at=started_at,
        )


class RepairAgent(BaseAgent):
    name = "repair_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    async def run(self, report: Report) -> tuple[Report, AgentStep]:
        started_at = datetime.now(timezone.utc)
        if report.summary.get("findings_total", 0) == 0:
            return report, self._step(
                status="skipped",
                summary="no findings to repair",
                started_at=started_at,
            )

        try:
            repair_text = await asyncio.to_thread(self.llm.generate_repair, report)
        except Exception as exc:
            return report, self._step(
                status="failed",
                summary="repair generation failed",
                started_at=started_at,
                error=str(exc),
            )

        repaired_report = report.model_copy(update={"repair": repair_text})
        return repaired_report, self._step(
            status="completed",
            summary="generated repair guidance",
            started_at=started_at,
        )


class ReviewAgent(BaseAgent):
    name = "review_agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    async def run(self, report: Report) -> tuple[Report, AgentStep]:
        started_at = datetime.now(timezone.utc)
        if not report.repair:
            return report, self._step(
                status="skipped",
                summary="no repair guidance to review",
                started_at=started_at,
            )

        try:
            review_text = await asyncio.to_thread(self.llm.generate_review, report)
        except Exception as exc:
            return report, self._step(
                status="failed",
                summary="repair review failed",
                started_at=started_at,
                error=str(exc),
            )

        reviewed_report = report.model_copy(update={"review": review_text})
        return reviewed_report, self._step(
            status="completed",
            summary="reviewed repair guidance",
            started_at=started_at,
        )
