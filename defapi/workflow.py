from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from defapi.mcp import SemgrepMCP, TrivyMCP
from defapi.models import Finding, Report, ScanRecord, ScannerResult
from defapi.reports import ReportGenerator


class ScanState(TypedDict, total=False):
    record: ScanRecord
    target: Path
    scanner_results: list[ScannerResult]
    findings: list[Finding]
    report: Report


class ScanWorkflow:
    def __init__(self) -> None:
        self.semgrep = SemgrepMCP()
        self.trivy = TrivyMCP()
        self.report_generator = ReportGenerator()
        self.graph = self._build_graph()

    async def run(self, record: ScanRecord) -> Report:
        state = await self.graph.ainvoke({"record": record, "target": Path(record.target)})
        return state["report"]

    def _build_graph(self):
        graph = StateGraph(ScanState)
        graph.add_node("scan", self._scan)
        graph.add_node("report", self._report)
        graph.set_entry_point("scan")
        graph.add_edge("scan", "report")
        graph.add_edge("report", END)
        return graph.compile()

    async def _scan(self, state: ScanState) -> ScanState:
        target = state["target"]
        scanners = [self.semgrep.scan(target), self.trivy.scan(target)]
        scanner_results = await asyncio.gather(*scanners)
        findings = [finding for result in scanner_results for finding in result.findings]
        return {**state, "scanner_results": scanner_results, "findings": findings}

    async def _report(self, state: ScanState) -> ScanState:
        report = self.report_generator.build(
            state["record"],
            state.get("scanner_results", []),
        )
        return {**state, "report": report}
