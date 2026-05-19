from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from defapi.mcp import SemgrepMCP, TrivyMCP, ZapMCP
from defapi.models import Finding, Report, ScanRecord, ScannerResult
from defapi.models import PatchSuggestion, ValidationResult
from defapi.patches import PatchGenerator
from defapi.reports import ReportGenerator
from defapi.validation import ValidationLoop


class ScanState(TypedDict, total=False):
    record: ScanRecord
    target: Path
    scanner_results: list[ScannerResult]
    findings: list[Finding]
    patches: list[PatchSuggestion]
    validation: list[ValidationResult]
    report: Report


class ScanWorkflow:
    def __init__(self) -> None:
        self.semgrep = SemgrepMCP()
        self.trivy = TrivyMCP()
        self.zap = ZapMCP()
        self.patch_generator = PatchGenerator()
        self.validation_loop = ValidationLoop()
        self.report_generator = ReportGenerator()
        self.graph = self._build_graph()

    async def run(self, record: ScanRecord) -> Report:
        state = await self.graph.ainvoke({"record": record, "target": Path(record.target)})
        return state["report"]

    def _build_graph(self):
        graph = StateGraph(ScanState)
        graph.add_node("scan", self._scan)
        graph.add_node("patch", self._patch)
        graph.add_node("validate", self._validate)
        graph.add_node("report", self._report)
        graph.set_entry_point("scan")
        graph.add_edge("scan", "patch")
        graph.add_edge("patch", "validate")
        graph.add_edge("validate", "report")
        graph.add_edge("report", END)
        return graph.compile()

    async def _scan(self, state: ScanState) -> ScanState:
        record = state["record"]
        target = state["target"]
        scanners = [self.semgrep.scan(target), self.trivy.scan(target)]
        if record.include_zap:
            scanners.append(self.zap.scan(target))
        scanner_results = await asyncio.gather(*scanners)
        findings = [finding for result in scanner_results for finding in result.findings]
        return {**state, "scanner_results": scanner_results, "findings": findings}

    async def _patch(self, state: ScanState) -> ScanState:
        patches = self.patch_generator.generate(state["target"], state.get("findings", []))
        return {**state, "patches": patches}

    async def _validate(self, state: ScanState) -> ScanState:
        validation = self.validation_loop.validate(state["target"], state.get("patches", []))
        return {**state, "validation": validation}

    async def _report(self, state: ScanState) -> ScanState:
        report = self.report_generator.build(
            state["record"],
            state.get("scanner_results", []),
            state.get("patches", []),
            state.get("validation", []),
        )
        return {**state, "report": report}
