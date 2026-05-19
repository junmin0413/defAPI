from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from defapi.mcp import SemgrepMCP, TrivyMCP, ZapMCP
from defapi.models import Finding, Report, ScanRecord, ScannerResult
from defapi.models import PatchSuggestion, ValidationResult
from defapi.remediation import RemediationVerifier, create_default_remediator
from defapi.reports import ReportGenerator


class ScanState(TypedDict, total=False):
    record: ScanRecord
    target: Path
    scanner_results: list[ScannerResult]
    findings: list[Finding]
    initial_report: Report
    patches: list[PatchSuggestion]
    validation: list[ValidationResult]
    verification_scanner_results: list[ScannerResult]
    report: Report


class ScanWorkflow:
    def __init__(self) -> None:
        self.semgrep = SemgrepMCP()
        self.trivy = TrivyMCP()
        self.zap = ZapMCP()
        self.report_generator = ReportGenerator()
        self.remediator = create_default_remediator()
        self.verifier = RemediationVerifier(semgrep=self.semgrep, trivy=self.trivy, zap=self.zap)
        self.graph = self._build_graph()

    async def run(self, record: ScanRecord) -> Report:
        state = await self.graph.ainvoke({"record": record, "target": Path(record.target)})
        return state["report"]

    def _build_graph(self):
        graph = StateGraph(ScanState)
        graph.add_node("scan", self._scan)
        graph.add_node("initial_report", self._initial_report)
        graph.add_node("remediate", self._remediate)
        graph.add_node("verify", self._verify)
        graph.add_node("final_report", self._final_report)
        graph.set_entry_point("scan")
        graph.add_edge("scan", "initial_report")
        graph.add_edge("initial_report", "remediate")
        graph.add_edge("remediate", "verify")
        graph.add_edge("verify", "final_report")
        graph.add_edge("final_report", END)
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

    async def _initial_report(self, state: ScanState) -> ScanState:
        initial_report = self.report_generator.build(
            state["record"],
            state.get("scanner_results", []),
            [],
            [],
        )
        return {**state, "initial_report": initial_report}

    async def _remediate(self, state: ScanState) -> ScanState:
        patches = await asyncio.to_thread(
            self.remediator.generate,
            state["target"],
            state["initial_report"],
            state.get("findings", []),
        )
        return {**state, "patches": patches}

    async def _verify(self, state: ScanState) -> ScanState:
        validation, verification_scanner_results = await self.verifier.validate_and_rescan(
            state["target"],
            state["record"],
            state.get("patches", []),
        )
        return {**state, "validation": validation, "verification_scanner_results": verification_scanner_results}

    async def _final_report(self, state: ScanState) -> ScanState:
        report = self.report_generator.build(
            state["record"],
            state.get("scanner_results", []),
            state.get("patches", []),
            state.get("validation", []),
            state.get("verification_scanner_results", []),
        )
        return {**state, "report": report}
