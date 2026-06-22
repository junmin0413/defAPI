from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from defapi.mcp import CodeQLMCP, SemgrepMCP, TrivyMCP
from defapi.models import Finding, Report, ScanRecord, ScannerResult
from defapi.reports import ReportGenerator
from defapi.llm import LLMClient


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
        self.codeql = CodeQLMCP()
        self.report_generator = ReportGenerator()
        self.llm = LLMClient()
        self.graph = self._build_graph()

    async def run(self, record: ScanRecord) -> Report:
        state = await self.graph.ainvoke({"record": record, "target": Path(record.target)})
        return state["report"]

    def _build_graph(self):
        # LangGraph 노드는 단순합니다: 스캐너들을 병렬 실행한 뒤 보고서를 만듭니다.
        graph = StateGraph(ScanState)
        graph.add_node("scan", self._scan)
        graph.add_node("report", self._report)
        graph.add_node("repair", self._repair)

        graph.set_entry_point("scan")
        graph.add_edge("scan", "report")
        graph.add_edge("report", "repair")
        graph.add_edge("repair", END)
        return graph.compile()

    async def _scan(self, state: ScanState) -> ScanState:
        target = state["target"]
        # 각 MCP는 실패/미설치 상태를 ScannerResult로 감싸서 반환하므로 gather가 전체 workflow를 유지합니다.
        scanners = [
            self.semgrep.scan(target),
            self.trivy.scan(target),
            self.codeql.scan(target),
        ]
        scanner_results = await asyncio.gather(*scanners)
        findings = [finding for result in scanner_results for finding in result.findings]
        return {**state, "scanner_results": scanner_results, "findings": findings}

    async def _report(self, state: ScanState) -> ScanState:
        report = self.report_generator.build(
            state["record"],
            state.get("scanner_results", []),
        )
        return {**state, "report": report}
    
    async def _repair(self, state: ScanState) -> ScanState:
        report = state["report"]

        if report.summary.get("findings_total", 0) == 0:
            return state
        
        repair_text = await asyncio.to_thread(self.llm.generate_repair, report)
        repaired_report = report.model_copy(update={"repair": repair_text})
        return {**state, "report": repaired_report}
