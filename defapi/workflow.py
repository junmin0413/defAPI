from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from defapi.agents import ReportAgent, RepairAgent, ReviewAgent, ScanCoordinatorAgent
from defapi.models import AgentStep, Finding, Report, ScanRecord, ScannerResult


class ScanState(TypedDict, total=False):
    record: ScanRecord
    target: Path
    scanner_results: list[ScannerResult]
    findings: list[Finding]
    report: Report
    agent_steps: list[AgentStep]


class ScanWorkflow:
    def __init__(self) -> None:
        self.scan_coordinator = ScanCoordinatorAgent()
        self.report_agent = ReportAgent()
        self.repair_agent = RepairAgent()
        self.review_agent = ReviewAgent()
        self.graph = self._build_graph()

    async def run(self, record: ScanRecord) -> Report:
        state = await self.graph.ainvoke(
            {"record": record, "target": Path(record.target), "agent_steps": []}
        )
        return state["report"]

    def _build_graph(self):
        # LangGraph는 agent 간 handoff만 담당하고, 각 책임은 agent class에 둡니다.
        graph = StateGraph(ScanState)
        graph.add_node("scan", self._scan)
        graph.add_node("report", self._report)
        graph.add_node("repair", self._repair)
        graph.add_node("review", self._review)

        graph.set_entry_point("scan")
        graph.add_edge("scan", "report")
        graph.add_edge("report", "repair")
        graph.add_edge("repair", "review")
        graph.add_edge("review", END)
        return graph.compile()

    async def _scan(self, state: ScanState) -> ScanState:
        scanner_results, findings, agent_steps = await self.scan_coordinator.run(state["target"])
        return {
            **state,
            "scanner_results": scanner_results,
            "findings": findings,
            "agent_steps": [*state.get("agent_steps", []), *agent_steps],
        }

    async def _report(self, state: ScanState) -> ScanState:
        report, agent_step = await self.report_agent.run(
            state["record"],
            state.get("scanner_results", []),
        )
        agent_steps = [*state.get("agent_steps", []), agent_step]
        report = report.model_copy(update={"agent_steps": agent_steps})
        return {**state, "report": report, "agent_steps": agent_steps}

    async def _repair(self, state: ScanState) -> ScanState:
        report, agent_step = await self.repair_agent.run(state["report"])
        agent_steps = [*state.get("agent_steps", []), agent_step]
        report = report.model_copy(update={"agent_steps": agent_steps})
        return {**state, "report": report, "agent_steps": agent_steps}

    async def _review(self, state: ScanState) -> ScanState:
        report, agent_step = await self.review_agent.run(state["report"])
        agent_steps = [*state.get("agent_steps", []), agent_step]
        report = report.model_copy(update={"agent_steps": agent_steps})
        return {**state, "report": report, "agent_steps": agent_steps}
