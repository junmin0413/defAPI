from __future__ import annotations

from pathlib import Path
from typing import Any

from defapi.mcp.base import CommandMCP
from defapi.models import Finding, FindingSeverity, ScannerName


class SemgrepMCP(CommandMCP):
    scanner = ScannerName.semgrep
    executable = "semgrep"

    @property
    def accepted_return_codes(self) -> set[int]:
        return {0, 1}

    def command(self, target: Path) -> list[str]:
        rules = Path(__file__).resolve().parents[2] / "eval/semgrep_rules.yml"
        # 로컬 커스텀 룰이 있으면 우선 사용하고, 없으면 Semgrep Registry의 auto 설정을 씁니다.
        config = str(rules) if rules.exists() else "auto"
        return ["semgrep", "--config", config, "--json", "--quiet", "--metrics", "off", str(target)]

    def parse_findings(self, payload: dict[str, Any]) -> list[Finding]:
        findings: list[Finding] = []
        for item in payload.get("results", []):
            # Semgrep JSON의 위치/메타데이터 필드를 DefAPI 공통 Finding 형태로 접습니다.
            extra = item.get("extra", {})
            metadata = extra.get("metadata", {})
            start = item.get("start", {})
            end = item.get("end", {})
            findings.append(
                Finding(
                    scanner=ScannerName.semgrep,
                    rule_id=str(item.get("check_id", "semgrep.unknown")),
                    severity=self._severity(extra.get("severity")),
                    title=str(metadata.get("shortlink") or item.get("check_id") or "Semgrep finding"),
                    message=str(extra.get("message") or "Semgrep reported a code issue"),
                    file_path=item.get("path"),
                    start_line=start.get("line"),
                    end_line=end.get("line"),
                    cwe=self._list(metadata.get("cwe")),
                    references=self._list(metadata.get("references")),
                    raw=item,
                )
            )
        return findings

    def _severity(self, value: Any) -> FindingSeverity:
        normalized = str(value or "info").lower()
        if normalized == "error":
            return FindingSeverity.high

        if normalized == "warning":
            return FindingSeverity.medium

        if normalized in FindingSeverity.__members__:
            return FindingSeverity(normalized)

        return FindingSeverity.info

    def _list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
