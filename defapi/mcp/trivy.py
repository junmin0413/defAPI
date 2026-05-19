from __future__ import annotations

from pathlib import Path
from typing import Any

from defapi.mcp.base import CommandMCP
from defapi.models import Finding, FindingSeverity, ScannerName


class TrivyMCP(CommandMCP):
    scanner = ScannerName.trivy
    executable = "trivy"

    @property
    def accepted_return_codes(self) -> set[int]:
        return {0}

    def command(self, target: Path) -> list[str]:
        return ["trivy", "fs", "--format", "json", "--quiet", str(target)]

    def parse_findings(self, payload: dict[str, Any]) -> list[Finding]:
        findings: list[Finding] = []
        for result in payload.get("Results", []):
            file_path = result.get("Target")
            for vuln in result.get("Vulnerabilities", []) or []:
                findings.append(
                    Finding(
                        scanner=ScannerName.trivy,
                        rule_id=str(vuln.get("VulnerabilityID", "trivy.unknown")),
                        severity=self._severity(vuln.get("Severity")),
                        title=str(vuln.get("Title") or vuln.get("PkgName") or "Trivy vulnerability"),
                        message=str(vuln.get("Description") or "Trivy reported a vulnerable dependency"),
                        file_path=file_path,
                        cwe=[str(item) for item in vuln.get("CweIDs", []) or []],
                        references=[str(item) for item in vuln.get("References", []) or []],
                        raw=vuln,
                    )
                )
            for misconf in result.get("Misconfigurations", []) or []:
                findings.append(
                    Finding(
                        scanner=ScannerName.trivy,
                        rule_id=str(misconf.get("ID", "trivy.misconfiguration")),
                        severity=self._severity(misconf.get("Severity")),
                        title=str(misconf.get("Title") or "Trivy misconfiguration"),
                        message=str(misconf.get("Message") or misconf.get("Description") or "Trivy reported a misconfiguration"),
                        file_path=file_path,
                        start_line=misconf.get("CauseMetadata", {}).get("StartLine"),
                        end_line=misconf.get("CauseMetadata", {}).get("EndLine"),
                        references=[str(item) for item in misconf.get("References", []) or []],
                        raw=misconf,
                    )
                )
        return findings

    def _severity(self, value: Any) -> FindingSeverity:
        normalized = str(value or "info").lower()
        if normalized in FindingSeverity.__members__:
            return FindingSeverity(normalized)
        return FindingSeverity.info

