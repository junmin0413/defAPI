from __future__ import annotations

from defapi.models import Finding, Report


class RemediationPromptBuilder:
    def build(self, report: Report, finding: Finding, code_context: str) -> str:
        return (
            "You are a secure code remediation model.\n"
            "Return only a unified diff. Do not include markdown fences or unrelated edits.\n\n"
            "[Security Report]\n"
            f"scan_id: {report.scan_id}\n"
            f"target: {report.target}\n"
            f"findings_total: {report.summary.get('findings_total', 0)}\n\n"
            "[Finding]\n"
            f"scanner: {finding.scanner.value}\n"
            f"severity: {finding.severity.value}\n"
            f"rule_id: {finding.rule_id}\n"
            f"title: {finding.title}\n"
            f"message: {finding.message}\n"
            f"file: {finding.file_path or 'unknown'}\n"
            f"line: {finding.start_line or 'unknown'}\n\n"
            "[Vulnerable Code Context]\n"
            f"{code_context or 'No local code context is available.'}\n\n"
            "[Task]\n"
            "Generate the smallest safe unified diff that fixes this finding.\n"
        )
