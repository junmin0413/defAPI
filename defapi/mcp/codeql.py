from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from defapi.mcp.base import CommandMCP
from defapi.models import Finding, FindingSeverity, ScannerName, ScannerResult


class CodeQLMCP(CommandMCP):
    scanner = ScannerName.codeql
    executable = "codeql"
    command_timeout_seconds = 600

    async def scan(self, target: Path) -> ScannerResult:
        started_at = datetime.now(timezone.utc)
        executable = self.executable_path()
        if executable is None:
            return ScannerResult(
                scanner=self.scanner,
                status="skipped",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error="codeql executable is not installed",
            )

        language = self._infer_language(target)
        if language is None:
            return ScannerResult(
                scanner=self.scanner,
                status="skipped",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error="CodeQL currently supports Python/JavaScript/TypeScript targets in this MVP",
            )

        try:
            with tempfile.TemporaryDirectory(prefix="codeql-", dir=self._workdir()) as tmpdir:
                database_dir = Path(tmpdir) / "database"
                sarif_path = Path(tmpdir) / "results.sarif"
                source_root = target if target.is_dir() else target.parent

                # CodeQL은 먼저 source-root를 DB로 추출한 뒤, 그 DB에 query suite를 실행합니다.
                create_error = await self._run_command(
                    [
                        executable,
                        "database",
                        "create",
                        str(database_dir),
                        f"--language={language}",
                        "--source-root",
                        str(source_root),
                        "--overwrite",
                    ]
                )
                if create_error is not None:
                    return self._failed(started_at, create_error)

                analyze_error = await self._run_command(
                    [
                        executable,
                        "database",
                        "analyze",
                        str(database_dir),
                        self._query_suite(language),
                        "--format=sarif-latest",
                        "--output",
                        str(sarif_path),
                    ]
                )
                if analyze_error is not None:
                    return self._failed(started_at, analyze_error)

                payload = json.loads(sarif_path.read_text(encoding="utf-8"))
        except TimeoutError:
            return self._failed(
                started_at,
                f"codeql timed out after {self.command_timeout_seconds} seconds",
            )
        except (OSError, json.JSONDecodeError) as exc:
            return self._failed(started_at, f"failed to run codeql: {exc}")

        return ScannerResult(
            scanner=self.scanner,
            status="completed",
            findings=self.parse_findings(payload),
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    def command(self, target: Path) -> list[str]:
        # CodeQL은 create/analyze 두 단계라서 scan()에서 직접 명령을 조립합니다.
        return ["codeql", str(target)]

    def parse_findings(self, payload: dict[str, Any]) -> list[Finding]:
        findings: list[Finding] = []
        for run in payload.get("runs", []):
            rules = self._rules_by_id(run)
            for result in run.get("results", []):
                rule_id = str(
                    result.get("ruleId")
                    or result.get("rule", {}).get("id")
                    or "codeql.unknown"
                )
                rule = rules.get(rule_id, {})
                location = self._primary_location(result)
                findings.append(
                    Finding(
                        scanner=ScannerName.codeql,
                        rule_id=rule_id,
                        severity=self._severity(result, rule),
                        title=self._message_text(rule.get("shortDescription"))
                        or self._message_text(rule.get("fullDescription"))
                        or rule_id,
                        message=self._message_text(result.get("message"))
                        or "CodeQL reported a code issue",
                        file_path=location.get("file_path"),
                        start_line=location.get("start_line"),
                        end_line=location.get("end_line"),
                        cwe=self._cwe(rule),
                        references=self._references(rule),
                        raw=result,
                    )
                )
        return findings

    async def _run_command(self, command: list[str]) -> str | None:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.command_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.command_timeout_seconds,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        if process.returncode == 0:
            return None

        output = stderr.decode("utf-8", errors="replace").strip()
        if not output:
            output = stdout.decode("utf-8", errors="replace").strip()
        return output or f"codeql exited with {process.returncode}"

    def _failed(self, started_at: datetime, error: str) -> ScannerResult:
        return ScannerResult(
            scanner=self.scanner,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            error=error,
        )

    def _workdir(self) -> str:
        workdir = Path.cwd() / ".scanner"
        workdir.mkdir(exist_ok=True)
        return str(workdir)

    def _infer_language(self, target: Path) -> str | None:
        paths = [target] if target.is_file() else list(target.rglob("*"))
        suffixes = {path.suffix.lower() for path in paths if path.is_file()}
        if ".py" in suffixes:
            return "python"
        if suffixes & {".js", ".jsx", ".ts", ".tsx"}:
            return "javascript-typescript"
        return None

    def _query_suite(self, language: str) -> str:
        # 환경변수로 커스텀 suite를 지정하면 회사/프로젝트별 CodeQL 정책으로 쉽게 바꿀 수 있습니다.
        if configured := os.getenv("DEFAPI_CODEQL_QUERY_SUITE"):
            return configured
        suites = {
            "python": "codeql/python-queries:codeql-suites/python-security-and-quality.qls",
            "javascript-typescript": "codeql/javascript-queries:codeql-suites/javascript-security-and-quality.qls",
        }
        return suites[language]

    def _rules_by_id(self, run: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rules = run.get("tool", {}).get("driver", {}).get("rules", []) or []
        return {str(rule.get("id")): rule for rule in rules if rule.get("id")}

    def _primary_location(self, result: dict[str, Any]) -> dict[str, int | str | None]:
        locations = result.get("locations", []) or []
        if not locations:
            return {"file_path": None, "start_line": None, "end_line": None}

        physical = locations[0].get("physicalLocation", {})
        artifact = physical.get("artifactLocation", {})
        region = physical.get("region", {})
        return {
            "file_path": self._uri_to_path(str(artifact.get("uri") or "")) or None,
            "start_line": region.get("startLine"),
            "end_line": region.get("endLine"),
        }

    def _severity(self, result: dict[str, Any], rule: dict[str, Any]) -> FindingSeverity:
        properties = rule.get("properties", {})
        security_severity = properties.get("security-severity")
        if security_severity is not None:
            try:
                score = float(security_severity)
            except (TypeError, ValueError):
                score = 0.0
            if score >= 9.0:
                return FindingSeverity.critical
            if score >= 7.0:
                return FindingSeverity.high
            if score >= 4.0:
                return FindingSeverity.medium
            if score > 0.0:
                return FindingSeverity.low

        level = str(result.get("level") or "note").lower()
        if level == "error":
            return FindingSeverity.high
        if level == "warning":
            return FindingSeverity.medium
        return FindingSeverity.info

    def _cwe(self, rule: dict[str, Any]) -> list[str]:
        tags = rule.get("properties", {}).get("tags", []) or []
        cwes: list[str] = []
        for tag in tags:
            normalized = str(tag).upper().replace("EXTERNAL/CWE/CWE-", "CWE-")
            if normalized.startswith("CWE-"):
                cwes.append(normalized)
        return cwes

    def _references(self, rule: dict[str, Any]) -> list[str]:
        references: list[str] = []
        if help_uri := rule.get("helpUri"):
            references.append(str(help_uri))
        return references

    def _message_text(self, value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("text") or value.get("markdown") or "")
        return str(value or "")

    def _uri_to_path(self, uri: str) -> str:
        if uri.startswith("file://"):
            return unquote(urlparse(uri).path)
        return uri
