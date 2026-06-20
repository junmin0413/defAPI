from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from defapi.models import Finding, FindingSeverity, ScannerName, ScannerResult

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


class CommandMCP(ABC):
    # 외부 보안 CLI를 "MCP처럼" 다루기 위한 공통 어댑터입니다.
    # 하위 클래스는 command()로 실행 명령을 만들고, parse_findings()로
    # 각 스캐너의 JSON 결과를 DefAPI의 공통 Finding 모델로 변환합니다.
    scanner: ScannerName
    executable: str
    command_timeout_seconds = 120

    async def scan(self, target: Path) -> ScannerResult:
        started_at = datetime.now(timezone.utc)
        executable = self.executable_path()
        if executable is None:
            # CLI가 설치되지 않은 환경에서도 전체 스캔이 죽지 않도록 skipped로 남깁니다.
            return ScannerResult(
                scanner=self.scanner,
                status="skipped",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error=f"{self.executable} executable is not installed",
            )

        try:
            command = self.command(target)
            # command()는 테스트/가독성을 위해 실행 파일 이름을 넣어 반환하고,
            # 실제 실행 직전에는 PATH 또는 venv에서 찾은 절대 경로로 교체합니다.
            command[0] = executable
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.command_env(),
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.command_timeout_seconds,
            )
        except TimeoutError:
            # 멈춘 스캐너 프로세스가 남지 않도록 반드시 kill/wait까지 수행합니다.
            process.kill()
            await process.wait()
            return ScannerResult(
                scanner=self.scanner,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error=f"{self.executable} timed out after {self.command_timeout_seconds} seconds",
            )
        except OSError as exc:
            return ScannerResult(
                scanner=self.scanner,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error=f"failed to run {self.executable}: {exc}",
            )

        raw_stdout = stdout.decode("utf-8", errors="replace")
        raw_stderr = stderr.decode("utf-8", errors="replace").strip()
        # 대부분의 스캐너는 stdout으로 JSON을 주지만, 일부 실패 케이스는 stderr에 JSON을 씁니다.
        payload_text = raw_stdout or (raw_stderr if raw_stderr.startswith("{") else "")

        if process.returncode not in self.accepted_return_codes and not payload_text:
            return ScannerResult(
                scanner=self.scanner,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error=raw_stderr or raw_stdout or f"{self.executable} exited with {process.returncode}",
            )

        try:
            payload: dict[str, Any] = json.loads(payload_text or "{}")
        except json.JSONDecodeError as exc:
            return ScannerResult(
                scanner=self.scanner,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error=f"invalid JSON from {self.executable}: {exc}",
            )

        findings = self.parse_findings(payload)
        return ScannerResult(
            scanner=self.scanner,
            status="completed",
            findings=findings,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    @property
    def accepted_return_codes(self) -> set[int]:
        return {0}

    def executable_path(self) -> str | None:
        # pip로 설치된 CLI가 PATH에 없고 현재 venv/bin에만 있는 경우까지 찾습니다.
        if path := shutil.which(self.executable):
            return path
        venv_path = Path(sys.executable).parent / self.executable
        return str(venv_path) if venv_path.exists() else None

    def command_env(self) -> dict[str, str]:
        # 스캐너가 로그/인증서 파일을 안정적으로 찾도록 실행 환경을 최소 보정합니다.
        env = os.environ.copy()
        workdir = Path.cwd() / ".scanner"
        workdir.mkdir(exist_ok=True)
        env.setdefault("SEMGREP_LOG_FILE", str(workdir / "semgrep.log"))
        if certifi is not None:
            env.setdefault("SSL_CERT_FILE", certifi.where())
        return env

    @abstractmethod
    def command(self, target: Path) -> list[str]:
        """Build the scanner command for a local target."""

    @abstractmethod
    def parse_findings(self, payload: dict[str, Any]):
        """Convert scanner JSON output into normalized findings."""


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


class TrivyMCP(CommandMCP):
    scanner = ScannerName.trivy
    executable = "trivy"

    def command(self, target: Path) -> list[str]:
        return ["trivy", "fs", "--format", "json", "--quiet", str(target)]

    def parse_findings(self, payload: dict[str, Any]) -> list[Finding]:
        findings: list[Finding] = []
        for result in payload.get("Results", []):
            file_path = result.get("Target")
            # Trivy의 dependency CVE 결과를 Finding으로 정규화합니다.
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
            # IaC/Docker/Kubernetes 설정 오류는 Vulnerabilities와 다른 필드로 내려옵니다.
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
