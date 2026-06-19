from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from defapi.models import ScannerName, ScannerResult

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
