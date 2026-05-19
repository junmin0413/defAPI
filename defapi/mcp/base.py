from __future__ import annotations

import asyncio
import json
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from defapi.models import ScannerName, ScannerResult


class CommandMCP(ABC):
    scanner: ScannerName
    executable: str
    command_timeout_seconds = 120

    async def scan(self, target: Path) -> ScannerResult:
        started_at = datetime.now(timezone.utc)
        if shutil.which(self.executable) is None:
            return ScannerResult(
                scanner=self.scanner,
                status="skipped",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error=f"{self.executable} executable is not installed",
            )

        try:
            process = await asyncio.create_subprocess_exec(
                *self.command(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.command_timeout_seconds,
            )
        except TimeoutError:
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

        if process.returncode not in self.accepted_return_codes:
            return ScannerResult(
                scanner=self.scanner,
                status="failed",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error=raw_stderr or raw_stdout or f"{self.executable} exited with {process.returncode}",
            )

        try:
            payload: dict[str, Any] = json.loads(raw_stdout or "{}")
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

    @abstractmethod
    def command(self, target: Path) -> list[str]:
        """Build the scanner command for a local target."""

    @abstractmethod
    def parse_findings(self, payload: dict[str, Any]):
        """Convert scanner JSON output into normalized findings."""
