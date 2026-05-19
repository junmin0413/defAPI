from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from defapi.models import ScannerName, ScannerResult


class ZapMCP:
    scanner = ScannerName.zap

    async def scan(self, target: Path) -> ScannerResult:
        now = datetime.now(timezone.utc)
        return ScannerResult(
            scanner=ScannerName.zap,
            status="skipped",
            started_at=now,
            finished_at=now,
            error=f"ZAP scan is disabled for MVP 1: {target}",
        )

