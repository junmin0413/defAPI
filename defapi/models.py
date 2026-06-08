from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ScannerName(str, Enum):
    semgrep = "semgrep"
    trivy = "trivy"


class ScanStatus(str, Enum):
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"


class ScannerRunStatus(str, Enum):
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class FindingSeverity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class ScanRequest(BaseModel):
    target: str = Field(..., min_length=1, description="Local directory or file to scan")

    @field_validator("target")
    @classmethod
    def target_must_be_local_path(cls, value: str) -> str:
        path = Path(value).expanduser()
        if not path.exists():
            raise ValueError(f"target does not exist: {value}")
        return str(path.resolve())


class Finding(BaseModel):
    scanner: ScannerName
    rule_id: str
    severity: FindingSeverity = FindingSeverity.info
    title: str
    message: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    cwe: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ScannerResult(BaseModel):
    scanner: ScannerName
    status: ScannerRunStatus
    findings: list[Finding] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    error: str | None = None


class Report(BaseModel):
    scan_id: str
    target: str
    status: ScanStatus
    created_at: datetime
    completed_at: datetime | None = None
    scanner_results: list[ScannerResult] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class ScanRecord(BaseModel):
    scan_id: str = Field(default_factory=lambda: uuid4().hex)
    target: str
    status: ScanStatus = ScanStatus.created
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    report: Report | None = None
    error: str | None = None


class ScanResponse(BaseModel):
    scan_id: str
    status: ScanStatus
