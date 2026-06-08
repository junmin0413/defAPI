from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


DEFAULT_FIX_INSTRUCTION = "다음 스캔 결과의 보안 취약점을 분석하고 안전한 코드로 수정하라."


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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
    target: str = Field(..., min_length=1, description="스캔할 로컬 디렉터리 또는 파일 경로")

    @field_validator("target")
    @classmethod
    def target_must_be_local_path(cls, value: str) -> str:
        path = Path(value).expanduser()
        if not path.exists():
            raise ValueError(f"스캔 대상 경로가 존재하지 않습니다: {value}")
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

    def location_text(self) -> str:
        parts: list[str] = []
        if self.file_path:
            parts.append(self.file_path)
        if self.start_line is not None:
            if self.end_line is not None and self.end_line != self.start_line:
                parts.append(f"{self.start_line}-{self.end_line}행")
            else:
                parts.append(f"{self.start_line}행")
        return ":".join(parts) if parts else "위치 정보 없음"

    def to_prompt_block(self, index: int | None = None) -> str:
        prefix = f"[탐지 {index}]\n" if index is not None else ""
        cwe_text = ", ".join(self.cwe) if self.cwe else "없음"
        references_text = "\n".join(f"- {url}" for url in self.references[:5]) or "- 없음"
        return (
            f"{prefix}"
            f"스캐너: {self.scanner.value}\n"
            f"규칙 ID: {self.rule_id}\n"
            f"심각도: {self.severity.value}\n"
            f"제목: {self.title}\n"
            f"메시지: {self.message}\n"
            f"위치: {self.location_text()}\n"
            f"CWE: {cwe_text}\n"
            f"참고 링크:\n{references_text}"
        )


class ScannerResult(BaseModel):
    scanner: ScannerName
    status: ScannerRunStatus
    findings: list[Finding] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    error: str | None = None


class FineTuningSample(BaseModel):
    instruction: str
    input: str
    output: str


class ScanResult(BaseModel):
    scan_id: str
    target: str
    status: ScanStatus
    findings: list[Finding] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    scanner_statuses: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_report(cls, report: "Report") -> "ScanResult":
        return cls(
            scan_id=report.scan_id,
            target=report.target,
            status=report.status,
            findings=report.all_findings(),
            summary=report.summary,
            scanner_statuses={
                result.scanner.value: result.status.value
                for result in report.scanner_results
            },
        )

    def to_finetuning_input(self) -> str:
        if not self.findings:
            return (
                f"스캔 ID: {self.scan_id}\n"
                f"대상: {self.target}\n"
                "탐지 개수: 0\n\n"
                "스캐너가 보안 취약점을 찾지 못했다."
            )

        finding_blocks = "\n\n".join(
            finding.to_prompt_block(index=index)
            for index, finding in enumerate(self.findings, start=1)
        )
        return (
            f"스캔 ID: {self.scan_id}\n"
            f"대상: {self.target}\n"
            f"상태: {self.status.value}\n"
            f"스캐너 상태: {compact_json(self.scanner_statuses)}\n"
            f"요약: {compact_json(self.summary)}\n\n"
            f"{finding_blocks}"
        )

    def to_finetuning_sample(self, output: str, instruction: str = DEFAULT_FIX_INSTRUCTION) -> FineTuningSample:
        if not output.strip():
            raise ValueError("파인튜닝 샘플의 output은 비어 있을 수 없습니다.")
        return FineTuningSample(
            instruction=instruction,
            input=self.to_finetuning_input(),
            output=output.strip(),
        )


class Report(BaseModel):
    scan_id: str
    target: str
    status: ScanStatus
    created_at: datetime
    completed_at: datetime | None = None
    scanner_results: list[ScannerResult] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)

    def all_findings(self) -> list[Finding]:
        return [
            finding
            for scanner_result in self.scanner_results
            for finding in scanner_result.findings
        ]

    def to_scan_result(self) -> ScanResult:
        return ScanResult.from_report(self)

    def to_finetuning_input(self) -> str:
        return self.to_scan_result().to_finetuning_input()

    def to_finetuning_sample(self, output: str, instruction: str = DEFAULT_FIX_INSTRUCTION) -> FineTuningSample:
        return self.to_scan_result().to_finetuning_sample(output=output, instruction=instruction)


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
