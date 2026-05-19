from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from defapi.models import PatchSuggestion, Report, ScanRecord, ScanStatus, ScannerResult, ValidationResult


class ReportGenerator:
    def build(
        self,
        record: ScanRecord,
        scanner_results: list[ScannerResult],
        patches: list[PatchSuggestion],
        validation: list[ValidationResult],
    ) -> Report:
        summary = Counter()
        for result in scanner_results:
            summary[f"{result.scanner.value}_{result.status.value}"] += 1
            for finding in result.findings:
                summary[f"severity_{finding.severity.value}"] += 1
        summary["findings_total"] = sum(len(result.findings) for result in scanner_results)
        summary["patches_total"] = len(patches)
        summary["valid_patches_total"] = sum(1 for item in validation if item.valid)
        return Report(
            scan_id=record.scan_id,
            target=record.target,
            status=ScanStatus.completed,
            created_at=record.created_at,
            completed_at=datetime.now(timezone.utc),
            scanner_results=scanner_results,
            patches=patches,
            validation=validation,
            summary=dict(summary),
        )
