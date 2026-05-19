from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from defapi.api import app, records
from defapi.models import Finding, FindingSeverity, ScanRecord, ScannerName
from defapi.patches import PatchGenerator
from defapi.validation import ValidationLoop
from defapi.workflow import ScanWorkflow


def test_scan_api_returns_report_with_zap_skipped(tmp_path):
    sample = tmp_path / "app.py"
    sample.write_text("print('ok')\n")
    client = TestClient(app)

    response = client.post("/scan", json={"target": str(tmp_path), "include_zap": True})
    assert response.status_code == 200
    scan_id = response.json()["scan_id"]

    report_response = client.get(f"/report/{scan_id}")
    assert report_response.status_code == 200
    payload = report_response.json()
    assert payload["scan_id"] == scan_id
    assert payload["summary"]["findings_total"] == 0
    zap = [item for item in payload["scanner_results"] if item["scanner"] == "zap"][0]
    assert zap["status"] == "skipped"
    records.pop(scan_id, None)


def test_workflow_builds_report_with_mock_scanners(tmp_path):
    sample = tmp_path / "app.py"
    sample.write_text("secret = 'hardcoded'\n")
    workflow = ScanWorkflow()

    finding = Finding(
        scanner=ScannerName.semgrep,
        rule_id="python.lang.security.audit",
        severity=FindingSeverity.medium,
        title="Audit",
        message="Review hardcoded value",
        file_path=str(sample),
        start_line=1,
    )

    async def semgrep_scan(target):
        from defapi.models import ScannerResult

        return ScannerResult(scanner=ScannerName.semgrep, status="completed", findings=[finding])

    async def trivy_scan(target):
        from defapi.models import ScannerResult

        return ScannerResult(scanner=ScannerName.trivy, status="completed", findings=[])

    async def zap_scan(target):
        from defapi.models import ScannerResult

        return ScannerResult(scanner=ScannerName.zap, status="skipped", findings=[])

    workflow.semgrep.scan = semgrep_scan
    workflow.trivy.scan = trivy_scan
    workflow.zap.scan = zap_scan

    report = asyncio.run(workflow.run(ScanRecord(target=str(tmp_path), include_zap=False)))

    assert report.summary["findings_total"] == 1
    assert report.summary["patches_total"] == 1
    assert report.summary["valid_patches_total"] == 1
    assert "zap_skipped" not in report.summary


def test_workflow_runs_zap_only_when_requested(tmp_path):
    sample = tmp_path / "app.py"
    sample.write_text("print('ok')\n")
    workflow = ScanWorkflow()

    async def semgrep_scan(target):
        from defapi.models import ScannerResult

        return ScannerResult(scanner=ScannerName.semgrep, status="completed", findings=[])

    async def trivy_scan(target):
        from defapi.models import ScannerResult

        return ScannerResult(scanner=ScannerName.trivy, status="completed", findings=[])

    async def zap_scan(target):
        from defapi.models import ScannerResult

        return ScannerResult(scanner=ScannerName.zap, status="skipped", findings=[])

    workflow.semgrep.scan = semgrep_scan
    workflow.trivy.scan = trivy_scan
    workflow.zap.scan = zap_scan

    report = asyncio.run(workflow.run(ScanRecord(target=str(tmp_path), include_zap=True)))

    assert report.summary["zap_skipped"] == 1


def test_patch_validation_rejects_outside_target(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x\n")
    finding = Finding(
        scanner=ScannerName.semgrep,
        rule_id="x",
        message="bad",
        title="bad",
        file_path=str(outside),
    )

    patch = PatchGenerator().generate(tmp_path, [finding])[0]
    validation = ValidationLoop().validate(tmp_path, [patch])[0]

    assert validation.valid is False
    assert "outside" in validation.reason


def test_patch_validation_rejects_malformed_diff(tmp_path):
    sample = tmp_path / "requirements.txt"
    sample.write_text("demo==1.0.0\n")
    from defapi.models import PatchSuggestion

    patch = PatchSuggestion(
        finding_key="x",
        file_path=str(sample),
        strategy="test",
        unified_diff="not a diff",
        instructions="test",
        applicable=True,
    )

    validation = ValidationLoop().validate(tmp_path, [patch])[0]

    assert validation.valid is False
    assert "unified diff" in validation.reason
