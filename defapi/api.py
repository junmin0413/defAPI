from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from defapi.models import Report, ScanRecord, ScanRequest, ScanResponse, ScanStatus
from defapi.workflow import ScanWorkflow


app = FastAPI(title="defAPI MVP Security Scanner", version="0.1.0")
workflow = ScanWorkflow()
records: dict[str, ScanRecord] = {}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scan", response_model=ScanResponse)
async def create_scan(request: ScanRequest) -> ScanResponse:
    record = ScanRecord(
        target=request.target,
        include_zap=request.include_zap,
        apply_patches=request.apply_patches,
        status=ScanStatus.running,
    )
    records[record.scan_id] = record
    await _run_scan(record.scan_id)
    return ScanResponse(scan_id=record.scan_id, status=record.status)


@app.get("/report/{scan_id}", response_model=Report)
async def get_report(scan_id: str) -> Report:
    record = records.get(scan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="scan not found")
    if record.status == ScanStatus.failed:
        raise HTTPException(status_code=500, detail=record.error or "scan failed")
    if record.report is None:
        raise HTTPException(status_code=202, detail="scan is still running")
    return record.report


async def _run_scan(scan_id: str) -> None:
    record = records[scan_id]
    try:
        report = await workflow.run(record)
        record.report = report
        record.status = ScanStatus.completed
        record.completed_at = report.completed_at
    except Exception as exc:
        record.status = ScanStatus.failed
        record.completed_at = datetime.now(timezone.utc)
        record.error = str(exc)
