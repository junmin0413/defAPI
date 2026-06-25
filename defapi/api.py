from __future__ import annotations

from datetime import datetime, timezone
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from defapi.models import (
    Finding,
    Report,
    ScanRecord,
    ScanRequest,
    ScanResponse,
    ScanStatus,
)
from defapi.workflow import ScanWorkflow


app = FastAPI(title="defAPI MVP Security Scanner", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
workflow = ScanWorkflow()
records: dict[str, ScanRecord] = {}
projects: dict[str, dict[str, str]] = {}


class AuthRequest(BaseModel):
    email: str | None = None
    password: str | None = None
    displayName: str | None = None
    refreshToken: str | None = None
    idToken: str | None = None
    code: str | None = None


class ProjectCreateRequest(BaseModel):
    name: str = Field(default="default")
    language: str = Field(default="auto")
    description: str = Field(default="")
    visibility: str = Field(default="private")


class FrontendScanRequest(BaseModel):
    inputType: str = Field(default="code")
    content: str = Field(default="")
    filePath: str = Field(default="uploaded.txt")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
async def api_health() -> dict[str, str]:
    return await health()


@app.post("/scan", response_model=ScanResponse)
async def create_scan(request: ScanRequest) -> ScanResponse:
    record = ScanRecord(
        target=request.target,
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


def _auth_payload(email: str | None = None, display_name: str | None = None) -> dict:
    user_email = email or "local@defapi.dev"
    return {
        "user": {
            "id": "local-user",
            "email": user_email,
            "displayName": display_name or user_email.split("@", 1)[0],
        },
        "tokens": {
            "accessToken": f"local-access-{uuid4().hex}",
            "refreshToken": f"local-refresh-{uuid4().hex}",
        },
    }


@app.post("/api/auth/login")
async def frontend_login(request: AuthRequest) -> dict:
    return _auth_payload(request.email)


@app.post("/api/auth/signup")
async def frontend_signup(request: AuthRequest) -> dict:
    return _auth_payload(request.email, request.displayName)


@app.post("/api/auth/refresh")
async def frontend_refresh(_request: AuthRequest) -> dict:
    return {
        "tokens": {
            "accessToken": f"local-access-{uuid4().hex}",
            "refreshToken": f"local-refresh-{uuid4().hex}",
        }
    }


@app.post("/api/auth/logout")
async def frontend_logout() -> dict[str, bool]:
    return {"ok": True}


@app.post("/api/auth/oauth/google")
async def frontend_google_login(_request: AuthRequest) -> dict:
    return _auth_payload("google@defapi.dev", "google")


@app.post("/api/auth/oauth/github")
async def frontend_github_login(_request: AuthRequest) -> dict:
    return _auth_payload("github@defapi.dev", "github")


@app.get("/api/projects")
async def frontend_projects() -> dict:
    return {"projects": list(projects.values())}


@app.post("/api/projects")
async def frontend_create_project(request: ProjectCreateRequest) -> dict:
    project_id = uuid4().hex
    project = {
        "id": project_id,
        "name": request.name,
        "language": request.language,
        "description": request.description,
        "visibility": request.visibility,
    }
    projects[project_id] = project
    return {"project": project}


@app.post("/api/projects/{project_id}/upload")
async def frontend_upload(project_id: str, request: Request) -> dict:
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="project not found")
    content_type = request.headers.get("content-type", "application/octet-stream")
    return {
        "artifact": {
            "id": uuid4().hex,
            "filename": "uploaded",
            "contentType": content_type,
        }
    }


@app.post("/api/projects/{project_id}/scan")
async def frontend_start_scan(project_id: str, request: FrontendScanRequest) -> dict:
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="project not found")

    target = _write_frontend_scan_target(request)
    record = ScanRecord(target=str(target), status=ScanStatus.running)
    records[record.scan_id] = record
    await _run_scan(record.scan_id)
    return {
        "scanId": record.scan_id,
        "status": record.status.value,
        "reportId": record.scan_id if record.report else None,
    }


@app.get("/api/scans/{scan_id}")
async def frontend_scan_status(scan_id: str) -> dict:
    record = records.get(scan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return {
        "scan": {
            "id": scan_id,
            "scanId": scan_id,
            "status": record.status.value,
            "reportId": scan_id if record.report else None,
            "error": record.error,
        }
    }


@app.get("/api/reports/{report_id}")
async def frontend_report(report_id: str) -> dict:
    report = await get_report(report_id)
    return {"report": _to_frontend_report(report)}


def _write_frontend_scan_target(request: FrontendScanRequest) -> Path:
    safe_name = Path(request.filePath or "uploaded.txt").name or "uploaded.txt"
    target_dir = Path(tempfile.mkdtemp(prefix="defapi_frontend_"))
    target_file = target_dir / safe_name
    target_file.write_text(request.content or "", encoding="utf-8")
    return target_dir


def _to_frontend_report(report: Report) -> dict:
    findings = report.all_findings()
    return {
        "id": report.scan_id,
        "scanId": report.scan_id,
        "status": report.status.value,
        "target": report.target,
        "summary": report.summary,
        "createdAt": report.created_at.isoformat(),
        "completedAt": report.completed_at.isoformat() if report.completed_at else None,
        "issues": [
            _finding_to_frontend_issue(finding, index, report.repair)
            for index, finding in enumerate(findings, start=1)
        ],
    }


def _finding_to_frontend_issue(finding: Finding, index: int, repair: str | None) -> dict:
    severity = finding.severity.value.capitalize()
    return {
        "id": f"{finding.scanner.value}-{index}-{finding.rule_id}",
        "name": finding.title or finding.rule_id,
        "severity": severity,
        "confidence": 0.8,
        "description": finding.message,
        "fix_code": repair or "자동 수정 코드가 아직 생성되지 않았습니다.",
        "file_path": finding.file_path,
        "start_line": finding.start_line,
        "end_line": finding.end_line or finding.start_line,
        "exploit_example": "\n".join(finding.references[:3]),
        "display_meta": {
            "severity_color": _severity_color(severity),
            "highlight_lines": [
                line
                for line in (finding.start_line, finding.end_line)
                if line is not None
            ],
        },
    }


def _severity_color(severity: str) -> str:
    return {
        "Critical": "#ff4d4f",
        "High": "#ff8c42",
        "Medium": "#ffbf3c",
        "Low": "#4caf50",
        "Info": "#9ca3af",
    }.get(severity, "#9ca3af")


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        requested = FRONTEND_DIST / full_path
        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
