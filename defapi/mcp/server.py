from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import ValidationError

from defapi.mcp import CodeQLMCP, SemgrepMCP, TrivyMCP
from defapi.models import ScanRecord, ScanRequest, ScanStatus, ScannerName
from defapi.workflow import ScanWorkflow


mcp = FastMCP("defAPI Security Scanner")
workflow = ScanWorkflow()
scanners = {
    ScannerName.semgrep.value: SemgrepMCP(),
    ScannerName.trivy.value: TrivyMCP(),
    ScannerName.codeql.value: CodeQLMCP(),
}


def _validated_target(target: str) -> Path:
    try:
        request = ScanRequest(target=target)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return Path(request.target)


def _json_model(model) -> dict[str, Any]:
    return model.model_dump(mode="json")


@mcp.tool
def list_scanners() -> dict[str, list[str]]:
    return {"scanners": list(scanners)}

@mcp.tool
def scanner_health() -> dict[str, dict[str, str | None]]:
    return {
        name: {
            "executable": scanner.executable,
            "path": scanner.executable_path(),
        }
        for name, scanner in scanners.items()
    }


@mcp.tool
async def scan_with_scanner(target: str, scanner: str) -> dict[str, Any]:
    normalized_scanner = scanner.strip().lower()
    if normalized_scanner not in scanners:
        available = ", ".join(scanners)
        raise ValueError(f"Unknown scanner: {scanner}. Available scanners: {available}.")

    target_path = _validated_target(target)
    result = await scanners[normalized_scanner].scan(target_path)
    return _json_model(result)


@mcp.tool
async def scan_project(target: str) -> dict[str, Any]:
    target_path = _validated_target(target)
    record = ScanRecord(target=str(target_path), status=ScanStatus.running)
    report = await workflow.run(record)
    return _json_model(report)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
