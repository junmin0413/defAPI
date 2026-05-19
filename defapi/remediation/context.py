from __future__ import annotations

from pathlib import Path

from defapi.models import Finding


class CodeContextExtractor:
    def __init__(self, radius: int = 20) -> None:
        self.radius = radius

    def extract(self, target: Path, finding: Finding) -> str:
        file_path = self._resolve_file(target, finding.file_path)
        if file_path is None:
            return ""

        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""

        start_line = max((finding.start_line or 1) - self.radius, 1)
        end_line = min((finding.end_line or finding.start_line or 1) + self.radius, len(lines))
        width = len(str(end_line))
        return "\n".join(f"{line_no:>{width}} | {lines[line_no - 1]}" for line_no in range(start_line, end_line + 1))

    def _resolve_file(self, target: Path, file_path: str | None) -> Path | None:
        if not file_path:
            return None

        scan_root = target.resolve() if target.is_dir() else target.resolve().parent
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = scan_root / candidate

        try:
            resolved = candidate.resolve()
            resolved.relative_to(scan_root)
        except (OSError, ValueError):
            return None

        return resolved if resolved.is_file() else None
