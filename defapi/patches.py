from __future__ import annotations

from pathlib import Path

from defapi.models import Finding, PatchSuggestion, ScannerName


class PatchGenerator:
    def generate(self, target: Path, findings: list[Finding]) -> list[PatchSuggestion]:
        return [self._suggestion(target, finding) for finding in findings]

    def _suggestion(self, target: Path, finding: Finding) -> PatchSuggestion:
        key = finding_key(finding)
        file_path = self._resolve_file(target, finding.file_path)
        instructions = self._instructions(finding)
        diff = self._dependency_diff(target, file_path, finding)
        return PatchSuggestion(
            finding_key=key,
            file_path=str(file_path) if file_path else finding.file_path,
            strategy="rule-based remediation guidance",
            unified_diff=diff,
            instructions=instructions,
            applicable=diff is not None or file_path is not None,
        )

    def _resolve_file(self, target: Path, file_path: str | None) -> Path | None:
        if not file_path:
            return None
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = target / candidate if target.is_dir() else target.parent / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            return None
        if resolved.exists() and resolved.is_file():
            return resolved
        return None

    def _dependency_diff(self, target: Path, file_path: Path | None, finding: Finding) -> str | None:
        if finding.scanner != ScannerName.trivy or file_path is None:
            return None
        installed = finding.raw.get("InstalledVersion")
        fixed = finding.raw.get("FixedVersion")
        pkg = finding.raw.get("PkgName")
        if not installed or not fixed or not pkg:
            return None
        old_line = f"{pkg}=={installed}"
        new_line = f"{pkg}=={fixed}"
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            line_number = lines.index(old_line) + 1
        except ValueError:
            return None
        diff_path = self._diff_path(target, file_path)
        return (
            f"--- a/{diff_path}\n"
            f"+++ b/{diff_path}\n"
            f"@@ -{line_number},1 +{line_number},1 @@\n"
            f"-{old_line}\n"
            f"+{new_line}\n"
        )

    def _diff_path(self, target: Path, file_path: Path) -> str:
        scan_root = target.resolve() if target.is_dir() else target.resolve().parent
        try:
            return file_path.resolve().relative_to(scan_root).as_posix()
        except ValueError:
            return file_path.name

    def _instructions(self, finding: Finding) -> str:
        if finding.scanner == ScannerName.semgrep:
            location = f" at {finding.file_path}:{finding.start_line}" if finding.file_path and finding.start_line else ""
            return f"Review {finding.rule_id}{location}. {finding.message}"
        fixed = finding.raw.get("FixedVersion")
        pkg = finding.raw.get("PkgName")
        if fixed and pkg:
            return f"Upgrade {pkg} to {fixed} and rerun Trivy."
        return f"Remediate {finding.rule_id}. {finding.message}"


def finding_key(finding: Finding) -> str:
    parts = [
        finding.scanner.value,
        finding.rule_id,
        finding.file_path or "target",
        str(finding.start_line or 0),
    ]
    return "::".join(parts)
