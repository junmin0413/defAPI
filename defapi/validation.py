from __future__ import annotations

from pathlib import Path

from defapi.models import PatchSuggestion, ValidationResult


class ValidationLoop:
    def validate(self, target: Path, patches: list[PatchSuggestion]) -> list[ValidationResult]:
        return [self._validate_one(target, patch) for patch in patches]

    def _validate_one(self, target: Path, patch: PatchSuggestion) -> ValidationResult:
        scan_root = self._scan_root(target)
        if not patch.applicable:
            return ValidationResult(
                finding_key=patch.finding_key,
                valid=False,
                reason="No local file or dependency upgrade data is available for an automatic patch.",
            )
        if patch.file_path:
            path = Path(patch.file_path)
            if not path.exists():
                return ValidationResult(
                    finding_key=patch.finding_key,
                    valid=False,
                    reason=f"Patch target does not exist under scan target: {path}",
                )
            try:
                path.resolve().relative_to(scan_root)
            except ValueError:
                return ValidationResult(
                    finding_key=patch.finding_key,
                    valid=False,
                    reason=f"Patch target is outside scan target: {path}",
                )
        if patch.unified_diff and not self._looks_like_unified_diff(patch.unified_diff):
            return ValidationResult(
                finding_key=patch.finding_key,
                valid=False,
                reason="Generated diff is not a unified diff.",
            )
        return ValidationResult(
            finding_key=patch.finding_key,
            valid=True,
            reason="Patch suggestion is structurally valid for manual review.",
        )

    def _scan_root(self, target: Path) -> Path:
        resolved = target.resolve()
        return resolved if resolved.is_dir() else resolved.parent

    def _looks_like_unified_diff(self, diff: str) -> bool:
        lines = diff.splitlines()
        return len(lines) >= 3 and lines[0].startswith("--- ") and lines[1].startswith("+++ ")
