from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from defapi.mcp import SemgrepMCP, TrivyMCP, ZapMCP
from defapi.models import PatchSuggestion, ScanRecord, ScannerResult, ValidationResult
from defapi.validation import ValidationLoop


class RemediationVerifier:
    def __init__(
        self,
        validation_loop: ValidationLoop | None = None,
        semgrep: SemgrepMCP | None = None,
        trivy: TrivyMCP | None = None,
        zap: ZapMCP | None = None,
    ) -> None:
        self.validation_loop = validation_loop or ValidationLoop()
        self.semgrep = semgrep or SemgrepMCP()
        self.trivy = trivy or TrivyMCP()
        self.zap = zap or ZapMCP()

    async def validate_and_rescan(
        self,
        target: Path,
        record: ScanRecord,
        patches: list[PatchSuggestion],
    ) -> tuple[list[ValidationResult], list[ScannerResult]]:
        validation = self.validation_loop.validate(target, patches)
        if not record.apply_patches or not any(item.valid for item in validation):
            return validation, []

        verification_target = self._create_patched_copy(target, patches, validation)
        if verification_target is None:
            return validation, []

        try:
            with verification_target as sandbox_target:
                scanner_results = await self._scan_verification_target(sandbox_target, record.include_zap)
        except RuntimeError as exc:
            validation.append(
                ValidationResult(
                    finding_key="remediation_verification",
                    valid=False,
                    reason=f"Sandbox patch verification failed: {exc}",
                )
            )
            return validation, []
        return validation, scanner_results

    def _create_patched_copy(
        self,
        target: Path,
        patches: list[PatchSuggestion],
        validation: list[ValidationResult],
    ) -> TemporaryDirectoryPath | None:
        valid_keys = {item.finding_key for item in validation if item.valid}
        diffs = [patch.unified_diff for patch in patches if patch.finding_key in valid_keys and patch.unified_diff]
        if not diffs:
            return None

        return TemporaryDirectoryPath(target, "\n".join(diffs))

    async def _scan_verification_target(self, target: Path, include_zap: bool) -> list[ScannerResult]:
        scanners = [self.semgrep.scan(target), self.trivy.scan(target)]
        if include_zap:
            scanners.append(self.zap.scan(target))
        return list(await asyncio.gather(*scanners))


class TemporaryDirectoryPath:
    def __init__(self, source: Path, unified_diff: str) -> None:
        self.source = source
        self.unified_diff = unified_diff
        self._tempdir: TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._tempdir = TemporaryDirectory()
        try:
            sandbox = Path(self._tempdir.name) / self.source.name
            if self.source.is_dir():
                shutil.copytree(self.source, sandbox)
            else:
                sandbox.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.source, sandbox)

            result = subprocess.run(
                ["git", "apply", "-"],
                input=self.unified_diff,
                text=True,
                cwd=sandbox if sandbox.is_dir() else sandbox.parent,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "git apply failed in remediation sandbox")
            return sandbox
        except Exception:
            self._tempdir.cleanup()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
