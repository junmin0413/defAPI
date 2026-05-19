from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any

from defapi.models import Finding, PatchSuggestion, Report
from defapi.patches import PatchGenerator, finding_key
from defapi.remediation.context import CodeContextExtractor
from defapi.remediation.prompts import RemediationPromptBuilder


OPENAI_REMEDIATOR = "openai"
RULE_REMEDIATOR = "rule"
DEFAULT_OPENAI_MODEL = "gpt-5.2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER_API_KEYS = {"", "replace_with_rotated_openai_key", "your_api_key"}


def load_local_env(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def valid_api_key(value: str | None) -> bool:
    return bool(value and value.strip() not in PLACEHOLDER_API_KEYS)


class FineTunedLLMRemediator(ABC):
    @abstractmethod
    def generate(self, target: Path, report: Report, findings: list[Finding]) -> list[PatchSuggestion]:
        """Generate unified-diff patch suggestions from report and vulnerable code context."""


class RuleBasedLLMRemediator(FineTunedLLMRemediator):
    def __init__(
        self,
        patch_generator: PatchGenerator | None = None,
        context_extractor: CodeContextExtractor | None = None,
        prompt_builder: RemediationPromptBuilder | None = None,
    ) -> None:
        self.patch_generator = patch_generator or PatchGenerator()
        self.context_extractor = context_extractor or CodeContextExtractor()
        self.prompt_builder = prompt_builder or RemediationPromptBuilder()

    def generate(self, target: Path, report: Report, findings: list[Finding]) -> list[PatchSuggestion]:
        rule_patches = {patch.finding_key: patch for patch in self.patch_generator.generate(target, findings)}
        return [
            self._patch_from_finding(
                target,
                report,
                finding,
                rule_patches.get(finding_key(finding)),
            )
            for finding in findings
        ]

    def _patch_from_finding(
        self,
        target: Path,
        report: Report,
        finding: Finding,
        rule_patch: PatchSuggestion | None,
    ) -> PatchSuggestion:
        code_context = self.context_extractor.extract(target, finding)
        prompt = self.prompt_builder.build(report, finding, code_context)
        return PatchSuggestion(
            finding_key=finding_key(finding),
            file_path=rule_patch.file_path if rule_patch else finding.file_path,
            strategy="llm remediation fallback",
            unified_diff=rule_patch.unified_diff if rule_patch else None,
            instructions=self._instructions(prompt, rule_patch.instructions if rule_patch else finding.message),
            applicable=rule_patch.applicable if rule_patch else bool(code_context),
        )

    def _instructions(self, prompt: str, guidance: str) -> str:
        return (
            f"Fallback guidance: {guidance}\n\n"
            "[Model Prompt]\n"
            f"{prompt}"
        )


class OpenAIAPIRemediator(FineTunedLLMRemediator):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        context_extractor: CodeContextExtractor | None = None,
        prompt_builder: RemediationPromptBuilder | None = None,
        fallback: FineTunedLLMRemediator | None = None,
        client: Any | None = None,
    ) -> None:
        load_local_env()
        self.model = model or os.getenv("DEFAPI_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.context_extractor = context_extractor or CodeContextExtractor()
        self.prompt_builder = prompt_builder or RemediationPromptBuilder()
        self.fallback = fallback or RuleBasedLLMRemediator(
            context_extractor=self.context_extractor,
            prompt_builder=self.prompt_builder,
        )
        self.client = client

    def generate(self, target: Path, report: Report, findings: list[Finding]) -> list[PatchSuggestion]:
        if not valid_api_key(self.api_key) and self.client is None:
            return self.fallback.generate(target, report, findings)

        return [self._patch_from_finding(target, report, finding) for finding in findings]

    def _patch_from_finding(self, target: Path, report: Report, finding: Finding) -> PatchSuggestion:
        prompt = self._build_prompt(target, report, finding)
        try:
            diff = self._generate_diff(prompt)
        except Exception as exc:
            return self._fallback_patch(target, report, finding, prompt, f"OpenAI API call failed: {exc}")

        if not diff:
            return self._fallback_patch(target, report, finding, prompt, "OpenAI API returned an empty patch.")

        return PatchSuggestion(
            finding_key=finding_key(finding),
            file_path=finding.file_path,
            strategy=f"openai api remediation ({self.model})",
            unified_diff=diff,
            instructions=self._instructions(prompt),
            applicable=True,
        )

    def _build_prompt(self, target: Path, report: Report, finding: Finding) -> str:
        code_context = self.context_extractor.extract(target, finding)
        return self.prompt_builder.build(report, finding, code_context)

    def _instructions(self, prompt: str) -> str:
        return (
            "OpenAI API generated this patch from the MCP finding report and vulnerable code context.\n\n"
            "[Model Prompt]\n"
            f"{prompt}"
        )

    def _client(self):
        if self.client is not None:
            return self.client

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("openai package is not installed") from exc

        self.client = OpenAI(api_key=self.api_key)
        return self.client

    def _generate_diff(self, prompt: str) -> str:
        response = self._client().responses.create(
            model=self.model,
            instructions=(
                "You are a security patch generator. "
                "Return only a valid unified diff. Do not include markdown fences."
            ),
            input=prompt,
        )
        return self._normalize_diff(getattr(response, "output_text", ""))

    def _normalize_diff(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    def _fallback_patch(self, target: Path, report: Report, finding: Finding, prompt: str, reason: str) -> PatchSuggestion:
        fallback_patch = self.fallback.generate(target, report, [finding])[0]
        instructions = f"{reason}\n\n{fallback_patch.instructions}\n\n[OpenAI Prompt]\n{prompt}"
        return fallback_patch.model_copy(
            update={
                "strategy": "openai api remediation fallback",
                "instructions": instructions,
            }
        )


def create_default_remediator() -> FineTunedLLMRemediator:
    load_local_env()
    backend = os.getenv("DEFAPI_REMEDIATOR", OPENAI_REMEDIATOR).lower()
    if backend == RULE_REMEDIATOR:
        return RuleBasedLLMRemediator()
    return OpenAIAPIRemediator()
