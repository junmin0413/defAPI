# defAPI Architecture

This document describes the current MVP pipeline and how the scanner wrappers, workflow, patch generation, validation, and reporting components work together.

## High-Level Flow

```text
Client
  -> FastAPI /scan
  -> ScanRecord
  -> ScanWorkflow
    -> scan
    -> patch
    -> validate
    -> report
  -> FastAPI /report/{scan_id}
```

## API Layer

`defapi/api.py` exposes the public MVP interface.

- `GET /health`: service health check.
- `POST /scan`: validates the local target path, creates a scan record, runs the workflow, and returns a scan id.
- `GET /report/{scan_id}`: returns the final report for a completed scan.

Current limitation: scan records are stored in an in-memory dictionary. This is acceptable for the MVP but should be replaced with persistent job storage before production use.

## Workflow Layer

`defapi/workflow.py` uses LangGraph to define the pipeline:

```text
scan -> patch -> validate -> report
```

### scan

Runs scanner adapters and aggregates findings.

Current behavior:

- Always runs Semgrep and Trivy.
- Runs ZAP only when `include_zap=true`.
- Combines all findings into one normalized list.

### patch

Calls `PatchGenerator` to create a `PatchSuggestion` for each finding.

The current MVP logic is rule-based:

- Semgrep findings produce remediation instructions.
- Trivy dependency findings can produce a simple version upgrade diff when `PkgName`, `InstalledVersion`, and `FixedVersion` are available.

### validate

Calls `ValidationLoop` to perform structural checks.

Current checks:

- Patch target path exists.
- Patch target path is inside the scan target.
- Unified diff output has the minimum expected headers.

Future checks:

- Parse diff hunks.
- Run `git apply --check`.
- Apply patches in a temporary worktree.
- Run project tests.
- Re-run scanners to verify finding resolution.

### report

Calls `ReportGenerator` to produce a `Report` object with:

- scanner results
- patch suggestions
- validation results
- summary counters

## MCP Scanner Wrappers

The `defapi/mcp/` package contains scanner adapters. In this project, MCP means a wrapper boundary around external scanner tools. The goal is to hide scanner-specific command execution and JSON parsing behind a consistent Python interface.

## CommandMCP

`defapi/mcp/base.py` provides shared scanner behavior:

1. Check whether the executable exists.
2. Build the scanner command.
3. Run the command asynchronously.
4. Apply a timeout.
5. Decode stdout/stderr.
6. Parse JSON output.
7. Convert scanner-specific output into normalized findings.
8. Return a `ScannerResult`.

If the executable is missing, the scanner is marked as `skipped` instead of failing the whole scan.

## SemgrepMCP

`defapi/mcp/semgrep.py`

Command:

```text
semgrep --config auto --json --quiet {target}
```

Primary purpose:

- Detect source-code security issues.
- Parse Semgrep `results`.
- Convert each result into a normalized `Finding`.

Important mapped fields:

- `check_id` -> `rule_id`
- `extra.severity` -> `severity`
- `extra.message` -> `message`
- `path` -> `file_path`
- `start.line` -> `start_line`
- `end.line` -> `end_line`
- `extra.metadata.cwe` -> `cwe`
- `extra.metadata.references` -> `references`

## TrivyMCP

`defapi/mcp/trivy.py`

Command:

```text
trivy fs --format json --quiet {target}
```

Primary purpose:

- Detect dependency vulnerabilities.
- Detect filesystem and configuration issues.
- Parse Trivy `Vulnerabilities` and `Misconfigurations`.

Important mapped fields:

- `VulnerabilityID` or `ID` -> `rule_id`
- `Severity` -> `severity`
- `Title` -> `title`
- `Description` or `Message` -> `message`
- `Target` -> `file_path`
- `CweIDs` -> `cwe`
- `References` -> `references`

## ZapMCP

`defapi/mcp/zap.py`

Current status: placeholder.

ZAP scanning is disabled in the MVP and returns a skipped scanner result. This preserves the API and workflow shape while avoiding unsafe or incomplete dynamic scanning behavior.

Future ZAP support should include:

- URL target validation
- passive scan mode
- active scan opt-in
- authentication/session handling
- scan policy configuration
- timeout and risk controls

## Core Models

`defapi/models.py` defines the shared schema.

Key models:

- `ScanRequest`: user input for a scan.
- `ScanRecord`: internal scan state.
- `ScannerResult`: one scanner execution result.
- `Finding`: normalized vulnerability or security issue.
- `PatchSuggestion`: proposed remediation or diff.
- `ValidationResult`: structural validation outcome.
- `Report`: final scan output.

## LLM Patch Pipeline Target

The future LLM remediation pipeline should extend the current workflow:

```text
Finding
  -> CodeContextExtractor
  -> LLMPatchGenerator
  -> DiffValidator
  -> SandboxPatchApplier
  -> TestRunner
  -> ScannerRerunner
  -> DatasetLogger
```

The important rule is that LLM output is never trusted directly. The LLM should only produce candidate diffs. Those diffs must be validated before they are presented as recommended patches.
