# defAPI

defAPI is a FastAPI-based security scanning and remediation MVP. It normalizes findings from security scanners, generates patch suggestions, validates those suggestions, and produces a structured report that can later be used as supervised fine-tuning and DPO data for an open-source code LLM.

This project is being built toward a developer security agent that can move from:

```text
scan vulnerable code -> explain findings -> generate secure diffs -> validate patches -> produce reviewable reports
```

## Why This Project Exists

Security tools are good at detecting vulnerabilities, but developers still need to interpret scanner output, locate the vulnerable code, decide on a safe fix, apply the patch, and verify that the fix did not break the project.

defAPI focuses on the missing bridge between scanner findings and verified remediation:

- Normalize findings from tools such as Semgrep, Trivy, and OWASP ZAP.
- Convert scanner output into a consistent internal schema.
- Generate patch suggestions or remediation guidance.
- Validate whether a patch is structurally safe before it is applied.
- Build a dataset pipeline for future LoRA/SFT and DPO training.

## Current Status

This repository is an MVP, not a production security product yet.

Implemented:

- FastAPI scan/report endpoints
- LangGraph-based scan workflow
- Semgrep scanner wrapper
- Trivy scanner wrapper
- ZAP placeholder wrapper
- Finding normalization with Pydantic models
- Rule-based patch suggestions
- Basic patch validation guardrails
- Report generation
- LoRA/SFT and DPO training module skeletons
- Pytest MVP coverage

Planned:

- Background scan jobs
- Persistent scan storage
- LLM-generated unified diffs
- Sandbox patch application
- `git apply --check` validation
- Test runner integration
- Scanner re-run validation
- SFT/DPO dataset export

## Architecture

```text
FastAPI API
  -> ScanWorkflow
    -> MCP scanner wrappers
      -> SemgrepMCP
      -> TrivyMCP
      -> ZapMCP
    -> normalized Finding objects
    -> PatchGenerator
    -> ValidationLoop
    -> ReportGenerator
  -> /report/{scan_id}
```

The scanner wrappers act as adapters. Each external scanner has a different JSON output format, but defAPI converts them into a shared `Finding` model before downstream processing.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline.

## Repository Structure

```text
defapi/
  api.py                 # FastAPI endpoints
  models.py              # Pydantic request/response/domain models
  workflow.py            # LangGraph scan -> patch -> validate -> report pipeline
  patches.py             # Rule-based patch suggestion logic
  validation.py          # Patch safety and structural validation
  reports.py             # Final report generation
  mcp/
    base.py              # Shared command scanner wrapper
    semgrep.py           # Semgrep JSON parser
    trivy.py             # Trivy JSON parser
    zap.py               # ZAP MVP placeholder
  training/
    config.py            # Fine-tuning configuration
    lora.py              # AdaLoRA/SFT trainer factory
    dpo.py               # DPO trainer factory

scripts/
  train.py               # Training entrypoint

tests/
  test_mvp.py            # MVP regression tests
```

## API

### Health Check

```http
GET /health
```

Response:

```json
{"status": "ok"}
```

### Create Scan

```http
POST /scan
```

Request:

```json
{
  "target": "/path/to/local/project",
  "include_zap": false,
  "apply_patches": false
}
```

Response:

```json
{
  "scan_id": "generated-id",
  "status": "completed"
}
```

### Get Report

```http
GET /report/{scan_id}
```

The report contains scanner results, normalized findings, patch suggestions, validation results, and summary counts.

## Running Locally

Install MVP dependencies:

```bash
pip install -r requirements-mvp.txt
```

Run the API:

```bash
uvicorn defapi.api:app --reload
```

Run tests:

```bash
pytest
```

Optional scanner dependencies:

- `semgrep`
- `trivy`
- OWASP ZAP for future dynamic scanning

If a scanner executable is not installed, defAPI returns a skipped scanner result instead of crashing the full scan.

## Example Workflow

```bash
curl -X POST http://127.0.0.1:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"target": "/path/to/project", "include_zap": false}'
```

Then fetch the report:

```bash
curl http://127.0.0.1:8000/report/{scan_id}
```

## LLM Remediation Direction

The current patch generator is intentionally conservative and rule-based. The next milestone is an LLM remediation pipeline:

```text
Finding
  -> vulnerable code context extraction
  -> LLM patch prompt
  -> unified diff generation
  -> diff validation
  -> sandbox patch application
  -> tests
  -> scanner re-run
  -> accepted/rejected training data
```

The generated data can support:

- SFT/LoRA: train the model to produce secure diffs from scanner findings.
- DPO: prefer patches that apply cleanly, pass tests, and resolve scanner findings.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the training and product roadmap.

## Safety Principles

defAPI should never apply LLM-generated patches directly to a user's source tree without validation. A patch must be treated as a candidate until it passes:

- Path containment checks
- Unified diff parsing
- `git apply --check`
- Sandbox application
- Project tests
- Scanner re-run comparison
- Human review for high-risk changes

## Tech Stack

- Python
- FastAPI
- Pydantic
- LangGraph
- Semgrep
- Trivy
- Pytest
- Transformers
- PEFT
- TRL
- Weights & Biases

## Test Status

Current MVP tests:

```text
5 passed
```

## Portfolio Notes

This project demonstrates:

- Backend API design for security tooling
- Scanner adapter design and JSON normalization
- LangGraph workflow orchestration
- Guardrail-first remediation design
- LLM fine-tuning preparation for code security tasks
- Dataset planning for SFT and DPO

For a Korean Notion-style portfolio write-up, see [docs/PORTFOLIO.md](docs/PORTFOLIO.md).
