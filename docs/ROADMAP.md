# defAPI Roadmap

This roadmap is organized around the path from MVP scanner API to LLM-powered security remediation agent.

## Phase 1: MVP Stabilization

Goal: make the scanner pipeline reliable and easy to evaluate.

Tasks:

- Replace in-memory scan records with persistent storage.
- Convert `/scan` into an asynchronous background job.
- Add scanner fixture tests for Semgrep and Trivy JSON payloads.
- Add report schema versioning.
- Add structured error codes for skipped, failed, timed out, and completed scanners.
- Add dataset logging for every scan.

Expected output:

- Stable scan/report API.
- Reproducible scanner parsing.
- JSONL artifacts for future training data.

## Phase 2: Patch Validation Hardening

Goal: make patch suggestions safe enough for developer review.

Tasks:

- Parse unified diffs instead of checking only string headers.
- Add `git apply --check`.
- Create a temporary sandbox worktree per scan.
- Apply candidate patches only inside the sandbox.
- Add configurable test commands.
- Re-run scanners after patch application.
- Compare original findings against post-patch findings.

Patch acceptance criteria:

- Diff applies cleanly.
- Tests pass.
- Original finding is resolved.
- No new critical or high findings are introduced.
- Patch size is within a reasonable limit.

## Phase 3: LLM Patch Generation

Goal: generate candidate secure diffs from normalized findings.

New modules:

```text
defapi/context.py
defapi/llm/base.py
defapi/llm/prompts.py
defapi/llm/patcher.py
defapi/sandbox.py
defapi/datasets.py
```

Pipeline:

```text
Finding
  -> extract vulnerable code context
  -> create patch prompt
  -> generate unified diff
  -> validate diff
  -> sandbox apply
  -> run tests
  -> scanner re-run
  -> report accepted/rejected result
```

Prompt requirements:

- Return unified diff only.
- Modify the minimum necessary code.
- Preserve public APIs.
- Do not suppress scanner rules unless explicitly allowed.
- Do not delete tests or validation logic.
- Do not make broad dependency upgrades without a fixed version.

## Phase 4: SFT / LoRA Training

Goal: train an open-source code LLM to produce secure patches from scanner findings.

Input data:

```json
{
  "prompt": "Scanner finding + vulnerable code context + rules",
  "completion": "Explanation and secure unified diff"
}
```

Recommended first experiment:

- Base model: small code model first, such as 1B-3B class.
- Method: QLoRA or AdaLoRA.
- Dataset: validated patch examples only.
- Validation: fixed benchmark set with scanner re-run checks.

Do not optimize for low training loss alone. Optimize for validated patch success.

Core metrics:

- diff format success rate
- `git apply --check` success rate
- test pass rate
- scanner finding resolution rate
- new high/critical finding rate
- human acceptance rate

## Phase 5: DPO Training

Goal: teach the model to prefer safe, minimal, validated patches over unsafe or invalid patches.

DPO data format:

```json
{
  "prompt": "Finding + code context",
  "chosen": "validated secure patch",
  "rejected": "invalid, unsafe, too broad, or failing patch"
}
```

Good rejected examples:

- Diff does not apply.
- Patch hides the warning with `# nosec` or ignore comments.
- Patch removes validation or authentication.
- Patch changes unrelated behavior.
- Patch fails tests.
- Patch leaves the original finding unresolved.

Minimum requirement before DPO:

- A fixed benchmark set.
- At least several hundred high-quality preference pairs.
- Clear validation labels.
- Human-reviewed chosen examples for high-risk findings.

## Phase 6: Productization

Goal: turn defAPI into a practical developer-facing security assistant.

Features:

- GitHub PR integration.
- PR comments with finding explanation and suggested diff.
- Developer approval flow.
- CI-based validation.
- Scan history dashboard.
- Team policy configuration.
- Severity-based approval rules.
- Audit logs for every generated patch.

Default product behavior should be review-first, not auto-apply.

## Near-Term TODO

- Add `README.md` portfolio documentation.
- Add architecture and roadmap docs.
- Add scanner parser fixture tests.
- Add dataset export module.
- Add `git apply --check` validation.
- Add background scan jobs.
- Add persistent scan store.
- Add LLM patch prompt templates.
- Add benchmark examples for known vulnerable code patterns.
