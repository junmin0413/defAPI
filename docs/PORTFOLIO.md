# Projects

# defAPI

> **LLM 기반 취약 코드 분석 및 보안 패치 자동화 API**
> **2026.05 ~ 진행 중**

팀 구성

> **AI / Backend / Security - 1명**

**Github**

[defAPI](https://github.com/junmin0413/defAPI)

사용한 기술

- **`FastAPI`**
- **`Python`**
- **`Pydantic`**
- **`LangGraph`**
- **`Semgrep`**
- **`Trivy`**
- **`OWASP ZAP`**
- **`Transformers`**
- **`PEFT`**
- **`TRL`**
- **`LoRA / DPO`**
- **`Pytest`**
- **`Docker`**

## 개요

## 보안 스캐너 결과를 기반으로 취약 코드를 분석하고, LLM이 수정 가능한 패치 후보를 생성하도록 설계한 보안 자동화 API

개발 과정에서 취약점 스캐너를 사용하면 문제 위치와 경고는 확인할 수 있지만, 실제로 어떤 방식으로 코드를 수정해야 하는지 판단하는 과정은 여전히 개발자에게 남아 있습니다.

특히 Semgrep, Trivy, ZAP 같은 도구들은 각각 출력 형식과 탐지 대상이 다르기 때문에 여러 스캐너 결과를 하나의 서비스에서 일관되게 다루기 어렵습니다. 또한 LLM이 보안 패치를 생성하더라도, 해당 패치가 실제로 적용 가능한지, 테스트를 통과하는지, 기존 취약점이 해결되었는지 검증하지 않으면 실서비스에 사용하기 어렵습니다.

이 문제에서 출발해 defAPI를 기획했습니다.

defAPI는 로컬 프로젝트를 대상으로 보안 스캐너를 실행하고, 스캐너별 결과를 공통 `Finding` 스키마로 정규화한 뒤, 패치 제안과 검증 결과를 포함한 리포트를 생성하는 FastAPI 기반 보안 분석 MVP입니다.

최종 목표는 단순 취약점 탐지를 넘어, 다음과 같은 흐름을 자동화하는 개발자용 보안 에이전트입니다.

```text
취약 코드 스캔
→ Finding 정규화
→ 취약 코드 context 추출
→ LLM 패치 후보 생성
→ diff 검증
→ 테스트 실행
→ 스캐너 재실행
→ 검증된 보안 리포트 생성
```

저는 이 프로젝트에서 **AI/Backend/Security 파이프라인 전체 설계 및 구현**을 담당했습니다.

FastAPI API 설계, LangGraph 기반 워크플로우 구성, Semgrep/Trivy/ZAP scanner wrapper 구현, Pydantic 기반 결과 스키마 설계, rule-based patch suggestion, validation loop, 향후 LoRA/SFT 및 DPO 학습을 위한 training module skeleton까지 직접 구현했습니다.

## 왜 필요한가?

보안 스캐너는 취약점을 찾는 데는 강하지만, 개발자가 바로 적용할 수 있는 안전한 수정안을 제공하는 데는 한계가 있습니다.

예를 들어 Semgrep은 코드 패턴 기반 취약점을 탐지하고, Trivy는 dependency CVE나 misconfiguration을 찾아주지만, 각 도구의 JSON 구조가 다르고, severity나 위치 정보, CWE, reference 형식이 서로 다릅니다.

또한 LLM을 활용해 보안 패치를 생성하려면 단순히 "이 코드를 고쳐줘"라고 요청하는 것만으로는 부족합니다.

LLM이 생성한 diff는 다음 기준을 통과해야 합니다.

- 실제 파일에 적용 가능한 diff인지
- scan target 밖의 파일을 수정하지 않는지
- 취약점을 숨기기 위해 ignore 주석만 추가하지 않았는지
- 테스트를 깨지 않는지
- 스캐너를 다시 실행했을 때 원래 finding이 해결되는지

defAPI는 이러한 문제를 해결하기 위해 보안 스캐너, 패치 생성, 검증, 리포트, 학습 데이터 축적을 하나의 파이프라인으로 연결하는 것을 목표로 합니다.

## 담당한 역할

### AI 및 Backend / Security Pipeline 개발

- FastAPI 기반 보안 분석 API 설계 및 구현
- `/scan`, `/report/{scan_id}`, `/health` API 개발
- LangGraph 기반 `scan → patch → validate → report` 워크플로우 구성
- Semgrep, Trivy, ZAP scanner wrapper 설계
- scanner별 JSON 결과를 공통 `Finding` 모델로 정규화
- Pydantic 기반 request/response/domain model 설계
- rule-based patch suggestion 생성 로직 구현
- patch target path containment 검증 구현
- unified diff 구조 검증 로직 구현
- scanner timeout 및 skipped/failed/completed 상태 처리
- LoRA/SFT 및 DPO 학습 모듈 구조 분리
- Pytest 기반 MVP 회귀 테스트 작성
- 포트폴리오용 README, architecture, roadmap 문서 작성

## 서비스 아키텍처

```text
Client
  → FastAPI /scan
  → ScanRecord 생성
  → ScanWorkflow 실행
      → SemgrepMCP
      → TrivyMCP
      → ZapMCP
  → Finding 정규화
  → PatchGenerator
  → ValidationLoop
  → ReportGenerator
  → FastAPI /report/{scan_id}
```

## 결과물

<aside>

#### 보안 분석 API

---

## `/scan` API

- 사용자가 로컬 프로젝트 경로를 전달하면 해당 target을 대상으로 보안 스캔을 실행합니다.
- `ScanRequest` 모델에서 target 경로가 실제로 존재하는지 검증합니다.
- `include_zap` 옵션을 통해 ZAP scan 포함 여부를 선택할 수 있도록 설계했습니다.
- 스캔 요청은 내부적으로 `ScanRecord`로 관리되며, scan id를 기준으로 리포트를 조회할 수 있습니다.

```json
{
  "target": "/path/to/project",
  "include_zap": false,
  "apply_patches": false
}
```

## `/report/{scan_id}` API

- 특정 scan id에 대한 최종 보안 리포트를 반환합니다.
- 리포트에는 scanner 결과, finding 목록, patch suggestion, validation result, summary count가 포함됩니다.
- 스캔 실패, 아직 실행 중인 상태, 존재하지 않는 scan id에 대해 각각 다른 HTTP 응답을 반환하도록 처리했습니다.

</aside>

<aside>

#### Scanner MCP Wrapper

---

## SemgrepMCP

- `semgrep --config auto --json --quiet {target}` 명령을 실행합니다.
- Semgrep JSON의 `results`를 순회하며 `check_id`, `severity`, `message`, `path`, `start.line`, `end.line`, `metadata.cwe`, `metadata.references`를 공통 `Finding` 모델로 변환합니다.
- Semgrep의 `ERROR`, `WARNING` severity를 내부 severity enum으로 매핑했습니다.

## TrivyMCP

- `trivy fs --format json --quiet {target}` 명령을 실행합니다.
- Trivy의 `Vulnerabilities`와 `Misconfigurations`를 모두 파싱합니다.
- dependency 취약점의 `PkgName`, `InstalledVersion`, `FixedVersion` 정보는 추후 dependency upgrade patch 생성에 활용할 수 있도록 `raw`에 보존했습니다.

## ZapMCP

- MVP 단계에서는 실제 ZAP scan을 실행하지 않고 `skipped` scanner result를 반환합니다.
- 동적 스캔은 URL target, 인증 세션, active scan opt-in 등 안전장치가 필요하기 때문에 placeholder로 분리했습니다.

</aside>

<aside>

#### Patch Suggestion / Validation

---

## PatchGenerator

- 각 finding마다 `finding_key`를 생성하고, scanner 종류와 raw metadata를 기반으로 patch suggestion을 생성합니다.
- Semgrep finding은 rule-based remediation guidance를 생성합니다.
- Trivy dependency finding은 `PkgName`, `InstalledVersion`, `FixedVersion`이 있을 경우 간단한 dependency upgrade unified diff를 생성할 수 있도록 구성했습니다.

## ValidationLoop

- patch suggestion이 실제 파일을 대상으로 하는지 확인합니다.
- patch target이 scan target 내부에 있는지 `relative_to()` 기반으로 검증합니다.
- LLM 또는 rule-based generator가 생성한 diff가 최소한 unified diff 구조를 갖는지 검증합니다.
- 현재는 구조 검증 단계이며, 이후 `git apply --check`, sandbox apply, test runner, scanner re-run 검증으로 확장할 계획입니다.

</aside>

<aside>

#### LLM 학습 준비 구조

---

## Training Module

학습 관련 코드는 runtime scanner API와 분리하기 위해 `defapi/training/` 패키지로 정리했습니다.

```text
defapi/training/
  config.py
  lora.py
  dpo.py
```

## LoRA / SFT

- `FineTuneConfig`를 통해 base model, dataset path, output path, AdaLoRA hyperparameter를 관리합니다.
- `lora.py`에는 tokenizer, model, bitsandbytes config, AdaLoRA config, SFTTrainer 생성 로직을 분리했습니다.
- GPU 확보 후 scanner finding + vulnerable code context + secure diff 형태의 데이터셋으로 SFT를 진행할 계획입니다.

## DPO

- `dpo.py`에는 DPOTrainer 생성 구조를 분리했습니다.
- 향후 validation을 통과한 patch를 `chosen`, 실패한 patch를 `rejected`로 구성해 preference learning을 진행할 계획입니다.

</aside>

## 핵심 구현 기능

<aside>

#### 1. 보안 스캐너 결과 정규화 파이프라인

defAPI의 핵심은 서로 다른 scanner 결과를 하나의 내부 schema로 통합하는 것입니다.

보안 스캐너들은 각자 다른 JSON 구조를 반환합니다.

- Semgrep: `results`, `check_id`, `extra`, `metadata`
- Trivy: `Results`, `Vulnerabilities`, `Misconfigurations`
- ZAP: 향후 dynamic scan alert format

이 구조가 그대로 downstream logic으로 흘러가면 patch generation, validation, report generation이 scanner별 분기 코드로 복잡해질 수 있습니다.

이를 해결하기 위해 모든 scanner 결과를 다음 공통 모델로 변환했습니다.

```python
class Finding(BaseModel):
    scanner: ScannerName
    rule_id: str
    severity: FindingSeverity
    title: str
    message: str
    file_path: str | None
    start_line: int | None
    end_line: int | None
    cwe: list[str]
    references: list[str]
    raw: dict[str, Any]
```

이후 patch generation과 report generation은 scanner 원본 JSON이 아니라 공통 `Finding`만 바라보도록 구성했습니다.

### 결과

- scanner별 JSON 구조 차이를 wrapper 내부로 격리했습니다.
- downstream pipeline이 단순해졌습니다.
- 향후 Bandit, npm audit, pip-audit 같은 scanner를 추가하더라도 `Finding` 변환만 구현하면 확장할 수 있는 구조가 되었습니다.

</aside>

<aside>

#### 2. LangGraph 기반 보안 분석 워크플로우

초기 MVP라도 scan, patch, validate, report 단계가 명확히 분리되어야 이후 LLM patch generation과 dataset logging을 붙이기 쉽다고 판단했습니다.

그래서 단순 함수 호출 흐름이 아니라 LangGraph로 pipeline node를 분리했습니다.

```text
scan
→ patch
→ validate
→ report
```

각 단계의 역할은 다음과 같습니다.

- `scan`: Semgrep/Trivy/ZAP wrapper 실행 및 findings aggregation
- `patch`: finding 기반 patch suggestion 생성
- `validate`: patch suggestion 구조 검증
- `report`: scanner result, patch, validation, summary를 하나의 report로 구성

### 결과

- 파이프라인 단계별 책임이 명확해졌습니다.
- LLM patch generation node, sandbox validation node, dataset logging node를 추가하기 쉬운 구조가 되었습니다.
- 테스트에서 scanner를 mock으로 교체해 workflow 단위 검증이 가능해졌습니다.

</aside>

<aside>

#### 3. LLM 보안 패치 생성을 위한 Guardrail 설계

LLM이 생성한 보안 패치는 바로 적용하면 위험합니다.

예를 들어 LLM은 다음과 같은 잘못된 수정을 만들 수 있습니다.

- 취약점을 해결하지 않고 ignore 주석만 추가
- 인증/검증 로직 삭제
- target 밖의 파일 수정
- dependency를 무조건 latest로 올림
- 테스트를 삭제하거나 우회

이를 막기 위해 defAPI에서는 patch suggestion을 바로 적용하지 않고, validation result와 함께 report에 포함하는 구조로 설계했습니다.

현재 구현된 guardrail은 다음과 같습니다.

- patch target file 존재 여부 확인
- scan target 내부 파일인지 확인
- unified diff 최소 구조 확인
- scanner 실행 timeout 처리
- scanner executable 미설치 시 전체 scan 실패 대신 skipped 처리

향후 추가할 guardrail은 다음과 같습니다.

- `git apply --check`
- temporary worktree sandbox apply
- project test command 실행
- scanner re-run 후 original finding resolved 여부 확인
- new high/critical finding 증가 여부 확인
- human review required flag

</aside>

## 문제 해결 과정

<aside>

#### ⚠️ 문제 1

### 스캐너마다 JSON 결과 형식이 달라 후속 파이프라인이 복잡해지는 문제

- Semgrep과 Trivy는 탐지 대상과 JSON 구조가 완전히 다릅니다.
- Semgrep은 코드 위치와 rule metadata 중심이고, Trivy는 dependency CVE와 misconfiguration 중심입니다.
- scanner 원본 JSON을 그대로 사용하면 patch generation, validation, report 단계에서 scanner별 분기가 계속 늘어나는 문제가 있었습니다.

---

### 해결 방법

- `CommandMCP` base class를 만들고 scanner 실행, timeout, JSON parsing, error handling을 공통화했습니다.
- scanner별 wrapper는 command 생성과 `parse_findings()`만 구현하도록 분리했습니다.
- 모든 scanner 결과를 공통 `Finding` 모델로 정규화했습니다.
- scanner 원본 payload는 `raw` 필드에 보존해 이후 정밀 분석이나 학습 데이터 생성에 사용할 수 있도록 했습니다.

---

### 결과

- scanner adapter와 downstream pipeline의 결합도를 낮췄습니다.
- 새로운 scanner를 추가할 때 필요한 구현 범위가 명확해졌습니다.
- report, patch, validation 단계가 scanner별 원본 JSON 구조에 의존하지 않게 되었습니다.

</aside>

<aside>

#### ⚠️ 문제 2

### `include_zap` 옵션이 실제 동작과 맞지 않는 문제

- 초기 workflow에서는 `include_zap=False`여도 ZAP scanner가 항상 실행되는 구조였습니다.
- ZAP은 MVP에서 placeholder 상태였기 때문에, 사용자가 요청하지 않아도 report에 `zap_skipped`가 포함되는 문제가 있었습니다.
- API 옵션의 의미와 실제 workflow 동작이 불일치했습니다.

---

### 해결 방법

- workflow의 scan 단계에서 `record.include_zap`이 `True`일 때만 ZAP scanner를 추가하도록 수정했습니다.
- `include_zap=False`인 경우 Semgrep과 Trivy만 실행되도록 정리했습니다.
- 테스트에서 ZAP이 요청된 경우와 요청되지 않은 경우를 분리해 검증했습니다.

---

### 결과

- API 옵션과 실제 동작이 일치하게 되었습니다.
- MVP 단계에서 불필요한 ZAP skipped result가 report에 포함되지 않게 되었습니다.
- 향후 실제 ZAP 연동 시에도 opt-in 방식으로 안전하게 확장할 수 있게 되었습니다.

</aside>

<aside>

#### ⚠️ 문제 3

### LLM 패치를 바로 적용하면 위험한 문제

- LLM은 그럴듯한 diff를 생성할 수 있지만, 실제로 적용되지 않거나 취약점을 해결하지 못할 수 있습니다.
- 보안 패치에서는 잘못된 수정이 기능 버그뿐 아니라 새로운 취약점으로 이어질 수 있습니다.
- 따라서 patch generation보다 patch validation이 더 중요하다고 판단했습니다.

---

### 해결 방법

- 현재 MVP에서는 patch를 직접 적용하지 않고 `PatchSuggestion`으로만 report에 포함했습니다.
- validation 단계에서 target path containment와 unified diff 구조를 검증했습니다.
- 향후 `git apply --check`, sandbox apply, test runner, scanner re-run 검증을 roadmap에 포함했습니다.

---

### 결과

- MVP 단계에서도 "자동 적용"이 아니라 "검증 가능한 후보 제안" 중심의 안전한 구조를 유지했습니다.
- LLM patch generation을 붙이더라도 validation node를 통해 위험한 patch를 걸러낼 수 있는 기반을 만들었습니다.

</aside>

## 배운 점

<aside>

### AI 보안 자동화에서는 생성보다 검증이 중요하다는 점

LLM을 활용하면 취약 코드에 대한 수정안을 빠르게 생성할 수 있지만, 보안 영역에서는 생성된 결과를 신뢰하는 순간 위험해질 수 있습니다.

이번 프로젝트를 설계하면서 LLM patch는 최종 답이 아니라 candidate diff로 다뤄야 하며, 실제 가치는 `git apply`, test, scanner re-run 같은 검증 루프에서 나온다는 점을 배웠습니다.

</aside>

<aside>

### 외부 도구 연동에서는 adapter boundary가 중요하다는 점

Semgrep, Trivy, ZAP은 모두 보안 도구지만 입력 방식, 출력 형식, 실패 방식이 다릅니다.

이를 그대로 서비스 로직에 섞으면 기능이 늘어날수록 유지보수가 어려워집니다. scanner별 차이는 MCP wrapper 내부에 가두고, 내부 파이프라인은 공통 `Finding` 모델만 사용하도록 설계하면서 adapter boundary의 중요성을 체감했습니다.

</aside>

<aside>

### LLM 학습 전에 데이터 파이프라인이 먼저 필요하다는 점

처음에는 GPU를 확보하면 바로 LoRA/SFT와 DPO를 진행할 수 있다고 생각했지만, 실제로는 좋은 학습 데이터를 만들기 위한 기준이 먼저 필요했습니다.

어떤 patch가 좋은 patch인지, 어떤 patch를 rejected로 볼 것인지, scanner finding이 실제로 해결되었는지 평가할 수 있어야 DPO 데이터셋도 의미가 있습니다.

이 프로젝트를 통해 모델 학습 자체보다 scanner result, patch, validation result를 꾸준히 축적하는 데이터 파이프라인이 선행되어야 한다는 점을 배웠습니다.

</aside>

<aside>

### MVP에서도 테스트 가능한 구조가 중요하다는 점

보안 스캐너는 로컬 설치 여부나 실행 환경에 따라 결과가 달라질 수 있습니다.

그래서 workflow 테스트에서는 scanner 실행 자체보다 scanner wrapper를 mock으로 교체하고, report summary와 validation 결과를 검증하는 방식으로 테스트했습니다.

이를 통해 외부 도구에 의존하는 프로젝트에서도 핵심 비즈니스 로직은 안정적으로 테스트할 수 있다는 점을 배웠습니다.

</aside>
