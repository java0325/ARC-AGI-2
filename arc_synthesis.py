"""
ARC-AGI-2 Program Synthesis Engine
====================================
LLM 기반 규칙 코드 생성 + 로컬 샌드박스 검증 + 반복 정제(Refinement) 루프.

아키텍처
--------
  PromptBuilder      – Task → LLM 프롬프트 문자열
  LLMClient          – OpenAI / Anthropic 공통 인터페이스
  CodeSandbox        – subprocess 기반 안전 실행 + 검증
  ProgramSynthesizer – 생성 → 검증 → 피드백 → 재시도 루프

사용법 (예시)
-------------
  from arc_pipeline import ARCDataLoader
  from arc_synthesis import ProgramSynthesizer, OpenAIClient

  loader = ARCDataLoader()
  tasks  = loader.load_evaluation()

  client     = OpenAIClient(api_key="sk-...", model="o4-mini")
  synthesizer = ProgramSynthesizer(client)

  result = synthesizer.solve(tasks["0934a4d8"])
  print(result)
"""

from __future__ import annotations

import abc
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# arc_pipeline 의 타입을 재사용
sys.path.insert(0, str(Path(__file__).parent))
from arc_pipeline import Grid, Task, Prediction, TaskId

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

DSL_MODULE_DIR = str(Path(__file__).parent)

# 샌드박스에서 import 가능하도록 DSL 공개 API 요약 (프롬프트 삽입용)
DSL_REFERENCE = """
## arc_dsl 라이브러리 레퍼런스 (import 가능)

```python
from arc_dsl import GeometricOps, ObjectOps, ColorOps, SizeAnalyzer, GridDSL

# ─ GeometricOps ─────────────────────────────────────────────────────────
GeometricOps.rotate(grid, degrees)          # degrees: 0/90/180/270
GeometricOps.flip_horizontal(grid)          # 좌우 반전
GeometricOps.flip_vertical(grid)            # 상하 반전
GeometricOps.flip_diagonal_main(grid)       # 주대각선 전치 (\\)
GeometricOps.flip_diagonal_anti(grid)       # 부대각선 전치 (/)
GeometricOps.scale_up(grid, factor)         # 각 셀을 factor×factor 확대
GeometricOps.scale_down(grid, factor)       # factor 마다 1픽셀 추출 축소
GeometricOps.pad(grid, top, bottom, left, right, fill=0)
GeometricOps.crop(grid, r1, c1, r2, c2)     # [r1:r2, c1:c2] 슬라이싱
GeometricOps.trim(grid, background=0)       # 배경 테두리 제거
GeometricOps.overlay(base, patch, r, c, transparent=None)

# ─ ObjectOps ────────────────────────────────────────────────────────────
ObjectOps.find_objects(grid, background=0, connectivity=4)  # → list[ArcObject]
ObjectOps.isolate_shapes(grid, background=0, connectivity=4)# → list[Grid]
ObjectOps.label_grid(grid, background=0, connectivity=4)    # → list[list[int]]
ObjectOps.count_objects(grid, background=0, connectivity=4) # → int
ObjectOps.bounding_box_grid(grid, background=0)             # → (r1,c1,r2,c2)

# ArcObject 속성: .color .cells .bounding_box .height .width .size .to_grid()

# ─ ColorOps ─────────────────────────────────────────────────────────────
ColorOps.most_common_color(grid, exclude=None)
ColorOps.least_common_color(grid, exclude=None)
ColorOps.unique_colors(grid)                                 # → list[int]
ColorOps.replace_color(grid, old_color, new_color)
ColorOps.flood_fill(grid, start=(r,c), new_color, connectivity=4)
ColorOps.color_mask(grid, color)                             # → list[list[bool]]
ColorOps.apply_palette(grid, mapping={old:new, ...})
```
""".strip()


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """샌드박스 실행 결과."""
    success: bool                          # 모든 train 쌍 통과 여부
    passed: int = 0                        # 통과한 train 쌍 수
    total: int = 0                         # 전체 train 쌍 수
    outputs: list[Grid | None] = field(default_factory=list)  # 각 train input 에 대한 실제 출력
    error: str = ""                        # 예외 메시지 (있으면)
    exec_time_ms: float = 0.0


@dataclass
class SynthesisResult:
    """단일 Task 합성 결과."""
    task_id: TaskId
    success: bool                          # 최종 성공 여부
    predictions: list[Prediction]          # test input 별 예측
    attempts: int = 0                      # LLM 호출 횟수
    final_code: str = ""                   # 최종 채택된 코드
    error: str = ""                        # 실패 시 이유

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"[{status}] task={self.task_id}  "
            f"attempts={self.attempts}  "
            f"predictions={len(self.predictions)}"
        )


# ---------------------------------------------------------------------------
# LLMClient 추상 클래스 + 구현체
# ---------------------------------------------------------------------------

@dataclass
class LLMMessage:
    role: str    # "system" | "user" | "assistant"
    content: str


class LLMClient(abc.ABC):
    """OpenAI / Anthropic 공통 인터페이스."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """메시지를 보내고 모델의 텍스트 응답을 반환한다."""

    @staticmethod
    def _extract_code(text: str) -> str:
        """응답에서 ```python ... ``` 블록을 추출한다."""
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        # 코드블록이 없으면 전체를 코드로 간주
        return text.strip()


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions API 클라이언트."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "o4-mini",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("openai 패키지가 필요합니다: pip install openai") from e

        from openai import OpenAI  # type: ignore

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.model = model

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    """Anthropic Messages API 클라이언트."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-5",
    ) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError("anthropic 패키지가 필요합니다: pip install anthropic") from e

        import anthropic  # type: ignore

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.model = model

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        system_msgs = [m for m in messages if m.role == "system"]
        user_msgs   = [m for m in messages if m.role != "system"]

        system_text = "\n\n".join(m.content for m in system_msgs)

        response = self._client.messages.create(
            model=self.model,
            system=system_text or "You are an expert ARC puzzle solver.",
            messages=[{"role": m.role, "content": m.content} for m in user_msgs],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Task 데이터를 LLM 프롬프트로 변환한다."""

    # 색상 코드 → 시각적 블록 문자 (터미널/프롬프트 가독성)
    _COLOR_CHARS = {
        0: ".", 1: "B", 2: "R", 3: "G", 4: "Y",
        5: "W", 6: "M", 7: "O", 8: "A", 9: "P",
    }

    @classmethod
    def _grid_to_text(cls, grid: Grid, use_color_chars: bool = True) -> str:
        """Grid 를 사람이 읽기 쉬운 텍스트로 변환한다."""
        if use_color_chars:
            rows = [" ".join(cls._COLOR_CHARS.get(c, str(c)) for c in row) for row in grid]
        else:
            rows = [" ".join(str(c) for c in row) for row in grid]
        size_label = f"({len(grid)}×{len(grid[0]) if grid else 0})"
        return size_label + "\n" + "\n".join(rows)

    @classmethod
    def _pair_block(cls, idx: int, inp: Grid, out: Grid | None) -> str:
        inp_text = cls._grid_to_text(inp)
        if out is not None:
            out_text = cls._grid_to_text(out)
            return (
                f"### Example {idx + 1}\n"
                f"**Input:**\n```\n{inp_text}\n```\n"
                f"**Output:**\n```\n{out_text}\n```"
            )
        else:
            return (
                f"### Test Input {idx + 1}\n"
                f"```\n{inp_text}\n```"
            )

    @classmethod
    def system_prompt(cls) -> str:
        return textwrap.dedent("""
            You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles.
            ARC grids are 2D lists of integers 0-9 (max 30×30).
            Your task is to infer the transformation rule from input/output examples
            and implement it as a Python function.

            STRICT OUTPUT RULES:
            1. Reply with ONLY a single Python code block (```python ... ```).
            2. The code block must define exactly one function: `transform(grid: list[list[int]]) -> list[list[int]]`.
            3. Do NOT call `transform` inside the code block.
            4. Do NOT include any explanation outside the code block.
            5. You may import from `arc_dsl` (GeometricOps, ObjectOps, ColorOps, etc.).
            6. Use `import copy` if needed; do not use any other external library.

            Color legend: . = 0(black)  B = 1(blue)  R = 2(red)   G = 3(green)
                          Y = 4(yellow) W = 5(grey)  M = 6(magenta) O = 7(orange)
                          A = 8(azure)  P = 9(purple)
        """).strip()

    @classmethod
    def initial_prompt(cls, task: Task) -> str:
        """최초 코드 생성 요청 프롬프트."""
        blocks = [
            cls._pair_block(i, pair.input, pair.output)
            for i, pair in enumerate(task.train)
        ]
        train_section = "\n\n".join(blocks)

        test_blocks = [
            cls._pair_block(i, inp, None)
            for i, inp in enumerate(task.test_inputs)
        ]
        test_section = "\n\n".join(test_blocks)

        return textwrap.dedent(f"""
            ## ARC Task  (task_id: {task.task_id})

            Study the following input→output examples carefully and identify the transformation rule.

            {train_section}

            ---

            {test_section}

            ---

            {DSL_REFERENCE}

            ---

            Now write the `transform(grid)` function that applies the rule you identified.
            Verify mentally that your function produces the correct output for EVERY example above.
        """).strip()

    @classmethod
    def refinement_prompt(
        cls,
        task: Task,
        prev_code: str,
        sandbox_result: SandboxResult,
    ) -> str:
        """검증 실패 시 피드백과 함께 코드 수정 요청 프롬프트."""
        diff_lines: list[str] = []

        for i, (pair, actual) in enumerate(
            zip(task.train, sandbox_result.outputs)
        ):
            expected_text = cls._grid_to_text(pair.output)  # type: ignore[arg-type]
            if actual is None:
                diff_lines.append(
                    f"Example {i+1}: RUNTIME ERROR — actual output was None"
                )
            else:
                actual_text = cls._grid_to_text(actual)
                if actual != pair.output:
                    diff_lines.append(
                        f"Example {i+1}:\n"
                        f"  Expected:\n```\n{expected_text}\n```\n"
                        f"  Got:\n```\n{actual_text}\n```"
                    )

        diff_section = "\n\n".join(diff_lines) if diff_lines else "(no diff available)"

        error_section = ""
        if sandbox_result.error:
            error_section = f"\n**Runtime error:**\n```\n{sandbox_result.error}\n```\n"

        return textwrap.dedent(f"""
            Your previous implementation is INCORRECT.
            It passed {sandbox_result.passed}/{sandbox_result.total} training examples.
            {error_section}
            ## Failures

            {diff_section}

            ## Your Previous Code

            ```python
            {prev_code}
            ```

            Please carefully analyze the failures above, reconsider the transformation rule,
            and provide a corrected `transform(grid)` function.
            Output ONLY the corrected Python code block.
        """).strip()


# ---------------------------------------------------------------------------
# CodeSandbox
# ---------------------------------------------------------------------------

class CodeSandbox:
    """LLM 생성 코드를 subprocess 로 격리 실행하고 train 검증 결과를 반환한다."""

    @staticmethod
    def _build_runner_src(
        dsl_dir: str,
        code: str,
        train_pairs: list[list[Any]],
    ) -> str:
        pairs_repr = repr(train_pairs)
        return (
            "import sys, json, traceback, copy\n"
            f"sys.path.insert(0, {dsl_dir!r})\n\n"
            f"{code}\n\n"
            f"_train_pairs = {pairs_repr}\n"
            "_results = []\n"
            "for _inp, _expected in _train_pairs:\n"
            "    try:\n"
            "        _actual = transform(copy.deepcopy(_inp))\n"
            "        _passed = (_actual == _expected)\n"
            "        _results.append({'passed': _passed, 'output': _actual, 'error': ''})\n"
            "    except Exception as _e:\n"
            "        _results.append({'passed': False, 'output': None, 'error': traceback.format_exc()})\n"
            "print(json.dumps(_results))\n"
        )

    def __init__(self, timeout_sec: float = 10.0) -> None:
        self.timeout_sec = timeout_sec

    def _build_runner(self, code: str, train_pairs: list[tuple[Grid, Grid]]) -> str:
        """실행할 runner 스크립트 문자열을 생성한다."""
        return self._build_runner_src(
            DSL_MODULE_DIR,
            code,
            [[inp, out] for inp, out in train_pairs],
        )

    def run(self, code: str, task: Task) -> SandboxResult:
        """code 를 실행하여 모든 train 쌍에 대해 검증한 결과를 반환한다."""
        train_pairs: list[tuple[Grid, Grid]] = [
            (pair.input, pair.output)  # type: ignore[misc]
            for pair in task.train
            if pair.output is not None
        ]

        runner_src = self._build_runner(code, train_pairs)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(runner_src)
            tmp_path = tmp.name

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if proc.returncode != 0:
                return SandboxResult(
                    success=False,
                    total=len(train_pairs),
                    outputs=[None] * len(train_pairs),
                    error=proc.stderr.strip()[:2000],
                    exec_time_ms=elapsed_ms,
                )

            raw = proc.stdout.strip()
            if not raw:
                return SandboxResult(
                    success=False,
                    total=len(train_pairs),
                    outputs=[None] * len(train_pairs),
                    error="No output from sandbox",
                    exec_time_ms=elapsed_ms,
                )

            results: list[dict[str, Any]] = json.loads(raw)
            outputs: list[Grid | None] = [r["output"] for r in results]
            errors   = [r["error"] for r in results if r["error"]]
            passed   = sum(1 for r in results if r["passed"])

            return SandboxResult(
                success=(passed == len(train_pairs)),
                passed=passed,
                total=len(train_pairs),
                outputs=outputs,
                error="\n".join(errors)[:2000],
                exec_time_ms=elapsed_ms,
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                total=len(train_pairs),
                outputs=[None] * len(train_pairs),
                error=f"Timeout ({self.timeout_sec}s) exceeded",
            )
        except json.JSONDecodeError as e:
            return SandboxResult(
                success=False,
                total=len(train_pairs),
                outputs=[None] * len(train_pairs),
                error=f"JSON parse error: {e}",
            )
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    def apply(self, code: str, inputs: list[Grid]) -> list[Grid | None]:
        """검증 완료된 code 를 임의의 input 목록에 적용한다."""
        runner_src = (
            f"import sys, json, traceback, copy\n"
            f"sys.path.insert(0, {DSL_MODULE_DIR!r})\n\n"
            f"{code}\n\n"
            f"test_inputs = {inputs!r}\n"
            f"outputs = []\n"
            f"for inp in test_inputs:\n"
            f"    try:\n"
            f"        outputs.append(transform(copy.deepcopy(inp)))\n"
            f"    except Exception:\n"
            f"        outputs.append(None)\n"
            f"print(json.dumps(outputs))\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(runner_src)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True,
                timeout=self.timeout_sec,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return [None] * len(inputs)
            return json.loads(proc.stdout.strip())
        except Exception:
            return [None] * len(inputs)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# ProgramSynthesizer
# ---------------------------------------------------------------------------

class ProgramSynthesizer:
    """LLM + 샌드박스 반복 루프로 ARC Task 를 푼다."""

    def __init__(
        self,
        client: LLMClient,
        max_attempts: int = 5,
        sandbox_timeout: float = 10.0,
        verbose: bool = True,
    ) -> None:
        self.client  = client
        self.max_attempts   = max_attempts
        self.sandbox = CodeSandbox(timeout_sec=sandbox_timeout)
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _make_fallback_predictions(self, task: Task) -> list[Prediction]:
        """코드 합성 실패 시 0으로 채운 더미 예측을 반환한다."""
        preds: list[Prediction] = []
        for inp in task.test_inputs:
            r, c = len(inp), len(inp[0]) if inp else 1
            blank: Grid = [[0] * c for _ in range(r)]
            preds.append(Prediction(attempt_1=blank, attempt_2=blank))
        return preds

    def solve(self, task: Task) -> SynthesisResult:
        """단일 Task 를 해결하고 SynthesisResult 를 반환한다."""
        self._log(f"\n{'='*60}")
        self._log(f"[Synthesizer] Task: {task.task_id}  "
                  f"(train={len(task.train)}, test={len(task.test_inputs)})")
        self._log(f"{'='*60}")

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=PromptBuilder.system_prompt()),
            LLMMessage(role="user",   content=PromptBuilder.initial_prompt(task)),
        ]

        last_code = ""
        last_sandbox: SandboxResult | None = None

        for attempt in range(1, self.max_attempts + 1):
            self._log(f"\n[Attempt {attempt}/{self.max_attempts}] LLM 호출 중...")

            # ── 1. LLM 코드 생성 ─────────────────────────────────────────
            try:
                raw_response = self.client.chat(
                    messages,
                    temperature=0.0 if attempt == 1 else 0.3,
                )
            except Exception as e:
                self._log(f"  LLM 오류: {e}")
                break

            code = LLMClient._extract_code(raw_response)
            last_code = code
            self._log(f"  코드 추출 완료 ({len(code)} chars)")

            # ── 2. 샌드박스 검증 ─────────────────────────────────────────
            self._log("  샌드박스 검증 중...")
            sandbox_result = self.sandbox.run(code, task)
            last_sandbox = sandbox_result

            self._log(
                f"  결과: {sandbox_result.passed}/{sandbox_result.total} 통과"
                f"  ({sandbox_result.exec_time_ms:.1f} ms)"
            )
            if sandbox_result.error:
                self._log(f"  오류: {sandbox_result.error[:300]}")

            # ── 3. 통과 시 test 적용 ─────────────────────────────────────
            if sandbox_result.success:
                self._log("  ✓ 모든 train 통과! test 에 적용 중...")
                raw_preds = self.sandbox.apply(code, task.test_inputs)

                predictions: list[Prediction] = []
                for raw in raw_preds:
                    if raw is None:
                        r = len(task.test_inputs[0]) if task.test_inputs else 1
                        c = len(task.test_inputs[0][0]) if task.test_inputs else 1
                        raw = [[0] * c for _ in range(r)]
                    predictions.append(Prediction(attempt_1=raw, attempt_2=raw))

                return SynthesisResult(
                    task_id=task.task_id,
                    success=True,
                    predictions=predictions,
                    attempts=attempt,
                    final_code=code,
                )

            # ── 4. 실패 시 피드백 메시지 추가 ────────────────────────────
            if attempt < self.max_attempts:
                self._log("  피드백 구성 중...")
                messages.append(LLMMessage(role="assistant", content=raw_response))
                messages.append(LLMMessage(
                    role="user",
                    content=PromptBuilder.refinement_prompt(task, code, sandbox_result),
                ))

        # ── 최대 시도 초과 ────────────────────────────────────────────────
        self._log(f"\n  ✗ {self.max_attempts}회 시도 후 실패. 더미 예측 반환.")
        error_msg = (
            f"failed after {self.max_attempts} attempts. "
            f"last: {last_sandbox.passed if last_sandbox else 0}/"
            f"{last_sandbox.total if last_sandbox else len(task.train)} passed"
        )
        return SynthesisResult(
            task_id=task.task_id,
            success=False,
            predictions=self._make_fallback_predictions(task),
            attempts=self.max_attempts,
            final_code=last_code,
            error=error_msg,
        )

    def solve_all(
        self,
        tasks: dict[TaskId, Task],
        max_tasks: int | None = None,
    ) -> dict[TaskId, SynthesisResult]:
        """여러 Task 를 순서대로 해결하고 결과 dict 를 반환한다."""
        results: dict[TaskId, SynthesisResult] = {}
        items = list(tasks.items())
        if max_tasks is not None:
            items = items[:max_tasks]

        for i, (task_id, task) in enumerate(items, 1):
            self._log(f"\n[{i}/{len(items)}] Solving task: {task_id}")
            results[task_id] = self.solve(task)

        solved = sum(1 for r in results.values() if r.success)
        self._log(f"\n{'='*60}")
        self._log(f"완료: {solved}/{len(results)} tasks 해결")
        self._log(f"{'='*60}")
        return results


# ---------------------------------------------------------------------------
# Pipeline 통합 헬퍼
# ---------------------------------------------------------------------------

def results_to_predictions(
    results: dict[TaskId, SynthesisResult],
) -> dict[TaskId, list[Prediction]]:
    """SynthesisResult dict 를 ARCPipeline.writer 에 전달할 형태로 변환한다."""
    return {task_id: res.predictions for task_id, res in results.items()}


# ---------------------------------------------------------------------------
# 빠른 사용 예시 (파일 직접 실행 시)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 실제 실행: OPENAI_API_KEY 환경변수 필요
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정하세요.")
        sys.exit(1)

    from arc_pipeline import ARCDataLoader, ARCSubmissionWriter

    loader = ARCDataLoader()
    tasks  = loader.load_evaluation()

    client      = OpenAIClient(api_key=api_key, model="o4-mini")
    synthesizer = ProgramSynthesizer(client, max_attempts=5, verbose=True)

    # 처음 3개 task 만 테스트
    results = synthesizer.solve_all(tasks, max_tasks=3)

    predictions = results_to_predictions(results)
    writer = ARCSubmissionWriter()
    writer.save(predictions, "submission_synthesis.json")
