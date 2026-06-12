"""
ARC-AGI-2 Ensemble & 2-Attempts Strategy
==========================================
다형성 예측(Diverse Predictions) + 백업 휴리스틱(Fallback) 조합으로
2번의 시도(attempt_1, attempt_2)를 최적화한다.

클래스 구성
-----------
  HeuristicSolver    – 규칙 기반 폴백 그리드 생성
  SolvedCode         – 통과 코드 하나를 감싸는 컨테이너
  DiverseSynthesizer – 최대 2개의 서로 다른 통과 코드를 수집
  EnsemblePredictor  – (passing_codes, fallback) → list[Prediction]
  EnsembleResult     – 단일 Task 앙상블 결과
"""

from __future__ import annotations

import copy
import sys
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent))
from arc_pipeline import Grid, Task, Prediction, TaskId
from arc_synthesis import (
    CodeSandbox,
    LLMClient,
    LLMMessage,
    PromptBuilder,
    ProgramSynthesizer,
    SandboxResult,
    SynthesisResult,
)

# ---------------------------------------------------------------------------
# 전략 레이블
# ---------------------------------------------------------------------------

Strategy = Literal[
    "diverse_2",          # 서로 다른 통과 코드 2개
    "single_fallback",    # 통과 코드 1개 + 폴백
    "partial_heuristic",  # 부분 통과 코드 + 휴리스틱
    "full_heuristic",     # 통과 코드 없음, 전부 휴리스틱
]


# ---------------------------------------------------------------------------
# HeuristicSolver
# ---------------------------------------------------------------------------

class HeuristicSolver:
    """LLM 합성이 실패할 때 사용하는 규칙 기반 폴백 예측기."""

    # ── 출력 크기 예측 ────────────────────────────────────────────────────

    @staticmethod
    def predict_output_size(task: Task) -> tuple[int, int] | None:
        """학습 예제의 input/output 크기 관계에서 출력 크기를 추론한다."""
        pairs = [(p.input, p.output) for p in task.train if p.output is not None]
        if not pairs:
            return None

        out_sizes = [(len(o), len(o[0])) for _, o in pairs]

        # 1) 모든 출력 크기가 동일 → 고정 크기
        if len(set(out_sizes)) == 1:
            return out_sizes[0]

        # 2) 상수 배율 존재 → 배율 적용
        in_sizes = [(len(i), len(i[0])) for i, _ in pairs]
        row_scales = [Fraction(o[0], i[0]) for i, o in zip(in_sizes, out_sizes)]
        col_scales = [Fraction(o[1], i[1]) for i, o in zip(in_sizes, out_sizes)]

        if len(set(row_scales)) == 1 and len(set(col_scales)) == 1:
            rs, cs = row_scales[0], col_scales[0]
            if rs.denominator == 1 and cs.denominator == 1:
                for test_inp in task.test_inputs:
                    r = int(len(test_inp) * rs)
                    c = int(len(test_inp[0]) * cs)
                    return (r, c)  # 첫 번째 test input 기준

        return None

    # ── 단일 Grid 폴백 ────────────────────────────────────────────────────

    @staticmethod
    def fill_dominant_output_color(task: Task, size: tuple[int, int]) -> Grid:
        """학습 출력들에서 가장 많이 등장한 색상으로 채운 Grid 를 반환한다."""
        counts: Counter = Counter()
        for pair in task.train:
            if pair.output is not None:
                for row in pair.output:
                    counts.update(row)
        dominant = counts.most_common(1)[0][0] if counts else 0
        r, c = size
        return [[dominant] * c for _ in range(r)]

    @staticmethod
    def copy_input(inp: Grid) -> Grid:
        """입력 그리드를 그대로 복사하여 반환한다."""
        return copy.deepcopy(inp)

    @staticmethod
    def fill_most_common_input_color(inp: Grid, size: tuple[int, int]) -> Grid:
        """입력에서 가장 많이 등장한 색상으로 size 크기의 Grid 를 채운다."""
        counts: Counter = Counter(c for row in inp for c in row)
        dominant = counts.most_common(1)[0][0] if counts else 0
        r, c = size
        return [[dominant] * c for _ in range(r)]

    # ── test input 별 최선 폴백 ───────────────────────────────────────────

    @classmethod
    def best_guess(cls, task: Task, test_inp: Grid) -> Grid:
        """단일 test input 에 대한 최선 휴리스틱 Grid 를 반환한다.

        우선순위:
        1. 예측된 출력 크기 + 학습 출력 지배 색상
        2. 입력과 같은 크기 + 학습 출력 지배 색상
        3. 입력 그대로 복사
        """
        size = cls.predict_output_size(task)
        if size is None:
            size = (len(test_inp), len(test_inp[0]) if test_inp else 1)

        # 학습 출력 지배 색상으로 채우기
        counts: Counter = Counter()
        for pair in task.train:
            if pair.output is not None:
                for row in pair.output:
                    counts.update(row)
        if counts:
            return cls.fill_dominant_output_color(task, size)

        return cls.copy_input(test_inp)

    @classmethod
    def alternative_guess(cls, task: Task, test_inp: Grid) -> Grid:
        """best_guess 와 다른 다양한 폴백을 반환한다.

        best_guess 가 '지배 색상으로 채우기' 이므로,
        대안으로 '입력 그대로 복사' 를 사용한다.
        """
        return cls.copy_input(test_inp)


# ---------------------------------------------------------------------------
# SolvedCode
# ---------------------------------------------------------------------------

@dataclass
class SolvedCode:
    """검증 통과 또는 최선 부분 통과 코드 하나를 감싼다."""
    code: str
    sandbox_result: SandboxResult
    test_outputs: list[Grid | None]   # test_inputs 에 적용한 결과

    @property
    def passed(self) -> int:
        return self.sandbox_result.passed

    @property
    def total(self) -> int:
        return self.sandbox_result.total

    @property
    def is_passing(self) -> bool:
        return self.sandbox_result.success

    @property
    def conciseness(self) -> int:
        """실제 코드 줄 수 (주석·공백 제외). 낮을수록 간결."""
        return sum(
            1 for line in self.code.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    def outputs_differ_from(self, other: "SolvedCode") -> bool:
        """test 출력이 other 와 하나라도 다른지 확인한다."""
        for a, b in zip(self.test_outputs, other.test_outputs):
            if a != b:
                return True
        return False


# ---------------------------------------------------------------------------
# EnsembleResult
# ---------------------------------------------------------------------------

@dataclass
class EnsembleResult:
    """단일 Task 앙상블 결과."""
    task_id: TaskId
    predictions: list[Prediction]       # 최종 (attempt_1, attempt_2)
    passing_codes: list[SolvedCode]     # 검증 통과 코드 목록
    best_partial: SolvedCode | None     # 통과 코드 없을 때 최선 부분 코드
    strategy: Strategy
    total_llm_calls: int = 0

    def __str__(self) -> str:
        return (
            f"[{self.task_id}] strategy={self.strategy}  "
            f"passing={len(self.passing_codes)}  "
            f"llm_calls={self.total_llm_calls}"
        )


# ---------------------------------------------------------------------------
# EnsemblePredictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """passing_codes + fallback 을 조합하여 최종 Prediction 목록을 만든다."""

    def __init__(self, sandbox: CodeSandbox) -> None:
        self.sandbox   = sandbox
        self.heuristic = HeuristicSolver()

    def _safe_output(self, raw: Grid | None, fallback: Grid) -> Grid:
        return raw if raw is not None else fallback

    def _apply_code(self, code: str, task: Task) -> list[Grid | None]:
        return self.sandbox.apply(code, task.test_inputs)

    def predict(
        self,
        task: Task,
        passing_codes: list[SolvedCode],
        best_partial: SolvedCode | None,
    ) -> tuple[list[Prediction], Strategy]:
        """최종 Prediction 목록과 사용된 전략 레이블을 반환한다."""

        n_test = len(task.test_inputs)

        # ── 전략 1: 서로 다른 통과 코드 2개 ─────────────────────────────
        if len(passing_codes) >= 2:
            # 간결한 순으로 정렬; 동점은 출력 다양성 최대화
            sorted_codes = sorted(passing_codes, key=lambda s: s.conciseness)
            primary   = sorted_codes[0]
            secondary = sorted_codes[1]

            predictions = [
                Prediction(
                    attempt_1=self._safe_output(
                        primary.test_outputs[i],
                        self.heuristic.best_guess(task, task.test_inputs[i]),
                    ),
                    attempt_2=self._safe_output(
                        secondary.test_outputs[i],
                        self.heuristic.best_guess(task, task.test_inputs[i]),
                    ),
                )
                for i in range(n_test)
            ]
            return predictions, "diverse_2"

        # ── 전략 2: 통과 코드 1개 + 폴백 ────────────────────────────────
        if len(passing_codes) == 1:
            primary = passing_codes[0]
            predictions = [
                Prediction(
                    attempt_1=self._safe_output(
                        primary.test_outputs[i],
                        self.heuristic.best_guess(task, task.test_inputs[i]),
                    ),
                    attempt_2=self.heuristic.alternative_guess(task, task.test_inputs[i]),
                )
                for i in range(n_test)
            ]
            return predictions, "single_fallback"

        # ── 전략 3: 최선 부분 통과 코드 + 휴리스틱 ──────────────────────
        if best_partial is not None and best_partial.passed > 0:
            predictions = [
                Prediction(
                    attempt_1=self._safe_output(
                        best_partial.test_outputs[i],
                        self.heuristic.best_guess(task, task.test_inputs[i]),
                    ),
                    attempt_2=self.heuristic.best_guess(task, task.test_inputs[i]),
                )
                for i in range(n_test)
            ]
            return predictions, "partial_heuristic"

        # ── 전략 4: 전부 휴리스틱 ────────────────────────────────────────
        predictions = [
            Prediction(
                attempt_1=self.heuristic.best_guess(task, task.test_inputs[i]),
                attempt_2=self.heuristic.alternative_guess(task, task.test_inputs[i]),
            )
            for i in range(n_test)
        ]
        return predictions, "full_heuristic"


# ---------------------------------------------------------------------------
# DiverseSynthesizer
# ---------------------------------------------------------------------------

class DiverseSynthesizer:
    """최대 2개의 서로 다른 통과 코드를 수집하는 앙상블 합성기.

    작동 순서
    ---------
    1. 기본 합성 (temperature=0, 최대 max_attempts 회)
    2. 다양성 탐색 (temperature↑, 최대 diversity_attempts 회)
       - 이미 통과 코드가 있으면 "다른 방법으로" 프롬프트 변경
       - 없으면 동일 프롬프트 + 높은 온도로 재시도
    3. EnsemblePredictor 로 최종 Prediction 조립
    """

    _DIVERSITY_TEMPS = [0.7, 0.9]   # 다양성 탐색 시 사용하는 온도 목록

    def __init__(
        self,
        client: LLMClient,
        max_attempts: int = 5,
        diversity_attempts: int = 2,
        sandbox_timeout: float = 10.0,
        verbose: bool = True,
    ) -> None:
        self.client             = client
        self.diversity_attempts = diversity_attempts
        self.verbose            = verbose
        self._sandbox           = CodeSandbox(sandbox_timeout)
        self._predictor         = EnsemblePredictor(self._sandbox)
        self._base              = ProgramSynthesizer(
            client,
            max_attempts=max_attempts,
            sandbox_timeout=sandbox_timeout,
            verbose=verbose,
        )

    # ── 내부 유틸 ─────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _diversity_user_prompt(task: Task, passing_codes: list[SolvedCode]) -> str:
        """이미 통과한 코드를 제시하고 다른 접근법을 요청하는 프롬프트."""
        existing = "\n\n".join(
            f"**Existing solution {i+1} (DO NOT reuse this approach):**\n"
            f"```python\n{sc.code}\n```"
            for i, sc in enumerate(passing_codes)
        )
        return (
            f"The following solution(s) already pass all training examples "
            f"for task `{task.task_id}`.\n\n"
            f"{existing}\n\n"
            "Please provide a **completely different algorithmic approach** "
            "that also solves the task correctly. "
            "Your solution MUST use a different strategy/method than shown above. "
            "Output ONLY the Python code block with the `transform(grid)` function."
        )

    def _wrap_synthesis_result(
        self,
        result: SynthesisResult,
        task: Task,
    ) -> SolvedCode | None:
        """SynthesisResult 를 SolvedCode 로 변환한다."""
        if not result.final_code:
            return None
        # sandbox 재실행으로 정확한 SandboxResult 획득
        sr = self._sandbox.run(result.final_code, task)
        test_outs = self._sandbox.apply(result.final_code, task.test_inputs)
        return SolvedCode(code=result.final_code, sandbox_result=sr, test_outputs=test_outs)

    def _try_diversity_pass(
        self,
        task: Task,
        passing_codes: list[SolvedCode],
        temperature: float,
    ) -> SolvedCode | None:
        """다양성 탐색 1회를 수행하고 SolvedCode 또는 None 을 반환한다."""
        if passing_codes:
            user_content = self._diversity_user_prompt(task, passing_codes)
        else:
            user_content = PromptBuilder.initial_prompt(task)

        messages = [
            LLMMessage(role="system", content=PromptBuilder.system_prompt()),
            LLMMessage(role="user",   content=user_content),
        ]

        try:
            raw = self.client.chat(messages, temperature=temperature)
        except Exception as e:
            self._log(f"    LLM 오류: {e}")
            return None

        code = LLMClient._extract_code(raw)
        sr   = self._sandbox.run(code, task)
        test_outs = self._sandbox.apply(code, task.test_inputs)
        return SolvedCode(code=code, sandbox_result=sr, test_outputs=test_outs)

    def _is_duplicate(
        self, candidate: SolvedCode, existing: list[SolvedCode]
    ) -> bool:
        """코드 및 test 출력이 기존 것과 동일한지 확인한다."""
        for sc in existing:
            same_code    = candidate.code.strip() == sc.code.strip()
            same_outputs = not candidate.outputs_differ_from(sc)
            if same_code or same_outputs:
                return True
        return False

    # ── 메인 solve ────────────────────────────────────────────────────────

    def solve(self, task: Task) -> EnsembleResult:
        """Task 를 앙상블 전략으로 해결하고 EnsembleResult 를 반환한다."""
        self._log(f"\n{'='*64}")
        self._log(f"[Ensemble] task={task.task_id}  "
                  f"train={len(task.train)}  test={len(task.test_inputs)}")
        self._log(f"{'='*64}")

        passing_codes: list[SolvedCode] = []
        best_partial:  SolvedCode | None = None
        llm_calls = 0

        # ── Phase 1: 기본 합성 ───────────────────────────────────────────
        self._log("[Phase 1] 기본 합성 시작...")
        base_result = self._base.solve(task)
        llm_calls  += base_result.attempts

        base_sc = self._wrap_synthesis_result(base_result, task)
        if base_sc is not None:
            if base_sc.is_passing:
                passing_codes.append(base_sc)
                self._log(f"  ✓ 통과 코드 1개 획득 (conciseness={base_sc.conciseness})")
            elif best_partial is None or base_sc.passed > best_partial.passed:
                best_partial = base_sc
                self._log(f"  ~ 최선 부분 코드: {base_sc.passed}/{base_sc.total}")

        # ── Phase 2: 다양성 탐색 ────────────────────────────────────────
        self._log(f"[Phase 2] 다양성 탐색 (최대 {self.diversity_attempts}회)...")
        for i, temp in enumerate(self._DIVERSITY_TEMPS[: self.diversity_attempts]):
            if len(passing_codes) >= 2:
                break

            self._log(f"  [Diversity {i+1}] temperature={temp}")
            llm_calls += 1
            candidate = self._try_diversity_pass(task, passing_codes, temp)

            if candidate is None:
                continue

            if candidate.is_passing:
                if not self._is_duplicate(candidate, passing_codes):
                    passing_codes.append(candidate)
                    self._log(
                        f"  ✓ 다양성 통과 코드 획득 "
                        f"({len(passing_codes)}번째, conciseness={candidate.conciseness})"
                    )
                else:
                    self._log("  ~ 중복 코드/출력 — 스킵")
            else:
                if best_partial is None or candidate.passed > best_partial.passed:
                    best_partial = candidate
                    self._log(f"  ~ 부분 통과 업데이트: {candidate.passed}/{candidate.total}")

        # ── Phase 3: 앙상블 조립 ────────────────────────────────────────
        self._log(f"[Phase 3] 앙상블 조립  passing={len(passing_codes)}")
        predictions, strategy = self._predictor.predict(task, passing_codes, best_partial)

        self._log(f"  전략={strategy}  attempt_1≠attempt_2="
                  f"{any(p.attempt_1 != p.attempt_2 for p in predictions)}")

        return EnsembleResult(
            task_id=task.task_id,
            predictions=predictions,
            passing_codes=passing_codes,
            best_partial=best_partial,
            strategy=strategy,
            total_llm_calls=llm_calls,
        )

    def solve_all(
        self,
        tasks: dict[TaskId, Task],
        max_tasks: int | None = None,
        progress_fn=None,          # 외부 tqdm 업데이트 콜백
    ) -> dict[TaskId, EnsembleResult]:
        """전체 Task 를 순서대로 해결한다.

        Parameters
        ----------
        progress_fn : callable or None
            각 task 완료 시 호출되는 콜백(예: tqdm.update).
        """
        results: dict[TaskId, EnsembleResult] = {}
        items = list(tasks.items())
        if max_tasks is not None:
            items = items[:max_tasks]

        for task_id, task in items:
            results[task_id] = self.solve(task)
            if progress_fn is not None:
                progress_fn(1)

        solved = sum(1 for r in results.values() if r.passing_codes)
        diverse = sum(1 for r in results.values() if r.strategy == "diverse_2")
        self._log(f"\n{'='*64}")
        self._log(
            f"완료: {solved}/{len(results)} tasks 해결  "
            f"(diverse_2={diverse})"
        )
        self._log(f"{'='*64}")
        return results


# ---------------------------------------------------------------------------
# 통합 헬퍼
# ---------------------------------------------------------------------------

def ensemble_results_to_predictions(
    results: dict[TaskId, EnsembleResult],
) -> dict[TaskId, list[Prediction]]:
    """EnsembleResult dict → ARCSubmissionWriter 입력 형태로 변환."""
    return {task_id: res.predictions for task_id, res in results.items()}
