"""
ARC-AGI-2 Competition Pipeline
================================
데이터 로드 / 평가 메트릭 / 제출 파일 생성을 담당하는 기본 파이프라인.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Grid = list[list[int]]
TaskId = str


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    """학습/테스트 쌍 하나를 나타냄."""
    input: Grid
    output: Grid | None = None  # test input 에는 output 이 없음


@dataclass
class Task:
    """ARC task 하나를 나타냄."""
    task_id: TaskId
    train: list[Pair]
    test_inputs: list[Grid]
    test_outputs: list[Grid] = field(default_factory=list)  # 정답 (평가 시 사용)


@dataclass
class Prediction:
    """단일 test input 에 대한 2-attempt 예측."""
    attempt_1: Grid
    attempt_2: Grid


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

class ARCDataLoader:
    """ARC-AGI 대회 데이터를 로드한다."""

    DATA_DIR = Path("/Users/cwpark/kaggle/ARC-AGI-2/data/arc-prize-2026-arc-agi-2")

    # 파일명 매핑 (실제 파일명에 맞게 수정)
    FILES = {
        "training_challenges":   "arc-agi_training_challenges.json",
        "training_solutions":    "arc-agi_training_solutions.json",
        "evaluation_challenges": "arc-agi_evaluation_challenges.json",
        "evaluation_solutions":  "arc-agi_evaluation_solutions.json",
        "test_challenges":       "arc-agi_test_challenges.json",
    }

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_json(self, filename: str) -> dict:
        path = self.data_dir / filename
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _build_tasks(
        self,
        challenges: dict[TaskId, dict],
        solutions: dict[TaskId, list[Grid]] | None = None,
    ) -> dict[TaskId, Task]:
        tasks: dict[TaskId, Task] = {}

        for task_id, data in challenges.items():
            train_pairs = [
                Pair(input=p["input"], output=p["output"])
                for p in data["train"]
            ]
            test_inputs = [p["input"] for p in data["test"]]
            test_outputs: list[Grid] = solutions[task_id] if solutions else []

            tasks[task_id] = Task(
                task_id=task_id,
                train=train_pairs,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
            )

        return tasks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_training(self) -> dict[TaskId, Task]:
        """학습 데이터(challenges + solutions)를 로드한다."""
        challenges = self._load_json(self.FILES["training_challenges"])
        solutions  = self._load_json(self.FILES["training_solutions"])
        return self._build_tasks(challenges, solutions)

    def load_evaluation(self) -> dict[TaskId, Task]:
        """평가 데이터(challenges + solutions)를 로드한다."""
        challenges = self._load_json(self.FILES["evaluation_challenges"])
        solutions  = self._load_json(self.FILES["evaluation_solutions"])
        return self._build_tasks(challenges, solutions)

    def load_test(self) -> dict[TaskId, Task]:
        """최종 테스트 데이터(solutions 없음)를 로드한다."""
        challenges = self._load_json(self.FILES["test_challenges"])
        return self._build_tasks(challenges, solutions=None)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ARCEvaluator:
    """대회 규정에 따른 평가 메트릭을 계산한다.

    채점 규칙
    ---------
    - 각 test output 마다 2번의 시도(attempt_1, attempt_2)가 허용됨.
    - 둘 중 하나라도 정답 Grid 와 **크기 및 모든 셀 값**이 완전히 일치하면 1점, 아니면 0점.
    - 전체 점수 = (정답인 test output 수) / (전체 test output 수).
    """

    @staticmethod
    def grids_equal(a: Grid, b: Grid) -> bool:
        """두 Grid 가 완전히 동일한지 비교한다."""
        if len(a) != len(b):
            return False
        return all(row_a == row_b for row_a, row_b in zip(a, b))

    def score_prediction(
        self,
        prediction: Prediction,
        ground_truth: Grid,
    ) -> int:
        """단일 test output 에 대한 점수(0 또는 1)를 반환한다."""
        if self.grids_equal(prediction.attempt_1, ground_truth):
            return 1
        if self.grids_equal(prediction.attempt_2, ground_truth):
            return 1
        return 0

    def score_task(
        self,
        predictions: list[Prediction],
        ground_truths: list[Grid],
    ) -> float:
        """단일 task 의 평균 점수를 반환한다."""
        if not ground_truths:
            raise ValueError("ground_truths 가 비어 있습니다.")
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"predictions({len(predictions)})와 "
                f"ground_truths({len(ground_truths)}) 길이가 다릅니다."
            )
        total = sum(
            self.score_prediction(pred, gt)
            for pred, gt in zip(predictions, ground_truths)
        )
        return total / len(ground_truths)

    def evaluate(
        self,
        all_predictions: dict[TaskId, list[Prediction]],
        tasks: dict[TaskId, Task],
    ) -> dict[str, float]:
        """모든 task 에 대한 평가 결과를 반환한다.

        Returns
        -------
        dict with keys:
            "task_scores"   : {task_id: score}
            "overall_score" : 전체 test output 에 대한 평균 점수
        """
        task_scores: dict[TaskId, float] = {}
        total_correct = 0
        total_outputs = 0

        for task_id, task in tasks.items():
            if not task.test_outputs:
                continue
            preds = all_predictions.get(task_id, [])
            if not preds:
                # 예측 없음 → 해당 task 의 모든 output 0점
                task_scores[task_id] = 0.0
                total_outputs += len(task.test_outputs)
                continue

            score = self.score_task(preds, task.test_outputs)
            task_scores[task_id] = score

            correct = sum(
                self.score_prediction(pred, gt)
                for pred, gt in zip(preds, task.test_outputs)
            )
            total_correct += correct
            total_outputs += len(task.test_outputs)

        overall = total_correct / total_outputs if total_outputs > 0 else 0.0
        return {"task_scores": task_scores, "overall_score": overall}


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------

class ARCSubmissionWriter:
    """대회 규정에 맞는 submission.json 을 생성한다.

    포맷
    ----
    {
        "<task_id>": [
            {"attempt_1": <Grid>, "attempt_2": <Grid>},  // test input 0
            {"attempt_1": <Grid>, "attempt_2": <Grid>},  // test input 1
            ...
        ],
        ...
    }
    """

    def build_submission(
        self,
        all_predictions: dict[TaskId, list[Prediction]],
    ) -> dict[TaskId, list[dict[str, Grid]]]:
        """예측 결과를 submission dict 로 변환한다."""
        submission: dict[TaskId, list[dict[str, Grid]]] = {}

        for task_id, predictions in all_predictions.items():
            submission[task_id] = [
                {"attempt_1": pred.attempt_1, "attempt_2": pred.attempt_2}
                for pred in predictions
            ]

        return submission

    def save(
        self,
        all_predictions: dict[TaskId, list[Prediction]],
        output_path: Path | str = "submission.json",
    ) -> Path:
        """submission.json 파일을 저장하고 경로를 반환한다."""
        path = Path(output_path)
        submission = self.build_submission(all_predictions)
        with path.open("w", encoding="utf-8") as f:
            json.dump(submission, f)
        print(f"[SubmissionWriter] 저장 완료: {path.resolve()}  ({len(submission)} tasks)")
        return path


# ---------------------------------------------------------------------------
# Pipeline (통합 진입점)
# ---------------------------------------------------------------------------

SolverFn = Callable[[Task], list[Prediction]]


class ARCPipeline:
    """ARC-AGI 파이프라인의 통합 진입점.

    사용법
    ------
    def my_solver(task: Task) -> list[Prediction]:
        # task.test_inputs 의 각 Grid 에 대해 Prediction 을 반환
        return [
            Prediction(attempt_1=[[0]], attempt_2=[[0]])
            for _ in task.test_inputs
        ]

    pipeline = ARCPipeline()
    pipeline.run(split="evaluation", solver=my_solver)
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.loader    = ARCDataLoader(data_dir)
        self.evaluator = ARCEvaluator()
        self.writer    = ARCSubmissionWriter()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_solver(
        self,
        tasks: dict[TaskId, Task],
        solver: SolverFn,
    ) -> dict[TaskId, list[Prediction]]:
        all_predictions: dict[TaskId, list[Prediction]] = {}
        for task_id, task in tasks.items():
            preds = solver(task)
            assert len(preds) == len(task.test_inputs), (
                f"[{task_id}] solver 는 test_inputs({len(task.test_inputs)})와 "
                f"동일한 수의 Prediction({len(preds)})을 반환해야 합니다."
            )
            all_predictions[task_id] = preds
        return all_predictions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        solver: SolverFn,
        split: str = "test",
        submission_path: Path | str = "submission.json",
    ) -> dict[str, object]:
        """파이프라인 전체를 실행한다.

        Parameters
        ----------
        solver:
            Task → list[Prediction] 을 반환하는 함수.
        split:
            "training" | "evaluation" | "test"
        submission_path:
            저장할 submission.json 경로.

        Returns
        -------
        "predictions"    : {task_id: [Prediction, ...]}
        "submission_path": 저장된 파일 경로
        "scores"         : 평가 가능한 split 일 때 점수 dict, 없으면 None
        """
        print(f"[Pipeline] split={split} 로드 중 ...")
        if split == "training":
            tasks = self.loader.load_training()
        elif split == "evaluation":
            tasks = self.loader.load_evaluation()
        elif split == "test":
            tasks = self.loader.load_test()
        else:
            raise ValueError(f"알 수 없는 split: {split!r}")

        print(f"[Pipeline] {len(tasks)}개 task 에 대해 solver 실행 중 ...")
        all_predictions = self._run_solver(tasks, solver)

        saved_path = self.writer.save(all_predictions, submission_path)

        scores = None
        if split in ("training", "evaluation"):
            scores = self.evaluator.evaluate(all_predictions, tasks)
            overall = scores["overall_score"]
            print(f"[Pipeline] Overall score ({split}): {overall:.4f}")

        return {
            "predictions":     all_predictions,
            "submission_path": saved_path,
            "scores":          scores,
        }


# ---------------------------------------------------------------------------
# Example usage (더미 solver)
# ---------------------------------------------------------------------------

def dummy_solver(task: Task) -> list[Prediction]:
    """모든 셀을 0 으로 채우는 더미 solver."""
    predictions: list[Prediction] = []
    for grid in task.test_inputs:
        rows = len(grid)
        cols = len(grid[0]) if grid else 1
        blank: Grid = [[0] * cols for _ in range(rows)]
        predictions.append(Prediction(attempt_1=blank, attempt_2=blank))
    return predictions


if __name__ == "__main__":
    pipeline = ARCPipeline()
    result = pipeline.run(
        solver=dummy_solver,
        split="evaluation",
        submission_path="submission.json",
    )
    scores = result["scores"]
    if scores:
        print(f"\n전체 점수: {scores['overall_score']:.4f}")
        task_scores: dict = scores["task_scores"]
        perfect = sum(1 for s in task_scores.values() if s == 1.0)
        print(f"완벽히 맞춘 task: {perfect} / {len(task_scores)}")
