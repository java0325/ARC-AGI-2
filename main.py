#!/usr/bin/env python3
"""
ARC-AGI-2 메인 파이프라인 (main.py)
=====================================
Phase 1~3 모듈을 연결하여 최종 submission.json 을 생성한다.

사용법
------
# OpenAI 로 evaluation 셋 전체 실행
python main.py --split evaluation --provider openai --model o4-mini

# Anthropic 으로 evaluation 셋 처음 10개만 테스트
python main.py --split evaluation --provider anthropic --model claude-opus-4-5 --max-tasks 10

# 환경변수로 API 키 지정
OPENAI_API_KEY=sk-... python main.py --split test

# 출력 파일 지정
python main.py --split test --output my_submission.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── 의존성 지연 import (main() 호출 시 체크) ──────────────────────────────
def _require_tqdm():
    try:
        from tqdm import tqdm  # type: ignore  # noqa: F401
        return tqdm
    except ImportError:
        print("[오류] tqdm 이 설치되지 않았습니다: pip install tqdm")
        sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from arc_pipeline import ARCDataLoader, ARCEvaluator, ARCSubmissionWriter, Task, TaskId
from arc_synthesis import AnthropicClient, LLMClient, OpenAIClient
from arc_ensemble import (
    DiverseSynthesizer,
    EnsembleResult,
    ensemble_results_to_predictions,
)


# ---------------------------------------------------------------------------
# CLI 인수 파싱
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ARC-AGI-2 앙상블 합성 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 데이터 설정
    p.add_argument(
        "--split",
        choices=["training", "evaluation", "test"],
        default="evaluation",
        help="실행할 데이터 분할",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="데이터 디렉토리 경로 (기본: ARCDataLoader 내부 경로)",
    )

    # LLM 설정
    p.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM 제공자",
    )
    p.add_argument(
        "--model",
        default=None,
        help="모델명 (미지정 시 provider 기본값 사용)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API 키 (미지정 시 환경변수 OPENAI_API_KEY / ANTHROPIC_API_KEY 사용)",
    )

    # 합성 설정
    p.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="task 당 LLM 최대 시도 횟수 (기본 합성)",
    )
    p.add_argument(
        "--diversity-attempts",
        type=int,
        default=2,
        help="다양성 탐색 추가 시도 횟수",
    )
    p.add_argument(
        "--sandbox-timeout",
        type=float,
        default=10.0,
        help="샌드박스 실행 제한 시간(초)",
    )

    # 실행 제어
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="처리할 최대 task 수 (테스트용)",
    )
    p.add_argument(
        "--output",
        default="submission.json",
        help="저장할 submission.json 경로",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="각 task 의 상세 로그를 출력",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="training/evaluation 분할에서 평가 점수 계산을 건너뜀",
    )

    return p


# ---------------------------------------------------------------------------
# LLM 클라이언트 팩토리
# ---------------------------------------------------------------------------

def build_client(args: argparse.Namespace) -> LLMClient:
    if args.provider == "openai":
        model = args.model or "o4-mini"
        print(f"[설정] LLM: OpenAI / {model}")
        return OpenAIClient(api_key=args.api_key, model=model)
    else:
        model = args.model or "claude-opus-4-5"
        print(f"[설정] LLM: Anthropic / {model}")
        return AnthropicClient(api_key=args.api_key, model=model)


# ---------------------------------------------------------------------------
# 진행 상황 요약 출력
# ---------------------------------------------------------------------------

def _strategy_emoji(strategy: str) -> str:
    return {
        "diverse_2":       "★★",
        "single_fallback": "★·",
        "partial_heuristic": "~·",
        "full_heuristic":  "··",
    }.get(strategy, "??")


def print_summary(
    results: dict[TaskId, EnsembleResult],
    elapsed_sec: float,
    eval_scores: dict | None = None,
) -> None:
    total = len(results)
    strategy_counts: dict[str, int] = {}
    for r in results.values():
        strategy_counts[r.strategy] = strategy_counts.get(r.strategy, 0) + 1

    diverse2       = strategy_counts.get("diverse_2", 0)
    single_fb      = strategy_counts.get("single_fallback", 0)
    partial_heur   = strategy_counts.get("partial_heuristic", 0)
    full_heur      = strategy_counts.get("full_heuristic", 0)
    solved         = diverse2 + single_fb
    total_calls    = sum(r.total_llm_calls for r in results.values())
    avg_calls      = total_calls / total if total else 0

    print("\n" + "=" * 68)
    print(" ARC-AGI-2 실행 요약")
    print("=" * 68)
    print(f"  총 tasks          : {total}")
    print(f"  처리 시간          : {elapsed_sec:.1f}s  ({elapsed_sec/total:.1f}s/task)")
    print(f"  총 LLM 호출        : {total_calls}  (평균 {avg_calls:.1f}/task)")
    print()
    print(f"  ★★ diverse_2       : {diverse2:4d}  ({diverse2/total*100:.1f}%)  — 두 통과 코드")
    print(f"  ★· single_fallback : {single_fb:4d}  ({single_fb/total*100:.1f}%)  — 1코드 + 폴백")
    print(f"  ~· partial_heuristic:{partial_heur:4d}  ({partial_heur/total*100:.1f}%)  — 부분코드 + 휴리스틱")
    print(f"  ·· full_heuristic  : {full_heur:4d}  ({full_heur/total*100:.1f}%)  — 전부 휴리스틱")
    print(f"\n  LLM 통과 코드 있음 : {solved:4d}  ({solved/total*100:.1f}%)")

    if eval_scores:
        print()
        print(f"  평가 점수 (overall): {eval_scores['overall_score']:.4f}")
        perfect = sum(1 for s in eval_scores["task_scores"].values() if s == 1.0)
        print(f"  완벽 task          : {perfect}/{total}")

    print("=" * 68)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    print(f"[데이터] split={args.split} 로드 중...")
    loader = ARCDataLoader(data_dir=args.data_dir)

    if args.split == "training":
        tasks = loader.load_training()
    elif args.split == "evaluation":
        tasks = loader.load_evaluation()
    else:
        tasks = loader.load_test()

    if args.max_tasks is not None:
        task_items = list(tasks.items())[: args.max_tasks]
        tasks = dict(task_items)

    print(f"[데이터] {len(tasks)}개 task 로드 완료")

    # ── LLM 클라이언트 ────────────────────────────────────────────────────
    client = build_client(args)

    # ── DiverseSynthesizer ────────────────────────────────────────────────
    synthesizer = DiverseSynthesizer(
        client=client,
        max_attempts=args.max_attempts,
        diversity_attempts=args.diversity_attempts,
        sandbox_timeout=args.sandbox_timeout,
        verbose=args.verbose,
    )

    # ── tqdm 진행 바 ──────────────────────────────────────────────────────
    tqdm = _require_tqdm()
    results: dict[TaskId, EnsembleResult] = {}
    start_time = time.perf_counter()

    bar_fmt = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]"
    )

    with tqdm(
        total=len(tasks),
        desc="Solving tasks",
        unit="task",
        bar_format=bar_fmt,
        dynamic_ncols=True,
    ) as pbar:

        def on_task_done(n: int = 1) -> None:
            pbar.update(n)
            solved = sum(1 for r in results.values() if r.passing_codes)
            pbar.set_postfix(
                solved=solved,
                strategy=results[list(results)[-1]].strategy if results else "-",
                refresh=True,
            )

        for task_id, task in tasks.items():
            result = synthesizer.solve(task)
            results[task_id] = result
            on_task_done()

    elapsed = time.perf_counter() - start_time

    # ── 평가 (training / evaluation 분할) ────────────────────────────────
    eval_scores = None
    if args.split in ("training", "evaluation") and not args.no_eval:
        predictions = ensemble_results_to_predictions(results)
        evaluator = ARCEvaluator()
        eval_scores = evaluator.evaluate(predictions, tasks)

    # ── submission.json 저장 ─────────────────────────────────────────────
    predictions = ensemble_results_to_predictions(results)
    writer = ARCSubmissionWriter()
    saved_path = writer.save(predictions, args.output)

    # ── 요약 출력 ─────────────────────────────────────────────────────────
    print_summary(results, elapsed, eval_scores)
    print(f"\n[완료] 제출 파일 저장: {saved_path}")


if __name__ == "__main__":
    main()
