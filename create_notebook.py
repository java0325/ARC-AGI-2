#!/usr/bin/env python3
"""arc_agi2_pipeline.ipynb 생성 스크립트 (v2)."""

import json, uuid
from pathlib import Path

# ── arc_dsl.py 내용을 repr() 로 안전하게 임베드 ───────────────────────────────
_ARC_DSL_CONTENT = Path("arc_dsl.py").read_text(encoding="utf-8")
_ARC_DSL_REPR    = repr(_ARC_DSL_CONTENT)   # 안전한 Python 문자열 리터럴

# ── 노트북 헬퍼 ────────────────────────────────────────────────────────────────
def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "id": uuid.uuid4().hex[:8], "metadata": {}, "outputs": [], "source": source}

def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": source}

# ══════════════════════════════════════════════════════════════════════════════
# Cell 소스
# ══════════════════════════════════════════════════════════════════════════════

MD_TITLE = """\
# ARC-AGI-2 Competition Pipeline
> 단일 Jupyter Notebook — 데이터 로드 / DSL / 합성 엔진 / 앙상블 전략을 모두 포함
>
> | Phase | 내용 |
> |---|---|
> | 1 | 데이터 파이프라인 (`ARCDataLoader`, `ARCEvaluator`, `ARCSubmissionWriter`) |
> | 2 | Grid DSL (`GeometricOps`, `ObjectOps`, `ColorOps`, `SizeAnalyzer`) |
> | 3 | 합성 엔진 (`LLMClient`, `CodeSandbox`, `ProgramSynthesizer`) |
> | 4 | 앙상블 전략 (`HeuristicSolver`, `DiverseSynthesizer`, `EnsemblePredictor`) |"""

CELL_INSTALL = """\
import subprocess, sys
for _p in ["tqdm", "openai", "anthropic"]:
    subprocess.run([sys.executable, "-m", "pip", "install", _p, "-q"], check=False)
print("✓ 패키지 설치 완료")"""

CELL_IMPORTS = """\
from __future__ import annotations
import abc, copy, json, os, re, subprocess, sys, tempfile, textwrap, time
from collections import Counter, deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Literal

NOTEBOOK_DIR   = Path(os.getcwd())
DSL_MODULE_DIR = str(NOTEBOOK_DIR)

# Kaggle 환경이면 대회 입력 경로를, 아니면 노트북과 같은 폴더를 사용
_KAGGLE_DATA = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
_LOCAL_DATA  = Path(".")
DATA_DIR     = _KAGGLE_DATA if _KAGGLE_DATA.exists() else _LOCAL_DATA

Grid         = list[list[int]]
TaskId       = str
Point        = tuple[int, int]
Connectivity = Literal[4, 8]

print(f"✓ 임포트 완료")
print(f"  NOTEBOOK_DIR = {NOTEBOOK_DIR}")
print(f"  DATA_DIR     = {DATA_DIR.resolve()}")
print(f"  환경         = {'Kaggle' if _KAGGLE_DATA.exists() else 'Local (fallback)'}")"""

# ── Phase 1: 데이터 파이프라인 ─────────────────────────────────────────────────
CELL_PHASE1 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 1: 데이터 파이프라인
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Pair:
    input: Grid
    output: Grid | None = None

@dataclass
class Task:
    task_id: TaskId
    train: list[Pair]
    test_inputs: list[Grid]
    test_outputs: list[Grid] = field(default_factory=list)

@dataclass
class Prediction:
    attempt_1: Grid
    attempt_2: Grid


class ARCDataLoader:
    FILES = {
        "training_challenges":   "arc-agi_training_challenges.json",
        "training_solutions":    "arc-agi_training_solutions.json",
        "evaluation_challenges": "arc-agi_evaluation_challenges.json",
        "evaluation_solutions":  "arc-agi_evaluation_solutions.json",
        "test_challenges":       "arc-agi_test_challenges.json",
    }

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def _load_json(self, filename: str) -> dict:
        with (self.data_dir / filename).open(encoding="utf-8") as f:
            return json.load(f)

    def _build_tasks(self, challenges: dict, solutions: dict | None = None) -> dict[TaskId, Task]:
        tasks: dict[TaskId, Task] = {}
        for tid, data in challenges.items():
            tasks[tid] = Task(
                task_id=tid,
                train=[Pair(input=p["input"], output=p["output"]) for p in data["train"]],
                test_inputs=[p["input"] for p in data["test"]],
                test_outputs=solutions[tid] if solutions else [],
            )
        return tasks

    def load_training(self)   -> dict[TaskId, Task]:
        return self._build_tasks(self._load_json(self.FILES["training_challenges"]),
                                  self._load_json(self.FILES["training_solutions"]))

    def load_evaluation(self) -> dict[TaskId, Task]:
        return self._build_tasks(self._load_json(self.FILES["evaluation_challenges"]),
                                  self._load_json(self.FILES["evaluation_solutions"]))

    def load_test(self)       -> dict[TaskId, Task]:
        return self._build_tasks(self._load_json(self.FILES["test_challenges"]))


class ARCEvaluator:
    @staticmethod
    def grids_equal(a: Grid, b: Grid) -> bool:
        return len(a) == len(b) and all(ra == rb for ra, rb in zip(a, b))

    def score_prediction(self, p: Prediction, gt: Grid) -> int:
        return 1 if (self.grids_equal(p.attempt_1, gt) or self.grids_equal(p.attempt_2, gt)) else 0

    def score_task(self, preds: list[Prediction], gts: list[Grid]) -> float:
        if not gts: raise ValueError("ground_truths 가 비어 있습니다.")
        if len(preds) != len(gts): raise ValueError(f"길이 불일치 {len(preds)} vs {len(gts)}")
        return sum(self.score_prediction(p, g) for p, g in zip(preds, gts)) / len(gts)

    def evaluate(self, all_preds: dict[TaskId, list[Prediction]],
                 tasks: dict[TaskId, Task]) -> dict[str, Any]:
        scores: dict[TaskId, float] = {}
        correct = total = 0
        for tid, task in tasks.items():
            if not task.test_outputs: continue
            preds = all_preds.get(tid, [])
            if not preds:
                scores[tid] = 0.0; total += len(task.test_outputs); continue
            scores[tid] = self.score_task(preds, task.test_outputs)
            correct += sum(self.score_prediction(p, g)
                           for p, g in zip(preds, task.test_outputs))
            total   += len(task.test_outputs)
        return {"task_scores": scores, "overall_score": correct / total if total else 0.0}


class ARCSubmissionWriter:
    def build_submission(self, all_preds: dict[TaskId, list[Prediction]]) -> dict:
        return {tid: [{"attempt_1": p.attempt_1, "attempt_2": p.attempt_2} for p in preds]
                for tid, preds in all_preds.items()}

    def save(self, all_preds: dict[TaskId, list[Prediction]],
             out: Path | str = "submission.json") -> Path:
        path = Path(out)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.build_submission(all_preds), f)
        print(f"[SubmissionWriter] 저장: {path.resolve()}  ({len(all_preds)} tasks)")
        return path


SolverFn = Callable[[Task], list[Prediction]]

def dummy_solver(task: Task) -> list[Prediction]:
    return [Prediction(attempt_1=[[0]*len(g[0]) for _ in g],
                       attempt_2=[[0]*len(g[0]) for _ in g])
            for g in task.test_inputs]

print("✓ Phase 1 로드 완료")
'''

# ── Phase 2: DSL (arc_dsl.py 를 디스크에 쓰고 import) ─────────────────────────
CELL_PHASE2 = f'''\
# ═══════════════════════════════════════════════════════════════════
# Phase 2: Grid DSL  (arc_dsl.py 디스크에 저장 → import)
# ═══════════════════════════════════════════════════════════════════

# arc_dsl.py 원본 내용을 임베드 — subprocess sandbox 가 이 파일을 import 함
_ARC_DSL_PY = {_ARC_DSL_REPR}

_dsl_path = NOTEBOOK_DIR / "arc_dsl.py"
_dsl_path.write_text(_ARC_DSL_PY, encoding="utf-8")

# 모듈 재로드 (이미 import 된 경우 대비)
import importlib, importlib.util
if "arc_dsl" in sys.modules:
    del sys.modules["arc_dsl"]
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from arc_dsl import (
    GeometricOps, ArcObject, ObjectOps, ColorOps,
    SizeRelationship, SizeAnalyzer, GridDSL,
)

print(f"✓ Phase 2 로드 완료  — arc_dsl.py 저장: {{_dsl_path}}")
print(f"  크기: {{_dsl_path.stat().st_size}} bytes")
'''

# ── Phase 3: 합성 엔진 ─────────────────────────────────────────────────────────
CELL_PHASE3 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 3: 프로그램 합성 엔진
# ═══════════════════════════════════════════════════════════════════

DSL_REFERENCE = """
## arc_dsl 레퍼런스 (import 가능)
```python
from arc_dsl import GeometricOps, ObjectOps, ColorOps, SizeAnalyzer, GridDSL
GeometricOps.rotate(grid, degrees)          # 0/90/180/270
GeometricOps.flip_horizontal/vertical(grid)
GeometricOps.scale_up(grid, factor)
GeometricOps.pad(grid, top,bottom,left,right,fill=0)
GeometricOps.crop(grid, r1,c1,r2,c2)
GeometricOps.trim(grid, background=0)
GeometricOps.overlay(base,patch,r,c,transparent=None)
ObjectOps.find_objects(grid, background=0, connectivity=4)  # list[ArcObject]
ObjectOps.isolate_shapes / label_grid / count_objects / bounding_box_grid
ColorOps.most_common_color / least_common_color / replace_color
ColorOps.flood_fill(grid,(r,c),new_color) / apply_palette(grid,{old:new})
```
""".strip()


@dataclass
class LLMMessage:
    role: str
    content: str


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def chat(self, messages: list[LLMMessage],
             temperature: float = 0.0, max_tokens: int = 4096) -> str: ...

    @staticmethod
    def _extract_code(text: str) -> str:
        matches = re.findall(r"```(?:python)?\\s*\\n(.*?)```", text, re.DOTALL)
        return max(matches, key=len).strip() if matches else text.strip()


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "o4-mini",
                 base_url: str | None = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("pip install openai") from e
        from openai import OpenAI  # type: ignore
        self._c = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self.model = model

    def chat(self, messages: list[LLMMessage],
             temperature: float = 0.0, max_tokens: int = 4096) -> str:
        r = self._c.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature, max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "claude-opus-4-5") -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError("pip install anthropic") from e
        import anthropic  # type: ignore
        self._c = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def chat(self, messages: list[LLMMessage],
             temperature: float = 0.0, max_tokens: int = 4096) -> str:
        sys_msgs  = [m for m in messages if m.role == "system"]
        user_msgs = [m for m in messages if m.role != "system"]
        r = self._c.messages.create(
            model=self.model,
            system="\\n\\n".join(m.content for m in sys_msgs) or "ARC expert.",
            messages=[{"role": m.role, "content": m.content} for m in user_msgs],
            temperature=temperature, max_tokens=max_tokens,
        )
        return r.content[0].text


class PromptBuilder:
    _CC = {0:".",1:"B",2:"R",3:"G",4:"Y",5:"W",6:"M",7:"O",8:"A",9:"P"}

    @classmethod
    def _g2t(cls, grid: Grid) -> str:
        return f"({len(grid)}x{len(grid[0]) if grid else 0})\\n" + "\\n".join(
            " ".join(cls._CC.get(c, str(c)) for c in row) for row in grid)

    @classmethod
    def _pair_block(cls, idx: int, inp: Grid, out: Grid | None) -> str:
        return (f"### Example {idx+1}\\n**Input:**\\n```\\n{cls._g2t(inp)}\\n```\\n"
                f"**Output:**\\n```\\n{cls._g2t(out)}\\n```" if out is not None else
                f"### Test Input {idx+1}\\n```\\n{cls._g2t(inp)}\\n```")

    @classmethod
    def system_prompt(cls) -> str:
        return textwrap.dedent("""
            You are an ARC puzzle expert. ARC grids are 2D lists of integers 0-9.
            RULES:
            1. Reply with ONLY one Python code block (```python ... ```).
            2. Define exactly: `transform(grid: list[list[int]]) -> list[list[int]]`.
            3. Do NOT call transform inside the block. No explanations outside the block.
            4. May import from `arc_dsl`; use only stdlib otherwise.
            Colors: .=0 B=1 R=2 G=3 Y=4 W=5 M=6 O=7 A=8 P=9
        """).strip()

    @classmethod
    def initial_prompt(cls, task: Task) -> str:
        train = "\\n\\n".join(cls._pair_block(i, p.input, p.output)
                               for i, p in enumerate(task.train))
        tests = "\\n\\n".join(cls._pair_block(i, inp, None)
                               for i, inp in enumerate(task.test_inputs))
        return f"## ARC Task ({task.task_id})\\n\\n{train}\\n\\n---\\n\\n{tests}\\n\\n---\\n\\n{DSL_REFERENCE}\\n\\n---\\n\\nWrite the `transform(grid)` function."

    @classmethod
    def refinement_prompt(cls, task: Task, prev_code: str, sr: "SandboxResult") -> str:
        diffs = [f"Example {i+1}: RUNTIME ERROR" if a is None else
                 f"Example {i+1}:\\n  Expected:\\n```\\n{cls._g2t(p.output)}\\n```\\n  Got:\\n```\\n{cls._g2t(a)}\\n```"
                 for i, (p, a) in enumerate(zip(task.train, sr.outputs)) if a != p.output]
        err = f"\\n**Error:**\\n```\\n{sr.error}\\n```\\n" if sr.error else ""
        return (f"INCORRECT: passed {sr.passed}/{sr.total}.{err}\\n## Failures\\n"
                f"{'\\n\\n'.join(diffs) or '(no diff)'}\\n\\n## Prev Code\\n```python\\n{prev_code}\\n```\\n"
                "Provide corrected `transform(grid)`. Code block ONLY.")


@dataclass
class SandboxResult:
    success: bool
    passed: int = 0
    total: int = 0
    outputs: list[Grid | None] = field(default_factory=list)
    error: str = ""
    exec_time_ms: float = 0.0


class CodeSandbox:
    def __init__(self, timeout_sec: float = 10.0) -> None:
        self.timeout_sec = timeout_sec

    @staticmethod
    def _runner_src(dsl_dir: str, code: str, pairs: list[list[Any]]) -> str:
        return (
            "import sys, json, traceback, copy\\n"
            f"sys.path.insert(0, {dsl_dir!r})\\n\\n"
            f"{code}\\n\\n"
            f"_pairs = {repr(pairs)}\\n"
            "_res = []\\n"
            "for _i, _e in _pairs:\\n"
            "    try:\\n"
            "        _o = transform(copy.deepcopy(_i))\\n"
            "        _res.append({'passed': _o == _e, 'output': _o, 'error': ''})\\n"
            "    except Exception:\\n"
            "        _res.append({'passed': False, 'output': None, 'error': traceback.format_exc()})\\n"
            "print(json.dumps(_res))\\n"
        )

    def _run(self, src: str) -> tuple[str, str, int]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(src); tmp = f.name
        try:
            p = subprocess.run([sys.executable, tmp], capture_output=True,
                               text=True, timeout=self.timeout_sec)
            return p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            return "", f"Timeout ({self.timeout_sec}s) exceeded", 1
        finally:
            try: Path(tmp).unlink()
            except OSError: pass

    def run(self, code: str, task: Task) -> SandboxResult:
        pairs = [[p.input, p.output] for p in task.train if p.output is not None]
        t0 = time.perf_counter()
        stdout, stderr, rc = self._run(self._runner_src(DSL_MODULE_DIR, code, pairs))
        ms = (time.perf_counter() - t0) * 1000
        if rc != 0:
            return SandboxResult(False, 0, len(pairs), [None]*len(pairs), stderr[:2000], ms)
        if not stdout.strip():
            return SandboxResult(False, 0, len(pairs), [None]*len(pairs), "No output", ms)
        try:
            results = json.loads(stdout.strip())
        except json.JSONDecodeError as e:
            return SandboxResult(False, 0, len(pairs), [None]*len(pairs), f"JSON: {e}", ms)
        passed = sum(1 for r in results if r["passed"])
        return SandboxResult(
            success=(passed == len(pairs)), passed=passed, total=len(pairs),
            outputs=[r["output"] for r in results],
            error="\\n".join(r["error"] for r in results if r["error"])[:2000],
            exec_time_ms=ms,
        )

    def apply(self, code: str, inputs: list[Grid]) -> list[Grid | None]:
        src = (
            f"import sys, json, traceback, copy\\n"
            f"sys.path.insert(0, {DSL_MODULE_DIR!r})\\n\\n"
            f"{code}\\n\\n"
            f"_ins = {repr(inputs)}\\n_outs = []\\n"
            "for _i in _ins:\\n"
            "    try: _outs.append(transform(copy.deepcopy(_i)))\\n"
            "    except Exception: _outs.append(None)\\n"
            "print(json.dumps(_outs))\\n"
        )
        stdout, _, rc = self._run(src)
        if rc != 0 or not stdout.strip(): return [None] * len(inputs)
        try: return json.loads(stdout.strip())
        except Exception: return [None] * len(inputs)


@dataclass
class SynthesisResult:
    task_id: TaskId
    success: bool
    predictions: list[Prediction]
    attempts: int = 0
    final_code: str = ""
    error: str = ""
    def __str__(self) -> str:
        return f"[{'OK' if self.success else 'FAIL'}] {self.task_id} attempts={self.attempts}"


class ProgramSynthesizer:
    def __init__(self, client: LLMClient, max_attempts: int = 5,
                 sandbox_timeout: float = 10.0, verbose: bool = True) -> None:
        self.client = client; self.max_attempts = max_attempts
        self.sandbox = CodeSandbox(sandbox_timeout); self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose: print(msg)

    def _fallback(self, task: Task) -> list[Prediction]:
        return [Prediction(attempt_1=[[0]*len(g[0]) for _ in g],
                           attempt_2=[[0]*len(g[0]) for _ in g])
                for g in task.test_inputs]

    def solve(self, task: Task) -> SynthesisResult:
        self._log(f"[Synthesizer] {task.task_id}")
        msgs = [LLMMessage("system", PromptBuilder.system_prompt()),
                LLMMessage("user",   PromptBuilder.initial_prompt(task))]
        last_code = ""; last_sr = None
        for attempt in range(1, self.max_attempts + 1):
            self._log(f"  Attempt {attempt}/{self.max_attempts}")
            try:
                raw = self.client.chat(msgs, temperature=0.0 if attempt == 1 else 0.3)
            except Exception as e:
                self._log(f"  LLM error: {e}"); break
            code = LLMClient._extract_code(raw); last_code = code
            sr = self.sandbox.run(code, task); last_sr = sr
            self._log(f"  {sr.passed}/{sr.total} passed  ({sr.exec_time_ms:.0f}ms)")
            if sr.error: self._log(f"  err: {sr.error[:120]}")
            if sr.success:
                self._log("  ✓ all train passed")
                raw_preds = self.sandbox.apply(code, task.test_inputs)
                preds = []
                for rp in raw_preds:
                    if rp is None:
                        r, c = len(task.test_inputs[0]), len(task.test_inputs[0][0])
                        rp = [[0]*c for _ in range(r)]
                    preds.append(Prediction(attempt_1=rp, attempt_2=rp))
                return SynthesisResult(task.task_id, True, preds, attempt, code)
            if attempt < self.max_attempts:
                msgs.extend([LLMMessage("assistant", raw),
                              LLMMessage("user", PromptBuilder.refinement_prompt(task, code, sr))])
        self._log(f"  ✗ failed after {self.max_attempts} attempts")
        return SynthesisResult(task.task_id, False, self._fallback(task),
                               self.max_attempts, last_code,
                               f"failed: {last_sr.passed if last_sr else 0}/{last_sr.total if last_sr else 0}")


def results_to_predictions(results: dict[TaskId, SynthesisResult]) -> dict[TaskId, list[Prediction]]:
    return {tid: r.predictions for tid, r in results.items()}


print("✓ Phase 3 로드 완료")
'''

# ── Phase 4: 앙상블 ────────────────────────────────────────────────────────────
CELL_PHASE4 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 4: 앙상블 전략
# ═══════════════════════════════════════════════════════════════════

Strategy = Literal["diverse_2", "single_fallback", "partial_heuristic", "full_heuristic"]


class HeuristicSolver:
    @staticmethod
    def predict_output_size(task: Task) -> tuple[int, int] | None:
        pairs = [(p.input, p.output) for p in task.train if p.output is not None]
        if not pairs: return None
        out_sz = [(len(o), len(o[0])) for _, o in pairs]
        if len(set(out_sz)) == 1: return out_sz[0]
        in_sz = [(len(i), len(i[0])) for i, _ in pairs]
        rs = [Fraction(o[0], i[0]) for i, o in zip(in_sz, out_sz)]
        cs = [Fraction(o[1], i[1]) for i, o in zip(in_sz, out_sz)]
        if len(set(rs)) == 1 and len(set(cs)) == 1 and rs[0].denominator == 1 and cs[0].denominator == 1:
            for inp in task.test_inputs:
                return (int(len(inp) * rs[0]), int(len(inp[0]) * cs[0]))
        return None

    @staticmethod
    def fill_dominant_output_color(task: Task, size: tuple[int, int]) -> Grid:
        cnt: Counter = Counter()
        for p in task.train:
            if p.output:
                for row in p.output: cnt.update(row)
        dom = cnt.most_common(1)[0][0] if cnt else 0
        r, c = size; return [[dom]*c for _ in range(r)]

    @staticmethod
    def copy_input(inp: Grid) -> Grid: return copy.deepcopy(inp)

    @classmethod
    def best_guess(cls, task: Task, inp: Grid) -> Grid:
        size = cls.predict_output_size(task) or (len(inp), len(inp[0]) if inp else 1)
        cnt: Counter = Counter()
        for p in task.train:
            if p.output:
                for row in p.output: cnt.update(row)
        return cls.fill_dominant_output_color(task, size) if cnt else cls.copy_input(inp)

    @classmethod
    def alternative_guess(cls, _: Task, inp: Grid) -> Grid:
        return cls.copy_input(inp)


@dataclass
class SolvedCode:
    code: str
    sandbox_result: SandboxResult
    test_outputs: list[Grid | None]

    @property
    def passed(self)     -> int:  return self.sandbox_result.passed
    @property
    def total(self)      -> int:  return self.sandbox_result.total
    @property
    def is_passing(self) -> bool: return self.sandbox_result.success
    @property
    def conciseness(self) -> int:
        return sum(1 for l in self.code.splitlines()
                   if l.strip() and not l.strip().startswith("#"))

    def outputs_differ_from(self, other: "SolvedCode") -> bool:
        return any(a != b for a, b in zip(self.test_outputs, other.test_outputs))


@dataclass
class EnsembleResult:
    task_id: TaskId
    predictions: list[Prediction]
    passing_codes: list[SolvedCode]
    best_partial: SolvedCode | None
    strategy: Strategy
    total_llm_calls: int = 0
    def __str__(self) -> str:
        return f"[{self.task_id}] {self.strategy} passing={len(self.passing_codes)}"


class EnsemblePredictor:
    def __init__(self, sandbox: CodeSandbox) -> None:
        self.sandbox = sandbox; self.h = HeuristicSolver()

    def _safe(self, raw: Grid | None, fb: Grid) -> Grid:
        return raw if raw is not None else fb

    def predict(self, task: Task, passing: list[SolvedCode],
                partial: SolvedCode | None) -> tuple[list[Prediction], Strategy]:
        n = len(task.test_inputs); h = self.h
        if len(passing) >= 2:
            sc = sorted(passing, key=lambda s: s.conciseness)
            return ([Prediction(self._safe(sc[0].test_outputs[i], h.best_guess(task, task.test_inputs[i])),
                                self._safe(sc[1].test_outputs[i], h.best_guess(task, task.test_inputs[i])))
                     for i in range(n)], "diverse_2")
        if len(passing) == 1:
            p = passing[0]
            return ([Prediction(self._safe(p.test_outputs[i], h.best_guess(task, task.test_inputs[i])),
                                h.alternative_guess(task, task.test_inputs[i]))
                     for i in range(n)], "single_fallback")
        if partial and partial.passed > 0:
            return ([Prediction(self._safe(partial.test_outputs[i], h.best_guess(task, task.test_inputs[i])),
                                h.best_guess(task, task.test_inputs[i]))
                     for i in range(n)], "partial_heuristic")
        return ([Prediction(h.best_guess(task, task.test_inputs[i]),
                            h.alternative_guess(task, task.test_inputs[i]))
                 for i in range(n)], "full_heuristic")


class DiverseSynthesizer:
    _TEMPS = [0.7, 0.9]

    def __init__(self, client: LLMClient, max_attempts: int = 5,
                 diversity_attempts: int = 2, sandbox_timeout: float = 10.0,
                 verbose: bool = True) -> None:
        self.client = client; self.da = diversity_attempts; self.verbose = verbose
        self._sb  = CodeSandbox(sandbox_timeout)
        self._ep  = EnsemblePredictor(self._sb)
        self._base = ProgramSynthesizer(client, max_attempts, sandbox_timeout, verbose)

    def _log(self, m: str) -> None:
        if self.verbose: print(m)

    def _wrap(self, res: SynthesisResult, task: Task) -> SolvedCode | None:
        if not res.final_code: return None
        sr  = self._sb.run(res.final_code, task)
        out = self._sb.apply(res.final_code, task.test_inputs)
        return SolvedCode(res.final_code, sr, out)

    def _div_pass(self, task: Task, passing: list[SolvedCode], temp: float) -> SolvedCode | None:
        if passing:
            existing = "\\n\\n".join(
                f"**Solution {i+1} (DO NOT reuse):**\\n```python\\n{sc.code}\\n```"
                for i, sc in enumerate(passing))
            content = f"Task `{task.task_id}` solutions already exist:\\n{existing}\\n\\nProvide a DIFFERENT approach."
        else:
            content = PromptBuilder.initial_prompt(task)
        try:
            raw = self.client.chat([LLMMessage("system", PromptBuilder.system_prompt()),
                                    LLMMessage("user", content)], temperature=temp)
        except Exception as e:
            self._log(f"  LLM error: {e}"); return None
        code = LLMClient._extract_code(raw)
        return SolvedCode(code, self._sb.run(code, task), self._sb.apply(code, task.test_inputs))

    def _is_dup(self, cand: SolvedCode, existing: list[SolvedCode]) -> bool:
        return any(cand.code.strip() == s.code.strip() or not cand.outputs_differ_from(s)
                   for s in existing)

    def solve(self, task: Task) -> EnsembleResult:
        self._log(f"[Ensemble] {task.task_id}")
        passing: list[SolvedCode] = []; partial: SolvedCode | None = None; calls = 0
        res = self._base.solve(task); calls += res.attempts
        sc = self._wrap(res, task)
        if sc:
            if sc.is_passing: passing.append(sc)
            elif not partial or sc.passed > partial.passed: partial = sc
        for i, t in enumerate(self._TEMPS[:self.da]):
            if len(passing) >= 2: break
            calls += 1; cand = self._div_pass(task, passing, t)
            if cand is None: continue
            if cand.is_passing:
                if not self._is_dup(cand, passing): passing.append(cand)
            elif not partial or cand.passed > partial.passed: partial = cand
        preds, strategy = self._ep.predict(task, passing, partial)
        return EnsembleResult(task.task_id, preds, passing, partial, strategy, calls)

    def solve_all(self, tasks: dict[TaskId, Task], max_tasks: int | None = None,
                  progress_fn=None) -> dict[TaskId, EnsembleResult]:
        results: dict[TaskId, EnsembleResult] = {}
        items = list(tasks.items())[:max_tasks] if max_tasks else list(tasks.items())
        for tid, task in items:
            results[tid] = self.solve(task)
            if progress_fn: progress_fn(1)
        return results


def ensemble_results_to_predictions(results: dict[TaskId, EnsembleResult]) -> dict[TaskId, list[Prediction]]:
    return {tid: r.predictions for tid, r in results.items()}


print("✓ Phase 4 로드 완료")
'''

# ── Test cells ─────────────────────────────────────────────────────────────────
CELL_TEST1 = '''\
# ── Test 1: 데이터 로드 ──────────────────────────────────────────────────────────
loader      = ARCDataLoader()
train_tasks = loader.load_training()
eval_tasks  = loader.load_evaluation()
test_tasks  = loader.load_test()

print(f"training   : {len(train_tasks)} tasks")
print(f"evaluation : {len(eval_tasks)} tasks")
print(f"test       : {len(test_tasks)} tasks")

_tid  = list(train_tasks.keys())[0]
_task = train_tasks[_tid]
print(f"\\n[{_tid}]  train={len(_task.train)}  test={len(_task.test_inputs)}")
print(f"  첫 번째 train input  : {len(_task.train[0].input)}×{len(_task.train[0].input[0])}")
print(f"  첫 번째 train output : {len(_task.train[0].output)}×{len(_task.train[0].output[0])}")
print("\\n✓ Test 1 통과")
'''

CELL_TEST2 = '''\
# ── Test 2: DSL 함수 검증 ─────────────────────────────────────────────────────
g = [[1,2,3],[4,5,6],[7,8,9]]

# 회전: 4번 돌면 원래대로
r90  = GeometricOps.rotate_90(g)
r180 = GeometricOps.rotate_180(g)
r270 = GeometricOps.rotate_270(g)
assert GeometricOps.rotate_90(r90)  == r180, "90x2 != 180"
assert GeometricOps.rotate_90(r180) == r270, "90x3 != 270"
assert GeometricOps.rotate_90(r270) == g,    "90x4 != identity"
print("✓ 회전 (90/180/270/identity)")

assert GeometricOps.flip_horizontal(g)[0] == [3,2,1], f"flip_h: {GeometricOps.flip_horizontal(g)[0]}"
assert GeometricOps.flip_vertical(g)[0]   == [7,8,9], f"flip_v: {GeometricOps.flip_vertical(g)[0]}"
assert GeometricOps.flip_diagonal_main(g)[0] == [1,4,7]
print("✓ 대칭")

assert GeometricOps.pad(g, top=1, left=1)[0] == [0,0,0,0]
assert GeometricOps.crop(g,0,0,2,2) == [[1,2],[4,5]]
assert GeometricOps.trim([[0,0,0],[0,5,0],[0,0,0]]) == [[5]]
print("✓ 패딩/crop/trim")

grid2 = [[1,1,0,2],[1,0,0,2],[0,0,3,0]]
objs = ObjectOps.find_objects(grid2)
assert len(objs)==3 and {o.color for o in objs}=={1,2,3}
print("✓ CCL 오브젝트 감지")

filled = ColorOps.flood_fill(grid2,(0,0),9)
assert filled[0][0]==9 and filled[0][2]==0
print("✓ Flood fill")

grid3 = [[1,1,2],[2,2,3]]
assert ColorOps.most_common_color(grid3)==2
assert ColorOps.least_common_color(grid3)==3
print("✓ 색상 카운팅")

dsl = GridDSL(g)
assert dsl.rotate(90).flip_h().grid is not None
print("✓ GridDSL 체이닝")

print("\\n✓ Test 2 통과")
'''

CELL_TEST3 = '''\
# ── Test 3: CodeSandbox ──────────────────────────────────────────────────────
sb    = CodeSandbox(timeout_sec=5.0)
task0 = train_tasks[list(train_tasks.keys())[0]]

# ① 정답 코드 (이 태스크의 패턴: 각 행을 좌우반전 후 행 자체를 3번, 행 블록을 3번 반복)
correct_code = """
def transform(grid):
    result = []
    for rb in range(3):
        for ri in range(len(grid)):
            row = grid[ri] if rb % 2 == 0 else grid[ri][::-1]
            result.append(row * 3)
    return result
"""
res = sb.run(correct_code, task0)
print(f"[정답 코드] {res.passed}/{res.total} passed  success={res.success}  {res.exec_time_ms:.0f}ms")
assert res.success, f"정답 코드 실패: {res.error}"

# ② 오답 코드
res2 = sb.run("def transform(g):\\n    return [[0]*6 for _ in range(6)]", task0)
print(f"[오답 코드] {res2.passed}/{res2.total} passed  success={res2.success}")
assert not res2.success

# ③ 타임아웃
sb1 = CodeSandbox(timeout_sec=1.0)
res3 = sb1.run("def transform(g):\\n    import time;time.sleep(30);return g", task0)
assert "Timeout" in res3.error, f"타임아웃 기대: {res3.error}"
print(f"[타임아웃]  error={res3.error!r}")

# ④ apply
outs = sb.apply(correct_code, task0.test_inputs)
assert outs[0] is not None
print(f"[apply]    output={len(outs[0])}×{len(outs[0][0])}")

print("\\n✓ Test 3 통과")
'''

CELL_TEST4 = '''\
# ── Test 4: 앙상블 전략 단위 테스트 ──────────────────────────────────────────
ok_sr  = SandboxResult(success=True,  passed=2, total=2, outputs=[])
bad_sr = SandboxResult(success=False, passed=1, total=2, outputs=[])
sc1 = SolvedCode("def transform(g):\\n    return g",     ok_sr,  [[[0]]])
sc2 = SolvedCode("def transform(g):\\n    return [[1]]", ok_sr,  [[[1]]])
sc_p = SolvedCode("def transform(g): return g",          bad_sr, [[[0]]])

_task1 = eval_tasks[list(eval_tasks.keys())[0]]
ep = EnsemblePredictor(CodeSandbox())

_, s = ep.predict(_task1, [sc1, sc2], None);    assert s=="diverse_2";          print(f"✓ diverse_2")
_, s = ep.predict(_task1, [sc1],      None);    assert s=="single_fallback";    print(f"✓ single_fallback")
_, s = ep.predict(_task1, [],         sc_p);    assert s=="partial_heuristic";  print(f"✓ partial_heuristic")
_, s = ep.predict(_task1, [],         None);    assert s=="full_heuristic";     print(f"✓ full_heuristic")

print("\\n✓ Test 4 통과")
'''

CELL_TEST5 = '''\
# ── Test 5: 더미 solver로 전체 파이프라인 E2E ───────────────────────────────
from tqdm.notebook import tqdm as _ntqdm

_sample = dict(list(eval_tasks.items())[:5])
_preds: dict[TaskId, list[Prediction]] = {}

for _tid, _task in _ntqdm(_sample.items(), desc="dummy E2E", unit="task"):
    _preds[_tid] = dummy_solver(_task)

_scores = ARCEvaluator().evaluate(_preds, _sample)
print(f"\\n더미 solver 점수 (0이 정상): {_scores['overall_score']:.4f}")
assert _scores["overall_score"] == 0.0

_path = ARCSubmissionWriter().save(_preds, "submission_test.json")
assert _path.exists()

print("\\n✓ Test 5 통과 — 전체 파이프라인 정상")
'''

# ── Config & Run ───────────────────────────────────────────────────────────────
CELL_CONFIG = '''\
# ═══════════════════════════════════════════════════════════════════
# ⚙️ 설정 (여기만 수정)
# ═══════════════════════════════════════════════════════════════════
PROVIDER            = "openai"        # "openai" | "anthropic"
MODEL               = "o4-mini"
API_KEY             = None            # None → 환경변수 사용
SPLIT               = "evaluation"    # "training" | "evaluation" | "test"
MAX_TASKS           = None            # None = 전체 / 정수 = 처음 N개
MAX_ATTEMPTS        = 5
DIVERSITY_ATTEMPTS  = 2
SANDBOX_TIMEOUT     = 10.0
VERBOSE             = False
OUTPUT_FILE         = "submission.json"

print(f"설정: {PROVIDER}/{MODEL}  split={SPLIT}  max_tasks={MAX_TASKS}")
'''

CELL_RUN = '''\
# ═══════════════════════════════════════════════════════════════════
# 🚀 메인 실행
# ═══════════════════════════════════════════════════════════════════
from tqdm.notebook import tqdm as _nb_tqdm

_loader = ARCDataLoader()
_tasks  = {"training": _loader.load_training,
           "evaluation": _loader.load_evaluation,
           "test": _loader.load_test}[SPLIT]()
if MAX_TASKS: _tasks = dict(list(_tasks.items())[:MAX_TASKS])
print(f"✓ {len(_tasks)} tasks 로드")

_client = (OpenAIClient(api_key=API_KEY, model=MODEL)
           if PROVIDER == "openai"
           else AnthropicClient(api_key=API_KEY, model=MODEL))
print(f"✓ LLM 클라이언트: {PROVIDER}/{MODEL}")

_synth = DiverseSynthesizer(
    client=_client, max_attempts=MAX_ATTEMPTS,
    diversity_attempts=DIVERSITY_ATTEMPTS,
    sandbox_timeout=SANDBOX_TIMEOUT, verbose=VERBOSE,
)

_results: dict[TaskId, EnsembleResult] = {}
_t0 = time.perf_counter()

with _nb_tqdm(total=len(_tasks), desc="Solving", unit="task") as _pbar:
    for _tid, _task in _tasks.items():
        _results[_tid] = _synth.solve(_task)
        _solved = sum(1 for r in _results.values() if r.passing_codes)
        _pbar.set_postfix(solved=_solved, strat=_results[_tid].strategy[:4])
        _pbar.update(1)

_elapsed = time.perf_counter() - _t0

# 평가
_eval_scores = None
if SPLIT in ("training", "evaluation"):
    _eval_scores = ARCEvaluator().evaluate(ensemble_results_to_predictions(_results), _tasks)

# 저장
_saved = ARCSubmissionWriter().save(ensemble_results_to_predictions(_results), OUTPUT_FILE)

# 요약
_strat = {}
for _r in _results.values(): _strat[_r.strategy] = _strat.get(_r.strategy, 0) + 1
_tot   = len(_results)
_calls = sum(r.total_llm_calls for r in _results.values())

print("\\n" + "="*64)
print(" ARC-AGI-2 실행 요약")
print("="*64)
print(f"  총 tasks   : {_tot}")
print(f"  처리 시간  : {_elapsed:.1f}s  ({_elapsed/_tot:.1f}s/task)")
print(f"  총 LLM 호출: {_calls}  (평균 {_calls/_tot:.1f}/task)")
print()
for _s, _n in sorted(_strat.items()):
    _em = {"diverse_2":"★★","single_fallback":"★·","partial_heuristic":"~·","full_heuristic":"··"}.get(_s,"??")
    print(f"  {_em} {_s:<22}: {_n:4d}  ({_n/_tot*100:.1f}%)")
_lv = _strat.get("diverse_2",0) + _strat.get("single_fallback",0)
print(f"\\n  LLM 통과 있음: {_lv:4d}  ({_lv/_tot*100:.1f}%)")
if _eval_scores:
    _pf = sum(1 for s in _eval_scores["task_scores"].values() if s==1.0)
    print(f"\\n  평가 점수: {_eval_scores['overall_score']:.4f}")
    print(f"  완벽 task : {_pf}/{_tot}")
print("="*64)
print(f"\\n제출 파일: {_saved}")
'''

# ══════════════════════════════════════════════════════════════════════════════
# 노트북 조립
# ══════════════════════════════════════════════════════════════════════════════

cells = [
    md_cell(MD_TITLE),
    md_cell("## 0. 패키지 설치"),
    code_cell(CELL_INSTALL),
    md_cell("## 1. 공통 임포트 및 전역 경로"),
    code_cell(CELL_IMPORTS),
    md_cell("## Phase 1: 데이터 파이프라인"),
    code_cell(CELL_PHASE1),
    md_cell("## Phase 2: Grid DSL\n`arc_dsl.py` 를 디스크에 저장하고 import 합니다."),
    code_cell(CELL_PHASE2),
    md_cell("## Phase 3: 프로그램 합성 엔진"),
    code_cell(CELL_PHASE3),
    md_cell("## Phase 4: 앙상블 전략"),
    code_cell(CELL_PHASE4),
    md_cell("## 🧪 자체 테스트 (LLM 불필요)"),
    code_cell(CELL_TEST1),
    code_cell(CELL_TEST2),
    code_cell(CELL_TEST3),
    code_cell(CELL_TEST4),
    code_cell(CELL_TEST5),
    md_cell("## ⚙️ 설정 및 🚀 실행"),
    code_cell(CELL_CONFIG),
    code_cell(CELL_RUN),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path("arc_agi2_pipeline.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"✓ 노트북 생성: {out}  ({out.stat().st_size // 1024} KB)")
