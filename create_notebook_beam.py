#!/usr/bin/env python3
"""arc_agi2_beam.ipynb 생성 스크립트 — LLM 없는 NumPy 빔서치 버전."""

import json, uuid
from pathlib import Path

def code_cell(source):
    return {"cell_type":"code","execution_count":None,
            "id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":source}

def md_cell(source):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":source}

# ══════════════════════════════════════════════════════════════════════════════
# CELL SOURCES
# ══════════════════════════════════════════════════════════════════════════════

MD_TITLE = """\
# ARC-AGI-2 — Heuristic Beam Search Engine (LLM-Free)
> **Icecuber / StochasticGoose 스타일** | NumPy 벡터화 DSL + 휴리스틱 빔서치
>
> | 모듈 | 내용 |
> |---|---|
> | Phase 1 | 데이터 파이프라인 (DataLoader / Evaluator / SubmissionWriter) |
> | Phase 2 | NumPy DSL 프리미티브 (8방향 대칭 / 스케일 / 타일 / 중력 / 색상) |
> | Phase 3 | 출력 크기 예측기 (결정론적 규칙 + 통계 폴백) |
> | Phase 4 | 연산자 라이브러리 (고정 + Task 적응형) |
> | Phase 5 | 휴리스틱 빔서치 엔진 (Early-Stopping / Depth-4) |
> | Phase 6 | 병렬 ARC Solver + Validation 파이프라인 |"""

CELL_INSTALL = """\
import subprocess, sys
for _p in ["numpy", "tqdm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", _p, "-q"], check=False)
print("✓ 패키지 확인 완료")"""

CELL_IMPORTS = """\
from __future__ import annotations
import copy, json, os, sys, time, warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
warnings.filterwarnings("ignore")

# ── 전역 경로 ─────────────────────────────────────────────────────────────────
NOTEBOOK_DIR   = Path(os.getcwd())
_KAGGLE_DATA   = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
_LOCAL_DATA    = Path(".")
DATA_DIR       = _KAGGLE_DATA if _KAGGLE_DATA.exists() else _LOCAL_DATA

# ── 탐색 설정 ─────────────────────────────────────────────────────────────────
BEAM_WIDTH     = 15      # 빔 너비
MAX_DEPTH      = 4       # 최대 탐색 깊이
TASK_TIMEOUT   = 60.0    # task 당 최대 탐색 시간(초)
MAX_WORKERS    = max(1, os.cpu_count() or 1)  # 병렬 워커 수

# ── 타입 별칭 ─────────────────────────────────────────────────────────────────
Grid   = list[list[int]]
NGrid  = np.ndarray        # int8 numpy array
TaskId = str

print(f"✓ 임포트 완료")
print(f"  DATA_DIR  = {DATA_DIR.resolve()}  (exists={DATA_DIR.exists()})")
print(f"  CPU cores = {MAX_WORKERS}  beam_width={BEAM_WIDTH}  max_depth={MAX_DEPTH}")"""

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
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def _load(self, fname):
        with (self.data_dir / fname).open(encoding="utf-8") as f:
            return json.load(f)

    def _build(self, challenges, solutions=None):
        tasks = {}
        for tid, data in challenges.items():
            tasks[tid] = Task(
                task_id=tid,
                train=[Pair(p["input"], p["output"]) for p in data["train"]],
                test_inputs=[p["input"] for p in data["test"]],
                test_outputs=solutions[tid] if solutions else [],
            )
        return tasks

    def load_training(self):
        return self._build(self._load(self.FILES["training_challenges"]),
                           self._load(self.FILES["training_solutions"]))
    def load_evaluation(self):
        return self._build(self._load(self.FILES["evaluation_challenges"]),
                           self._load(self.FILES["evaluation_solutions"]))
    def load_test(self):
        return self._build(self._load(self.FILES["test_challenges"]))


class ARCEvaluator:
    @staticmethod
    def grids_equal(a, b):
        return len(a)==len(b) and all(ra==rb for ra,rb in zip(a,b))

    def score_prediction(self, p: Prediction, gt: Grid) -> int:
        return 1 if (self.grids_equal(p.attempt_1, gt) or
                     self.grids_equal(p.attempt_2, gt)) else 0

    def evaluate(self, all_preds, tasks):
        correct = total = 0
        scores = {}
        for tid, task in tasks.items():
            if not task.test_outputs: continue
            preds = all_preds.get(tid, [])
            if not preds:
                scores[tid] = 0.0; total += len(task.test_outputs); continue
            c = sum(self.score_prediction(p,g) for p,g in zip(preds, task.test_outputs))
            scores[tid] = c / len(task.test_outputs)
            correct += c; total += len(task.test_outputs)
        return {"task_scores": scores,
                "overall_score": correct/total if total else 0.0}


class ARCSubmissionWriter:
    def save(self, all_preds, out="submission.json"):
        path = Path(out)
        sub = {tid: [{"attempt_1":p.attempt_1,"attempt_2":p.attempt_2}
                     for p in preds]
               for tid, preds in all_preds.items()}
        with path.open("w", encoding="utf-8") as f:
            json.dump(sub, f)
        print(f"[SubmissionWriter] {path.resolve()}  ({len(sub)} tasks)")
        return path


print("✓ Phase 1 로드 완료")
'''

# ── Phase 2: NumPy DSL ─────────────────────────────────────────────────────────
CELL_PHASE2 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 2: NumPy DSL 프리미티브
# ═══════════════════════════════════════════════════════════════════

def g2n(grid) -> NGrid:
    """list[list[int]] → numpy int8 배열."""
    return np.array(grid, dtype=np.int8)

def n2g(arr: NGrid) -> Grid:
    """numpy 배열 → list[list[int]]."""
    return arr.tolist()

# ── 8방향 대칭 변환 ─────────────────────────────────────────────────────────────

def sym_all(g: NGrid) -> list[tuple[str, NGrid]]:
    """8가지 회전/대칭 변환 배열 반환."""
    variants = []
    for k in range(4):
        r = np.rot90(g, k)
        variants.append((f"rot{k*90}", r))
        variants.append((f"rot{k*90}_fliplr", np.fliplr(r)))
    return variants

# ── 스케일 및 타일 ──────────────────────────────────────────────────────────────

def scale_up(g: NGrid, ry: int, rx: int) -> NGrid:
    return np.repeat(np.repeat(g, ry, axis=0), rx, axis=1)

def scale_down(g: NGrid, ry: int, rx: int) -> NGrid:
    return g[::ry, ::rx]

def tile(g: NGrid, ny: int, nx: int) -> NGrid:
    return np.tile(g, (ny, nx))

def crop_to_content(g: NGrid, bg: int = 0) -> NGrid:
    """배경색 테두리 제거."""
    rows = np.any(g != bg, axis=1)
    cols = np.any(g != bg, axis=0)
    if not rows.any() or not cols.any():
        return g
    return g[np.ix_(rows, cols)]

def pad_to_shape(g: NGrid, H: int, W: int, fill: int = 0) -> NGrid:
    """중앙 정렬 패딩으로 목표 크기 맞춤."""
    h, w = g.shape
    result = np.full((H, W), fill, dtype=np.int8)
    r0 = (H - h) // 2; c0 = (W - w) // 2
    r0 = max(0, r0); c0 = max(0, c0)
    he = min(H, r0 + h); we = min(W, c0 + w)
    result[r0:he, c0:we] = g[:he-r0, :we-c0]
    return result

def extend_to_shape(g: NGrid, H: int, W: int) -> NGrid:
    """패턴을 타일링하여 목표 크기로 확장."""
    h, w = g.shape
    if h == 0 or w == 0: return np.zeros((H, W), dtype=np.int8)
    ny = -(-H // h); nx = -(-W // w)          # ceiling division
    tiled = np.tile(g, (ny, nx))
    return tiled[:H, :W]

# ── 중력(Gravity) ───────────────────────────────────────────────────────────────

def gravity_down(g: NGrid, bg: int = 0) -> NGrid:
    result = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = g[:, c]; nz = col[col != bg]
        result[len(nz):, c] = bg; result[g.shape[0]-len(nz):, c] = nz
    return result

def gravity_up(g: NGrid, bg: int = 0) -> NGrid:
    result = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = g[:, c]; nz = col[col != bg]
        result[:len(nz), c] = nz
    return result

def gravity_left(g: NGrid, bg: int = 0) -> NGrid:
    result = np.full_like(g, bg)
    for r in range(g.shape[0]):
        row = g[r, :]; nz = row[row != bg]
        result[r, :len(nz)] = nz
    return result

def gravity_right(g: NGrid, bg: int = 0) -> NGrid:
    result = np.full_like(g, bg)
    for r in range(g.shape[0]):
        row = g[r, :]; nz = row[row != bg]
        result[r, g.shape[1]-len(nz):] = nz
    return result

# ── 색상 연산 ───────────────────────────────────────────────────────────────────

def recolor(g: NGrid, mapping: dict[int,int]) -> NGrid:
    """색상 매핑 적용."""
    result = g.copy()
    for src, dst in mapping.items():
        result[g == src] = dst
    return result

def swap_colors(g: NGrid, c1: int, c2: int) -> NGrid:
    return recolor(g, {c1: c2, c2: c1})

def replace_color(g: NGrid, src: int, dst: int) -> NGrid:
    result = g.copy(); result[g == src] = dst; return result

def normalize_colors(g: NGrid) -> tuple[NGrid, dict[int,int]]:
    """색상 빈도수 기준으로 정규화 (0=가장 많은 색). mapping: normalized→original."""
    flat = g.flatten()
    counts = Counter(flat.tolist())
    sorted_c = [c for c,_ in sorted(counts.items(), key=lambda x:-x[1])]
    fwd = {c: i for i, c in enumerate(sorted_c)}   # original → normalized
    inv = {i: c for c, i in fwd.items()}            # normalized → original
    return np.vectorize(fwd.get)(g).astype(np.int8), inv

def denormalize_colors(g: NGrid, inv: dict[int,int]) -> NGrid:
    return np.vectorize(lambda x: inv.get(x, x))(g).astype(np.int8)

# ── 오브젝트 연산 ───────────────────────────────────────────────────────────────

def find_objects(g: NGrid, bg: int = 0, connectivity: int = 4) -> list[np.ndarray]:
    """연결 컴포넌트 추출 → 각 오브젝트의 셀 좌표 목록."""
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    objects = []
    for r in range(H):
        for c in range(W):
            if visited[r,c] or g[r,c] == bg: continue
            color = g[r,c]; cells = []
            queue = deque([(r,c)]); visited[r,c] = True
            while queue:
                cr,cc = queue.popleft(); cells.append((cr,cc))
                for dr,dc in dirs:
                    nr,nc = cr+dr,cc+dc
                    if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and g[nr,nc]==color:
                        visited[nr,nc]=True; queue.append((nr,nc))
            objects.append(np.array(cells))
    return objects

def object_bbox(cells: np.ndarray) -> tuple[int,int,int,int]:
    """오브젝트 셀의 bounding box: (r0,c0,r1,c1) inclusive."""
    r0,c0 = cells.min(axis=0); r1,c1 = cells.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)

def extract_object(g: NGrid, cells: np.ndarray, bg: int = 0) -> NGrid:
    """오브젝트를 최소 bounding box로 잘라낸다."""
    r0,c0,r1,c1 = object_bbox(cells)
    sub = np.full((r1-r0+1, c1-c0+1), bg, dtype=np.int8)
    for r,c in cells:
        sub[r-r0, c-c0] = g[r,c]
    return sub

def flood_fill(g: NGrid, r0: int, c0: int, new_color: int,
               connectivity: int = 4) -> NGrid:
    """시작 위치에서 flood fill."""
    H, W = g.shape
    if not (0<=r0<H and 0<=c0<W): return g.copy()
    old = g[r0,c0]
    if old == new_color: return g.copy()
    result = g.copy()
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8: dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    queue = deque([(r0,c0)]); result[r0,c0] = new_color
    while queue:
        r,c = queue.popleft()
        for dr,dc in dirs:
            nr,nc = r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and result[nr,nc]==old:
                result[nr,nc]=new_color; queue.append((nr,nc))
    return result

def fill_outer_region(g: NGrid, fill_color: int = 0, bg: int = 0) -> NGrid:
    """외부 배경 영역을 fill_color로 채운다 (interior는 유지)."""
    H, W = g.shape
    result = g.copy()
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    visited = np.zeros((H,W), dtype=bool)
    queue = deque()
    for r in range(H):
        for c in [0, W-1]:
            if result[r,c]==bg and not visited[r,c]:
                visited[r,c]=True; queue.append((r,c))
    for c in range(W):
        for r in [0, H-1]:
            if result[r,c]==bg and not visited[r,c]:
                visited[r,c]=True; queue.append((r,c))
    while queue:
        r,c = queue.popleft(); result[r,c] = fill_color
        for dr,dc in dirs:
            nr,nc = r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and result[nr,nc]==bg:
                visited[nr,nc]=True; queue.append((nr,nc))
    return result

def outline_objects(g: NGrid, bg: int = 0) -> NGrid:
    """오브젝트 내부를 bg로 채워 윤곽선만 남긴다."""
    result = g.copy()
    H, W = g.shape
    for r in range(1, H-1):
        for c in range(1, W-1):
            if g[r,c] != bg:
                neighbors = [g[r+dr,c+dc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]]
                if all(n == g[r,c] for n in neighbors):
                    result[r,c] = bg
    return result

def color_counts_grid(g: NGrid, bg: int = 0) -> NGrid:
    """각 색상의 픽셀 수를 1×K 격자로 반환 (색상 오름차순)."""
    counts = Counter(g[g != bg].tolist())
    if not counts: return np.zeros((1,1), dtype=np.int8)
    sorted_colors = sorted(counts.keys())
    row = [counts[c] for c in sorted_colors]
    return np.array([row], dtype=np.int8)


print("✓ Phase 2 로드 완료 (NumPy DSL 프리미티브)")
'''

# ── Phase 3: Shape Predictor ───────────────────────────────────────────────────
CELL_PHASE3 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 3: 출력 크기 예측기
# ═══════════════════════════════════════════════════════════════════

class ShapePredictor:
    """Train 쌍에서 출력 크기 관계를 분석하여 Test 출력 크기를 예측한다."""

    def analyze(self, task: Task) -> "ShapeRule":
        pairs = [(p.input, p.output) for p in task.train if p.output is not None]
        if not pairs: return ShapeRule("unknown", None, None)

        in_shapes  = [(len(i),   len(i[0])   if i   else 0) for i,_ in pairs]
        out_shapes = [(len(o),   len(o[0])   if o   else 0) for _,o in pairs]

        # Rule 1: 고정 출력 크기
        if len(set(out_shapes)) == 1 and len(set(in_shapes)) > 1:
            return ShapeRule("fixed", out_shapes[0], None)

        # Rule 2: 입출력 동일 크기
        if all(i==o for i,o in zip(in_shapes, out_shapes)):
            return ShapeRule("identity", None, None)

        # Rule 3: 상수 배율
        row_scales = [Fraction(o[0], i[0]) for i,o in zip(in_shapes, out_shapes) if i[0]>0]
        col_scales = [Fraction(o[1], i[1]) for i,o in zip(in_shapes, out_shapes) if i[1]>0]
        if len(set(row_scales))==1 and len(set(col_scales))==1:
            return ShapeRule("scale", None, (row_scales[0], col_scales[0]))

        # Rule 4: 전치 (행↔열 교환)
        if all((i[0]==o[1] and i[1]==o[0]) for i,o in zip(in_shapes, out_shapes)):
            return ShapeRule("transpose", None, None)

        # Rule 5: 크롭 (출력이 입력보다 작음)
        if all(o[0]<=i[0] and o[1]<=i[1] for i,o in zip(in_shapes, out_shapes)):
            if len(set(out_shapes))==1:
                return ShapeRule("fixed", out_shapes[0], None)

        # 폴백: 학습 출력 크기 중 최빈값
        mode = Counter(out_shapes).most_common(1)[0][0]
        return ShapeRule("fallback_mode", mode, None)

    def predict(self, task: Task, test_input: Grid) -> tuple[int,int] | None:
        rule = self.analyze(task)
        H, W = len(test_input), len(test_input[0]) if test_input else 1

        if rule.kind == "identity":       return (H, W)
        if rule.kind == "fixed":          return rule.fixed
        if rule.kind == "fallback_mode":  return rule.fixed
        if rule.kind == "transpose":      return (W, H)
        if rule.kind == "scale" and rule.scale:
            rs, cs = rule.scale
            return (int(H * rs), int(W * cs))
        return None


@dataclass
class ShapeRule:
    kind: str
    fixed: tuple[int,int] | None
    scale: tuple[Fraction,Fraction] | None


_shape_predictor = ShapePredictor()
print("✓ Phase 3 로드 완료 (ShapePredictor)")
'''

# ── Phase 4: Operation Library ─────────────────────────────────────────────────
CELL_PHASE4 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 4: 연산자 라이브러리
# ═══════════════════════════════════════════════════════════════════

# ── 고정 연산자 (항상 포함) ────────────────────────────────────────────────────

FIXED_OPS: list[tuple[str, Callable]] = [
    # 8방향 대칭
    ("rot0",     lambda g: g.copy()),
    ("rot90",    lambda g: np.rot90(g, 1).copy()),
    ("rot180",   lambda g: np.rot90(g, 2).copy()),
    ("rot270",   lambda g: np.rot90(g, 3).copy()),
    ("flip_lr",  lambda g: np.fliplr(g).copy()),
    ("flip_ud",  lambda g: np.flipud(g).copy()),
    ("flip_d0",  lambda g: g.T.copy()),
    ("flip_d1",  lambda g: np.rot90(g.T, 2).copy()),
    # 스케일
    ("scale2",   lambda g: scale_up(g, 2, 2)),
    ("scale3",   lambda g: scale_up(g, 3, 3)),
    ("scale_r2", lambda g: scale_up(g, 2, 1)),
    ("scale_c2", lambda g: scale_up(g, 1, 2)),
    ("scale_r3", lambda g: scale_up(g, 3, 1)),
    ("scale_c3", lambda g: scale_up(g, 1, 3)),
    # 타일
    ("tile22",   lambda g: tile(g, 2, 2)),
    ("tile12",   lambda g: tile(g, 1, 2)),
    ("tile21",   lambda g: tile(g, 2, 1)),
    ("tile13",   lambda g: tile(g, 1, 3)),
    ("tile31",   lambda g: tile(g, 3, 1)),
    ("tile33",   lambda g: tile(g, 3, 3)),
    # 크롭/패딩
    ("crop_bg0", lambda g: crop_to_content(g, 0)),
    # 중력
    ("grav_d",   lambda g: gravity_down(g, 0)),
    ("grav_u",   lambda g: gravity_up(g, 0)),
    ("grav_l",   lambda g: gravity_left(g, 0)),
    ("grav_r",   lambda g: gravity_right(g, 0)),
    # 윤곽/채우기
    ("outline",  lambda g: outline_objects(g, 0)),
    ("fill_outer", lambda g: fill_outer_region(g, 0, 0)),
    # 색상 정규화 (빈도 기준 재매핑)
    ("norm_color", lambda g: normalize_colors(g)[0]),
    # 반전 색 구성
    ("invert_bg", lambda g: replace_color(g, 0,
                             int(Counter(g.flatten().tolist()).most_common()[0][0])
                             if Counter(g.flatten().tolist()).most_common()[0][0] != 0
                             else int(Counter(g.flatten().tolist()).most_common()[-1][0]))),
]

# ── Task 적응형 연산자 생성기 ──────────────────────────────────────────────────

def build_task_ops(task: Task) -> list[tuple[str, Callable]]:
    """Task 에 등장하는 색상 조합으로 색상 특화 연산자를 생성한다."""
    ops = list(FIXED_OPS)

    # 등장 색상 수집
    color_set: set[int] = set()
    for pair in task.train:
        color_set.update(int(c) for row in pair.input for c in row)
        if pair.output:
            color_set.update(int(c) for row in pair.output for c in row)
    colors = sorted(color_set)

    # 색상 쌍 교환
    for i, c1 in enumerate(colors):
        for c2 in colors[i+1:]:
            name = f"swap_{c1}_{c2}"
            ops.append((name, (lambda a, b: (lambda g: swap_colors(g, a, b)))(c1, c2)))

    # 단일 색상 보존 (나머지를 0으로)
    for c in colors:
        if c == 0: continue
        name = f"keep_{c}"
        ops.append((name, (lambda a: (lambda g: replace_color((g == a).astype(np.int8) * a,
                                                               0, 0)))(c)))

    # 배경색 변경 (0 ↔ 가장 흔한 비-0 색)
    if len(colors) >= 2:
        bg_cands = [c for c in colors if c != 0][:2]
        for bg in bg_cands:
            name = f"rebg_{bg}"
            ops.append((name, (lambda b: (lambda g: swap_colors(g, 0, b)))(bg)))

    # 출력 크기 예측 → 확장 / 크롭 연산자
    pred_shape = _shape_predictor.predict(task, task.test_inputs[0])
    if pred_shape and pred_shape != (len(task.train[0].input),
                                      len(task.train[0].input[0]) if task.train[0].input else 1):
        H, W = pred_shape
        ops.append((f"extend_{H}x{W}", lambda g, h=H, w=W: extend_to_shape(g, h, w)))
        ops.append((f"pad_{H}x{W}",    lambda g, h=H, w=W: pad_to_shape(g, h, w)))

    return ops


print(f"✓ Phase 4 로드 완료 (고정 연산자 {len(FIXED_OPS)}개 + Task 적응형)")
'''

# ── Phase 5: Beam Search Engine ────────────────────────────────────────────────
CELL_PHASE5 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 5: 휴리스틱 빔서치 엔진
# ═══════════════════════════════════════════════════════════════════

def pixel_score(pred: NGrid, target: NGrid) -> float:
    """두 격자의 픽셀 일치율 (크기 다르면 0.0, 일치하면 1.0)."""
    if pred.shape != target.shape:
        return 0.0
    return float(np.mean(pred == target))

def score_all(preds: list[NGrid], targets: list[NGrid]) -> float:
    """모든 학습 쌍의 평균 픽셀 일치율."""
    if not preds or not targets: return 0.0
    return float(np.mean([pixel_score(p, t) for p, t in zip(preds, targets)]))


class BeamSearchEngine:
    """
    DSL 연산자 조합을 트리 탐색으로 찾는 빔서치 엔진.

    알고리즘
    --------
    1. 초기 상태: 원본 Train Input 목록
    2. 각 깊이에서 현재 빔의 모든 상태 × 모든 연산자 적용
    3. 점수가 1.0 인 후보 발견 시 즉시 반환 (Early Stopping)
    4. 상위 BEAM_WIDTH 개만 유지
    5. 최대 MAX_DEPTH 깊이까지 반복
    """

    def __init__(self, beam_width: int = BEAM_WIDTH, max_depth: int = MAX_DEPTH,
                 timeout: float = TASK_TIMEOUT):
        self.beam_width = beam_width
        self.max_depth  = max_depth
        self.timeout    = timeout

    def solve(self, task: Task) -> "BeamResult":
        t0 = time.perf_counter()

        train_in  = [g2n(p.input)  for p in task.train]
        train_out = [g2n(p.output) for p in task.train if p.output is not None]

        if len(train_in) != len(train_out):
            return BeamResult(task.task_id, False, [], None, 0.0, "train/output mismatch")

        ops = build_task_ops(task)

        # 초기 점수 (identity)
        init_score = score_all(train_in, train_out)
        if init_score == 1.0:
            return BeamResult(task.task_id, True, [], train_in, init_score, "identity")

        # 빔: [(score, op_sequence, current_grids)]
        beam: list[tuple[float, list[str], list[NGrid]]] = [
            (init_score, [], list(train_in))
        ]
        best = beam[0]
        solutions: list[tuple[list[str], list[NGrid]]] = []

        for depth in range(self.max_depth):
            if time.perf_counter() - t0 > self.timeout:
                break

            candidates: list[tuple[float, list[str], list[NGrid]]] = []

            for cur_score, op_seq, cur_grids in beam:
                for op_name, op_fn in ops:
                    try:
                        new_grids = [op_fn(g) for g in cur_grids]
                        # 크기 0 격자 스킵
                        if any(g.size == 0 for g in new_grids): continue
                        s = score_all(new_grids, train_out)
                        new_seq = op_seq + [op_name]

                        if s == 1.0:
                            solutions.append((new_seq, new_grids))
                        else:
                            candidates.append((s, new_seq, new_grids))
                    except Exception:
                        continue

            # 완전 일치 2개 수집 시 조기 종료
            if len(solutions) >= 2:
                break

            if not candidates and not solutions:
                break

            # 중복 제거 (같은 점수 + 동일 그리드 → 제거)
            seen = set()
            unique = []
            for s, seq, grids in candidates:
                key = (s, tuple(seq[-1:]))
                if key not in seen:
                    seen.add(key); unique.append((s, seq, grids))

            unique.sort(key=lambda x: -x[0])
            beam = unique[:self.beam_width]

            if beam and beam[0][0] > best[0]:
                best = beam[0]

            # 해가 1개 이상이면 완전 탐색 계속
            if solutions: break

        elapsed = time.perf_counter() - t0

        if solutions:
            # attempt_1: 가장 간결한 코드(짧은 시퀀스)
            solutions.sort(key=lambda x: len(x[0]))
            seq1, grids1 = solutions[0]
            seq2, grids2 = solutions[1] if len(solutions) > 1 else (seq1, grids1)
            return BeamResult(task.task_id, True, seq1, grids1, 1.0, "",
                              second_seq=seq2, second_grids=grids2)

        # 부분 일치: best 반환
        s, seq, grids = best
        return BeamResult(task.task_id, False, seq, grids, s,
                          f"best_partial {s:.3f} depth={self.max_depth}")

    def apply_sequence(self, seq: list[str], inputs: list[Grid],
                       task: Task) -> list[NGrid]:
        """검증된 연산 시퀀스를 임의 입력에 적용."""
        ops_dict = {name: fn for name, fn in build_task_ops(task)}
        grids = [g2n(inp) for inp in inputs]
        for op_name in seq:
            fn = ops_dict.get(op_name)
            if fn is None: continue
            try:
                grids = [fn(g) for g in grids]
            except Exception:
                pass
        return grids


@dataclass
class BeamResult:
    task_id: TaskId
    success: bool
    seq1: list[str]
    grids1: list[NGrid] | None
    score: float
    error: str = ""
    second_seq: list[str] = field(default_factory=list)
    second_grids: list[NGrid] | None = None

    def __str__(self):
        status = "✓" if self.success else f"~{self.score:.2f}"
        return f"[{status}] {self.task_id}  seq={self.seq1}"


_beam_engine = BeamSearchEngine()
print(f"✓ Phase 5 로드 완료 (BeamSearchEngine  beam={BEAM_WIDTH}  depth={MAX_DEPTH})")
'''

# ── Phase 6: Solver + Parallel Pipeline ───────────────────────────────────────
CELL_PHASE6 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 6: ARC Solver + 병렬 검증 파이프라인
# ═══════════════════════════════════════════════════════════════════

class HeuristicFallback:
    """빔서치 실패 시 기본 폴백 예측 생성."""

    @staticmethod
    def best_guess(task: Task, test_inp: Grid) -> Grid:
        """출력 크기를 맞추고 학습 출력 지배 색으로 채운다."""
        pred_shape = _shape_predictor.predict(task, test_inp)
        H = pred_shape[0] if pred_shape else len(test_inp)
        W = pred_shape[1] if pred_shape else (len(test_inp[0]) if test_inp else 1)
        cnt: Counter = Counter()
        for p in task.train:
            if p.output:
                for row in p.output: cnt.update(row)
        dominant = cnt.most_common(1)[0][0] if cnt else 0
        return [[int(dominant)] * W for _ in range(H)]

    @staticmethod
    def copy_input(inp: Grid) -> Grid:
        return copy.deepcopy(inp)


class ARCSolver:
    """
    빔서치 엔진 + 폴백 전략을 통합하여 Task → list[Prediction] 을 생성한다.
    ThreadPoolExecutor 로 병렬 처리 지원.
    """

    def __init__(self, beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH,
                 timeout=TASK_TIMEOUT, max_workers=MAX_WORKERS):
        self.engine   = BeamSearchEngine(beam_width, max_depth, timeout)
        self.fallback = HeuristicFallback()
        self.max_workers = max_workers

    def solve_task(self, task: Task) -> tuple[TaskId, list[Prediction], BeamResult]:
        result = self.engine.solve(task)
        predictions = self._build_predictions(task, result)
        return task.task_id, predictions, result

    def _ngrid_to_list(self, ng: NGrid | None, fallback: Grid) -> Grid:
        if ng is None: return fallback
        try: return n2g(ng)
        except Exception: return fallback

    def _build_predictions(self, task: Task, result: BeamResult) -> list[Prediction]:
        preds = []
        for i, inp in enumerate(task.test_inputs):
            fb1 = self.fallback.best_guess(task, inp)
            fb2 = self.fallback.copy_input(inp)

            if result.success and result.grids1 is not None:
                # attempt_1: 첫 번째 성공 시퀀스를 test input 에 적용
                test_g1 = self.engine.apply_sequence(result.seq1, [inp], task)
                a1 = self._ngrid_to_list(test_g1[0] if test_g1 else None, fb1)
                # attempt_2: 두 번째 시퀀스 (있으면) 또는 폴백
                if result.second_seq and result.second_seq != result.seq1:
                    test_g2 = self.engine.apply_sequence(result.second_seq, [inp], task)
                    a2 = self._ngrid_to_list(test_g2[0] if test_g2 else None, fb2)
                else:
                    a2 = fb2
            elif result.grids1 is not None and result.score > 0.5:
                # 부분 성공: 최선 시퀀스 적용
                test_g1 = self.engine.apply_sequence(result.seq1, [inp], task)
                a1 = self._ngrid_to_list(test_g1[0] if test_g1 else None, fb1)
                a2 = fb1
            else:
                # 완전 실패: 폴백
                a1 = fb1; a2 = fb2

            preds.append(Prediction(attempt_1=a1, attempt_2=a2))
        return preds

    def solve_all(self, tasks: dict[TaskId, Task],
                  progress_fn=None) -> tuple[dict[TaskId, list[Prediction]],
                                             dict[TaskId, BeamResult]]:
        all_preds: dict[TaskId, list[Prediction]] = {}
        all_results: dict[TaskId, BeamResult]     = {}

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {ex.submit(self.solve_task, task): tid
                        for tid, task in tasks.items()}
                for fut in as_completed(futs):
                    try:
                        tid, preds, res = fut.result()
                    except Exception as e:
                        tid = futs[fut]
                        task = tasks[tid]
                        preds = [Prediction(
                            attempt_1=self.fallback.best_guess(task, inp),
                            attempt_2=self.fallback.copy_input(inp))
                                 for inp in task.test_inputs]
                        res = BeamResult(tid, False, [], None, 0.0, str(e))
                    all_preds[tid] = preds; all_results[tid] = res
                    if progress_fn: progress_fn(1)
        else:
            for tid, task in tasks.items():
                try:
                    _, preds, res = self.solve_task(task)
                except Exception as e:
                    preds = [Prediction(
                        attempt_1=self.fallback.best_guess(task, inp),
                        attempt_2=self.fallback.copy_input(inp))
                             for inp in task.test_inputs]
                    res = BeamResult(tid, False, [], None, 0.0, str(e))
                all_preds[tid] = preds; all_results[tid] = res
                if progress_fn: progress_fn(1)

        return all_preds, all_results


_solver = ARCSolver()
print(f"✓ Phase 6 로드 완료 (ARCSolver  workers={MAX_WORKERS})")
'''

# ── Tests ──────────────────────────────────────────────────────────────────────
CELL_TEST1 = '''\
# ── Test 1: 데이터 로드 ──────────────────────────────────────────────────────────
loader      = ARCDataLoader()
train_tasks = loader.load_training()
eval_tasks  = loader.load_evaluation()
test_tasks  = loader.load_test()

print(f"training   : {len(train_tasks)} tasks")
print(f"evaluation : {len(eval_tasks)} tasks")
print(f"test       : {len(test_tasks)} tasks")
tid  = list(train_tasks.keys())[0]
task = train_tasks[tid]
print(f"[{tid}] train={len(task.train)}  test={len(task.test_inputs)}")
print("\\n✓ Test 1 통과")
'''

CELL_TEST2 = '''\
# ── Test 2: NumPy DSL 검증 ────────────────────────────────────────────────────
g = g2n([[1,2,3],[4,5,6],[7,8,9]])

r90 = np.rot90(g)
assert np.array_equal(np.rot90(r90), np.rot90(g,2))
print("✓ 회전")

assert np.array_equal(np.fliplr(g)[0], [3,2,1])
print("✓ 대칭")

g2 = g2n([[0,1,0],[1,0,1],[0,1,0]])
assert scale_up(g2, 2, 2).shape == (6, 6)
assert tile(g2, 2, 3).shape == (6, 9)
print("✓ 스케일/타일")

sp = g2n([[0,0,0],[0,5,0],[0,0,0]])
assert crop_to_content(sp).shape == (1, 1)
print("✓ crop_to_content")

grav = g2n([[1,0,2],[0,3,0],[0,0,4]])
gd = gravity_down(grav)
assert all(gd[-1, c] != 0 for c in range(3) if grav[:, c].any())
print("✓ gravity")

fg = g2n([[0,1,0],[1,1,1],[0,1,0]])
ff = flood_fill(fg, 0, 0, 9)
assert ff[0,0] == 9 and fg[1,1] == 1
print("✓ flood_fill")

objs = find_objects(g2n([[1,1,0,2],[1,0,0,2],[0,0,3,0]]))
assert len(objs) == 3
print(f"✓ find_objects ({len(objs)} objects)")

ng, inv = normalize_colors(g2n([[2,2,1],[2,1,1]]))
restored = denormalize_colors(ng, inv)
assert np.array_equal(restored, g2n([[2,2,1],[2,1,1]]))
print("✓ normalize/denormalize colors")

print("\\n✓ Test 2 통과")
'''

CELL_TEST3 = '''\
# ── Test 3: ShapePredictor ────────────────────────────────────────────────────
sp_pred = ShapePredictor()

# 고정 크기
t_fixed = Task("_", [Pair([[1,2],[3,4]], [[0,0,0],[0,0,0],[0,0,0]]),
                     Pair([[5],[6]],      [[0,0,0],[0,0,0],[0,0,0]])],
               [[[0,0]]])
assert sp_pred.predict(t_fixed, [[0,0]]) == (3, 3), sp_pred.analyze(t_fixed)
print("✓ fixed shape")

# identity (test input도 2×2여야 output 2×2 예측)
t_id = Task("_", [Pair([[1,2],[3,4]], [[1,2],[3,4]])], [[[0,0],[0,0]]])
assert sp_pred.predict(t_id, [[0,0],[0,0]]) == (2, 2)
print("✓ identity shape")

# 2x 배율
t_scale = Task("_", [Pair([[1,2],[3,4]], [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]),
                     Pair([[5,6,7],[8,9,0]], [[5,5,6,6,7,7]]*2+[[8,8,9,9,0,0]]*2)],
               [[[1,0]]])
rule = sp_pred.analyze(t_scale)
assert rule.kind == "scale", f"got {rule.kind}"
assert sp_pred.predict(t_scale, [[1,0,0]]) == (2, 6)
print("✓ scale shape")

print("\\n✓ Test 3 통과")
'''

CELL_TEST4 = '''\
# ── Test 4: Beam Search 동작 확인 ─────────────────────────────────────────────
# 단순 회전 Task: output = rot90(input)
_rotated_task = Task(
    task_id="_rot_test",
    train=[
        Pair(input=[[1,2,3],[4,5,6],[7,8,9]],
             output=[[7,4,1],[8,5,2],[9,6,3]]),
        Pair(input=[[0,1,0],[1,0,1],[0,1,0]],
             output=[[0,1,0],[1,0,1],[0,1,0]]),  # 대칭 → 여러 정답 가능
    ],
    test_inputs=[[[1,0,0],[0,1,0],[0,0,1]]]
)

_engine = BeamSearchEngine(beam_width=10, max_depth=2, timeout=10.0)
_res = _engine.solve(_rotated_task)
print(f"[rot test] success={_res.success}  score={_res.score:.3f}  seq={_res.seq1}")
assert _res.score > 0.5, f"Expected score > 0.5, got {_res.score}"
print("✓ 빔서치 동작 확인")

# 완전 일치 Task: scale_up×2
_scale_task = Task(
    task_id="_scale_test",
    train=[
        Pair(input=[[1,2],[3,4]],
             output=[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]),
        Pair(input=[[5,6],[7,8]],
             output=[[5,5,6,6],[5,5,6,6],[7,7,8,8],[7,7,8,8]]),
    ],
    test_inputs=[[[0,1],[2,3]]]
)

_res2 = _engine.solve(_scale_task)
print(f"[scale test] success={_res2.success}  score={_res2.score:.3f}  seq={_res2.seq1}")
assert _res2.success, f"Expected success, seq={_res2.seq1}"
print("✓ 스케일 Task 완전 풀이 성공")

print("\\n✓ Test 4 통과")
'''

CELL_TEST5 = '''\
# ── Test 5: E2E 파이프라인 (5개 Task 기준) ─────────────────────────────────────
try:
    from tqdm.notebook import tqdm as _tqdm
except ImportError:
    from tqdm import tqdm as _tqdm

_sample = dict(list(eval_tasks.items())[:5])
_solver_test = ARCSolver(beam_width=8, max_depth=3, timeout=15.0, max_workers=1)
_all_preds: dict[TaskId, list[Prediction]] = {}
_all_res:   dict[TaskId, BeamResult]       = {}

_solved = 0
for _tid, _task in _tqdm(_sample.items(), desc="E2E Test", unit="task"):
    _, _preds, _res = _solver_test.solve_task(_task)
    _all_preds[_tid] = _preds
    _all_res[_tid]   = _res
    if _res.success: _solved += 1

_scores = ARCEvaluator().evaluate(_all_preds, _sample)
print(f"\\n5-task 샘플  solved={_solved}/5  overall_score={_scores[\'overall_score\']:.4f}")

_path = ARCSubmissionWriter().save(_all_preds, "submission_test.json")
assert _path.exists()

for _tid, _r in _all_res.items():
    _st = "✓" if _r.success else f"~{_r.score:.2f}"
    print(f"  [{_st}] {_tid}  seq={_r.seq1[:3]}{'...' if len(_r.seq1)>3 else ''}")

print("\\n✓ Test 5 통과 — E2E 파이프라인 정상")
'''

CELL_CONFIG = '''\
# ═══════════════════════════════════════════════════════════════════
# ⚙️ 설정
# ═══════════════════════════════════════════════════════════════════
SPLIT              = "evaluation"   # "training" | "evaluation" | "test"
MAX_TASKS          = None           # None = 전체 / 정수 = 처음 N개
OUTPUT_FILE        = "submission.json"

RUN_BEAM_WIDTH     = 15
RUN_MAX_DEPTH      = 4
RUN_TASK_TIMEOUT   = 60.0
RUN_MAX_WORKERS    = max(1, os.cpu_count() or 1)

print(f"설정: split={SPLIT}  beam={RUN_BEAM_WIDTH}  depth={RUN_MAX_DEPTH}"
      f"  timeout={RUN_TASK_TIMEOUT}s  workers={RUN_MAX_WORKERS}")
'''

CELL_RUN = '''\
# ═══════════════════════════════════════════════════════════════════
# 🚀 메인 실행
# ═══════════════════════════════════════════════════════════════════
try:
    from tqdm.notebook import tqdm as _ntqdm
except ImportError:
    from tqdm import tqdm as _ntqdm

_loader = ARCDataLoader()
_tasks  = {"training": _loader.load_training,
           "evaluation": _loader.load_evaluation,
           "test": _loader.load_test}[SPLIT]()
if MAX_TASKS:
    _tasks = dict(list(_tasks.items())[:MAX_TASKS])
print(f"✓ {len(_tasks)} tasks 로드 ({SPLIT})")

_run_solver = ARCSolver(
    beam_width=RUN_BEAM_WIDTH,
    max_depth=RUN_MAX_DEPTH,
    timeout=RUN_TASK_TIMEOUT,
    max_workers=RUN_MAX_WORKERS,
)

_all_preds: dict[TaskId, list[Prediction]] = {}
_all_res:   dict[TaskId, BeamResult]       = {}
_total_start = time.perf_counter()

with _ntqdm(total=len(_tasks), desc="Beam Search", unit="task") as _pbar:
    def _update(n): _pbar.update(n)
    _all_preds, _all_res = _run_solver.solve_all(_tasks, progress_fn=_update)

_elapsed = time.perf_counter() - _total_start

# 평가
_eval_scores = None
if SPLIT in ("training", "evaluation"):
    _eval_scores = ARCEvaluator().evaluate(_all_preds, _tasks)

# 저장
_saved = ARCSubmissionWriter().save(_all_preds, OUTPUT_FILE)

# ── 요약 ─────────────────────────────────────────────────────────────────────
_tot   = len(_all_res)
_solved   = sum(1 for r in _all_res.values() if r.success)
_partial  = sum(1 for r in _all_res.values() if not r.success and r.score > 0.5)
_failed   = _tot - _solved - _partial

print()
print("=" * 64)
print(" ARC-AGI-2 Beam Search 실행 요약")
print("=" * 64)
print(f"  총 tasks      : {_tot}")
print(f"  처리 시간     : {_elapsed:.1f}s  ({_elapsed/_tot:.1f}s/task)")
print()
print(f"  ✓  완전 풀이  : {_solved:4d}  ({_solved/_tot*100:.1f}%)")
print(f"  ~  부분 일치  : {_partial:4d}  ({_partial/_tot*100:.1f}%)")
print(f"  ✗  폴백       : {_failed:4d}  ({_failed/_tot*100:.1f}%)")

if _eval_scores:
    _pf = sum(1 for s in _eval_scores["task_scores"].values() if s == 1.0)
    print()
    print(f"  평가 점수(overall): {_eval_scores[\'overall_score\']:.4f}")
    print(f"  완벽 task         : {_pf}/{_tot}")

print("=" * 64)
print(f"\\n제출 파일: {_saved}")
'''

# ══════════════════════════════════════════════════════════════════════════════
# 노트북 조립
# ══════════════════════════════════════════════════════════════════════════════
cells = [
    md_cell(MD_TITLE),
    md_cell("## 0. 패키지 설치"),
    code_cell(CELL_INSTALL),
    md_cell("## 1. 임포트 및 전역 설정"),
    code_cell(CELL_IMPORTS),
    md_cell("## Phase 1: 데이터 파이프라인"),
    code_cell(CELL_PHASE1),
    md_cell("## Phase 2: NumPy DSL 프리미티브"),
    code_cell(CELL_PHASE2),
    md_cell("## Phase 3: 출력 크기 예측기 (ShapePredictor)"),
    code_cell(CELL_PHASE3),
    md_cell("## Phase 4: 연산자 라이브러리"),
    code_cell(CELL_PHASE4),
    md_cell("## Phase 5: 휴리스틱 빔서치 엔진"),
    code_cell(CELL_PHASE5),
    md_cell("## Phase 6: ARC Solver + 병렬 파이프라인"),
    code_cell(CELL_PHASE6),
    md_cell("## 🧪 자체 테스트"),
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
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
    },
    "cells": cells,
}

out = Path("arc_agi2_beam.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"✓ 노트북 생성: {out}  ({out.stat().st_size // 1024} KB)")
