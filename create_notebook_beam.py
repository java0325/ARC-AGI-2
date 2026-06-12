#!/usr/bin/env python3
"""arc_agi2_beam.ipynb 생성 스크립트 v2 — 확장 DSL + 2단계 빔서치."""

import json, uuid
from pathlib import Path

def code_cell(source):
    return {"cell_type":"code","execution_count":None,
            "id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":source}

def md_cell(source):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":source}

# ══════════════════════════════════════════════════════════════════════════════
MD_TITLE = """\
# ARC-AGI-2 — Heuristic Beam Search v2 (LLM-Free)
> **확장 DSL + 2단계 빔서치 + 학습 기반 연산자**
>
> | 모듈 | 내용 |
> |---|---|
> | Phase 1 | 데이터 파이프라인 |
> | Phase 2 | NumPy DSL 기본 프리미티브 |
> | Phase 2b | 확장 DSL (오브젝트·대칭·패턴·크롭·합성) |
> | Phase 3 | 출력 크기 예측기 |
> | Phase 4 | 연산자 라이브러리 (고정 + 학습 기반) |
> | Phase 5 | 2단계 빔서치 (글로벌 → 파인튜닝) |
> | Phase 6 | 병렬 ARC Solver + Validation |"""

CELL_INSTALL = """\
import subprocess, sys
for _p in ["numpy", "tqdm", "scipy"]:
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
BEAM_WIDTH     = 20
MAX_DEPTH      = 4
TASK_TIMEOUT   = 90.0
MAX_WORKERS    = max(1, os.cpu_count() or 1)

Grid   = list[list[int]]
NGrid  = np.ndarray
TaskId = str

print(f"✓ 임포트 완료  DATA_DIR={DATA_DIR.resolve()}  cores={MAX_WORKERS}")"""

# ── Phase 1 ────────────────────────────────────────────────────────────────────
CELL_PHASE1 = '''\
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

# ── Phase 2: 기본 DSL ──────────────────────────────────────────────────────────
CELL_PHASE2 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 2: NumPy DSL 기본 프리미티브
# ═══════════════════════════════════════════════════════════════════

def g2n(grid) -> NGrid:
    return np.array(grid, dtype=np.int8)

def n2g(arr: NGrid) -> Grid:
    return arr.tolist()

# ── 8방향 대칭 ──────────────────────────────────────────────────────────────────
def sym_all(g: NGrid) -> list[tuple[str, NGrid]]:
    variants = []
    for k in range(4):
        r = np.rot90(g, k)
        variants.append((f"rot{k*90}", r))
        variants.append((f"rot{k*90}_fliplr", np.fliplr(r)))
    return variants

# ── 스케일/타일 ─────────────────────────────────────────────────────────────────
def scale_up(g: NGrid, ry: int, rx: int) -> NGrid:
    return np.repeat(np.repeat(g, ry, axis=0), rx, axis=1)

def scale_down(g: NGrid, ry: int, rx: int) -> NGrid:
    return g[::ry, ::rx]

def tile(g: NGrid, ny: int, nx: int) -> NGrid:
    return np.tile(g, (ny, nx))

def crop_to_content(g: NGrid, bg: int = 0) -> NGrid:
    rows = np.any(g != bg, axis=1)
    cols = np.any(g != bg, axis=0)
    if not rows.any() or not cols.any():
        return g.copy()
    return g[np.ix_(rows, cols)]

def pad_to_shape(g: NGrid, H: int, W: int, fill: int = 0) -> NGrid:
    h, w = g.shape
    result = np.full((H, W), fill, dtype=np.int8)
    r0 = max(0, (H-h)//2); c0 = max(0, (W-w)//2)
    he = min(H, r0+h); we = min(W, c0+w)
    result[r0:he, c0:we] = g[:he-r0, :we-c0]
    return result

def extend_to_shape(g: NGrid, H: int, W: int) -> NGrid:
    h, w = g.shape
    if h==0 or w==0: return np.zeros((H,W), dtype=np.int8)
    ny = -(-H//h); nx = -(-W//w)
    return np.tile(g, (ny, nx))[:H, :W]

# ── 중력 ────────────────────────────────────────────────────────────────────────
def gravity_down(g: NGrid, bg: int = 0) -> NGrid:
    result = np.full_like(g, bg)
    for c in range(g.shape[1]):
        nz = g[:, c][g[:, c] != bg]
        result[g.shape[0]-len(nz):, c] = nz
    return result

def gravity_up(g: NGrid, bg: int = 0) -> NGrid:
    result = np.full_like(g, bg)
    for c in range(g.shape[1]):
        nz = g[:, c][g[:, c] != bg]
        result[:len(nz), c] = nz
    return result

def gravity_left(g: NGrid, bg: int = 0) -> NGrid:
    result = np.full_like(g, bg)
    for r in range(g.shape[0]):
        nz = g[r][g[r] != bg]
        result[r, :len(nz)] = nz
    return result

def gravity_right(g: NGrid, bg: int = 0) -> NGrid:
    result = np.full_like(g, bg)
    for r in range(g.shape[0]):
        nz = g[r][g[r] != bg]
        result[r, g.shape[1]-len(nz):] = nz
    return result

# ── 색상 연산 ───────────────────────────────────────────────────────────────────
def recolor(g: NGrid, mapping: dict[int,int]) -> NGrid:
    result = g.copy()
    for src, dst in mapping.items():
        result[g == src] = dst
    return result

def swap_colors(g: NGrid, c1: int, c2: int) -> NGrid:
    return recolor(g, {c1:c2, c2:c1})

def replace_color(g: NGrid, src: int, dst: int) -> NGrid:
    r = g.copy(); r[g==src] = dst; return r

def normalize_colors(g: NGrid) -> tuple[NGrid, dict[int,int]]:
    flat = g.flatten()
    counts = Counter(flat.tolist())
    sorted_c = [c for c,_ in sorted(counts.items(), key=lambda x:-x[1])]
    fwd = {c: i for i,c in enumerate(sorted_c)}
    inv = {i: c for c,i in fwd.items()}
    return np.vectorize(fwd.get)(g).astype(np.int8), inv

def denormalize_colors(g: NGrid, inv: dict[int,int]) -> NGrid:
    return np.vectorize(lambda x: inv.get(x, x))(g).astype(np.int8)

# ── 오브젝트 탐지 ───────────────────────────────────────────────────────────────
def find_objects(g: NGrid, bg: int = 0, connectivity: int = 4) -> list[np.ndarray]:
    H, W = g.shape
    visited = np.zeros((H,W), dtype=bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    objects = []
    for r in range(H):
        for c in range(W):
            if visited[r,c] or g[r,c]==bg: continue
            cells = []
            queue = deque([(r,c)]); visited[r,c]=True
            while queue:
                cr,cc = queue.popleft(); cells.append((cr,cc))
                for dr,dc in dirs:
                    nr,nc = cr+dr,cc+dc
                    if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and g[nr,nc]==g[r,c]:
                        visited[nr,nc]=True; queue.append((nr,nc))
            objects.append(np.array(cells))
    return objects

def extract_object(g: NGrid, cells: np.ndarray, bg: int = 0) -> NGrid:
    r0,c0 = cells.min(axis=0); r1,c1 = cells.max(axis=0)
    sub = np.full((r1-r0+1, c1-c0+1), bg, dtype=np.int8)
    for r,c in cells:
        sub[r-r0, c-c0] = g[r,c]
    return sub

def flood_fill(g: NGrid, r0: int, c0: int, new_color: int,
               connectivity: int = 4) -> NGrid:
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
    H, W = g.shape
    result = g.copy()
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    visited = np.zeros((H,W), dtype=bool)
    queue = deque()
    for r in range(H):
        for c in [0,W-1]:
            if result[r,c]==bg and not visited[r,c]:
                visited[r,c]=True; queue.append((r,c))
    for c in range(W):
        for r in [0,H-1]:
            if result[r,c]==bg and not visited[r,c]:
                visited[r,c]=True; queue.append((r,c))
    while queue:
        r,c = queue.popleft(); result[r,c]=fill_color
        for dr,dc in dirs:
            nr,nc = r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and result[nr,nc]==bg:
                visited[nr,nc]=True; queue.append((nr,nc))
    return result

def outline_objects(g: NGrid, bg: int = 0) -> NGrid:
    result = g.copy(); H,W = g.shape
    for r in range(1,H-1):
        for c in range(1,W-1):
            if g[r,c]!=bg:
                if all(g[r+dr,c+dc]==g[r,c] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]):
                    result[r,c]=bg
    return result


print("✓ Phase 2 로드 완료 (기본 DSL)")
'''

# ── Phase 2b: 확장 DSL ──────────────────────────────────────────────────────────
CELL_PHASE2B = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 2b: 확장 DSL — 오브젝트·대칭·패턴·합성 연산자
# ═══════════════════════════════════════════════════════════════════

# ── 오브젝트 선택 ───────────────────────────────────────────────────────────────

def keep_n_largest(g: NGrid, n: int = 1, bg: int = 0) -> NGrid:
    """크기 상위 n개 오브젝트만 유지."""
    objs = sorted(find_objects(g, bg, 4), key=len, reverse=True)[:n]
    result = np.full_like(g, bg)
    for obj in objs:
        for r,c in obj:
            result[r,c] = g[r,c]
    return result

def keep_n_smallest(g: NGrid, n: int = 1, bg: int = 0) -> NGrid:
    """크기 하위 n개 오브젝트만 유지."""
    objs = sorted(find_objects(g, bg, 4), key=len)[:n]
    result = np.full_like(g, bg)
    for obj in objs:
        for r,c in obj:
            result[r,c] = g[r,c]
    return result

def remove_small_objects(g: NGrid, max_size: int = 1, bg: int = 0) -> NGrid:
    """크기 max_size 이하 오브젝트 제거."""
    result = g.copy()
    for obj in find_objects(g, bg, 4):
        if len(obj) <= max_size:
            for r,c in obj:
                result[r,c] = bg
    return result

def color_objects_by_size(g: NGrid, bg: int = 0) -> NGrid:
    """각 오브젝트를 크기 순위(1=최대)로 재색칠."""
    objs = sorted(find_objects(g, bg, 4), key=len, reverse=True)
    result = np.full_like(g, bg)
    for i, obj in enumerate(objs, 1):
        color = min(i, 9)
        for r,c in obj:
            result[r,c] = color
    return result

def mark_unique_objects(g: NGrid, bg: int = 0) -> NGrid:
    """각 오브젝트를 순번(1,2,3…)으로 레이블링."""
    objs = find_objects(g, bg, 4)
    result = np.full_like(g, bg)
    for i, obj in enumerate(objs, 1):
        for r,c in obj:
            result[r,c] = i % 10
    return result

def fill_holes(g: NGrid, fill_color: int = -1, bg: int = 0) -> NGrid:
    """오브젝트 내부의 폐쇄 영역(구멍)을 채운다."""
    # 외부 bg 마킹 → 채워지지 않은 bg = 구멍
    H, W = g.shape
    exterior = np.zeros((H,W), dtype=bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    queue = deque()
    for r in range(H):
        for c in [0,W-1]:
            if g[r,c]==bg and not exterior[r,c]:
                exterior[r,c]=True; queue.append((r,c))
    for c in range(W):
        for r in [0,H-1]:
            if g[r,c]==bg and not exterior[r,c]:
                exterior[r,c]=True; queue.append((r,c))
    while queue:
        r,c = queue.popleft()
        for dr,dc in dirs:
            nr,nc = r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and not exterior[nr,nc] and g[nr,nc]==bg:
                exterior[nr,nc]=True; queue.append((nr,nc))

    result = g.copy()
    hole_mask = (~exterior) & (g == bg)
    if not hole_mask.any(): return result

    if fill_color == -1:
        # 구멍 주변 non-bg 색상 중 최빈값으로 채움
        neighbor_colors = []
        hole_positions = np.argwhere(hole_mask)
        for r,c in hole_positions:
            for dr,dc in dirs:
                nr,nc = r+dr,c+dc
                if 0<=nr<H and 0<=nc<W and g[nr,nc]!=bg:
                    neighbor_colors.append(int(g[nr,nc]))
        fc = Counter(neighbor_colors).most_common(1)[0][0] if neighbor_colors else 1
    else:
        fc = fill_color
    result[hole_mask] = fc
    return result

def center_content(g: NGrid, bg: int = 0) -> NGrid:
    """비배경 내용을 격자 중앙에 배치."""
    rows = np.any(g!=bg, axis=1); cols = np.any(g!=bg, axis=0)
    if not rows.any() or not cols.any(): return g.copy()
    cropped = g[np.ix_(rows, cols)]
    result = np.full_like(g, bg)
    r0 = (g.shape[0]-cropped.shape[0])//2
    c0 = (g.shape[1]-cropped.shape[1])//2
    result[r0:r0+cropped.shape[0], c0:c0+cropped.shape[1]] = cropped
    return result

# ── 대칭 완성 ───────────────────────────────────────────────────────────────────

def complete_symmetry_h(g: NGrid, bg: int = 0) -> NGrid:
    """좌-우 대칭으로 빈 셀을 채운다."""
    H, W = g.shape
    result = g.copy()
    for r in range(H):
        for c in range(W//2):
            l, rc = g[r,c], g[r,W-1-c]
            if l != bg and rc == bg: result[r,W-1-c] = l
            elif rc != bg and l == bg: result[r,c] = rc
    return result

def complete_symmetry_v(g: NGrid, bg: int = 0) -> NGrid:
    """상-하 대칭으로 빈 셀을 채운다."""
    H, W = g.shape
    result = g.copy()
    for r in range(H//2):
        for c in range(W):
            t, b = g[r,c], g[H-1-r,c]
            if t != bg and b == bg: result[H-1-r,c] = t
            elif b != bg and t == bg: result[r,c] = b
    return result

def complete_symmetry_rot180(g: NGrid, bg: int = 0) -> NGrid:
    """180도 회전 대칭으로 빈 셀을 채운다."""
    H, W = g.shape
    result = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r,c] == bg and g[H-1-r,W-1-c] != bg:
                result[r,c] = g[H-1-r,W-1-c]
    return result

# ── 미러 확장 ───────────────────────────────────────────────────────────────────

def mirror_extend_right(g: NGrid) -> NGrid:
    return np.hstack([g, np.fliplr(g)])

def mirror_extend_left(g: NGrid) -> NGrid:
    return np.hstack([np.fliplr(g), g])

def mirror_extend_down(g: NGrid) -> NGrid:
    return np.vstack([g, np.flipud(g)])

def mirror_extend_up(g: NGrid) -> NGrid:
    return np.vstack([np.flipud(g), g])

def tile_from_content(g: NGrid, bg: int = 0) -> NGrid:
    """비배경 영역을 잘라 원본 크기로 타일링."""
    cropped = crop_to_content(g, bg)
    if cropped.size == 0: return g.copy()
    return extend_to_shape(cropped, g.shape[0], g.shape[1])

# ── 테두리 ──────────────────────────────────────────────────────────────────────

def add_border(g: NGrid, color: int | None = None, bg: int = 0) -> NGrid:
    """격자 주위에 1픽셀 테두리 추가."""
    if color is None:
        cnt = Counter(g[g!=bg].tolist())
        color = cnt.most_common(1)[0][0] if cnt else 1
    H, W = g.shape
    result = np.full((H+2, W+2), color, dtype=np.int8)
    result[1:-1, 1:-1] = g
    return result

def remove_border(g: NGrid) -> NGrid:
    """1픽셀 테두리 제거."""
    if g.shape[0]<=2 or g.shape[1]<=2: return g.copy()
    return g[1:-1, 1:-1].copy()

# ── 색상 고급 연산 ──────────────────────────────────────────────────────────────

def sort_colors_by_frequency(g: NGrid, bg: int = 0) -> NGrid:
    """각 비배경 색을 빈도 순위(1=가장 많음)로 교체."""
    cnt = Counter(g[g!=bg].tolist())
    mapping = {c: i+1 for i,(c,_) in enumerate(cnt.most_common())}
    mapping[bg] = bg
    return np.vectorize(lambda x: mapping.get(x, x))(g).astype(np.int8)

def keep_most_common_color(g: NGrid, bg: int = 0) -> NGrid:
    """배경 제외 가장 많은 색만 유지."""
    cnt = Counter(g[g!=bg].tolist())
    if not cnt: return g.copy()
    mc = cnt.most_common(1)[0][0]
    return replace_color(np.where(g==mc, g, bg).astype(np.int8), bg, bg)

def keep_least_common_color(g: NGrid, bg: int = 0) -> NGrid:
    """배경 제외 가장 적은 색만 유지."""
    cnt = Counter(g[g!=bg].tolist())
    if not cnt: return g.copy()
    lc = cnt.most_common()[-1][0]
    return replace_color(np.where(g==lc, g, bg).astype(np.int8), bg, bg)

def invert_binary(g: NGrid, bg: int = 0) -> NGrid:
    """비배경 색을 배경으로, 배경을 비배경으로 반전."""
    cnt = Counter(g[g!=bg].tolist())
    if not cnt: return g.copy()
    fg = cnt.most_common(1)[0][0]
    result = g.copy()
    result[g==bg] = fg
    result[g!=bg] = bg
    return result

# ── 격자 분할 ───────────────────────────────────────────────────────────────────

def split_by_separator(g: NGrid, bg: int = 0) -> list[NGrid] | None:
    """균일한 행/열(구분선)으로 분할된 서브그리드 목록 반환."""
    H, W = g.shape
    sep_rows = [r for r in range(H) if len(set(g[r].tolist()))==1]
    sep_cols = [c for c in range(W) if len(set(g[:,c].tolist()))==1]
    if not sep_rows and not sep_cols: return None

    row_cuts = sorted(set([-1] + sep_rows + [H]))
    col_cuts = sorted(set([-1] + sep_cols + [W]))
    sub = []
    for i in range(len(row_cuts)-1):
        r0 = row_cuts[i]+1; r1 = row_cuts[i+1]
        if r0 >= r1: continue
        for j in range(len(col_cuts)-1):
            c0 = col_cuts[j]+1; c1 = col_cuts[j+1]
            if c0 >= c1: continue
            sub.append(g[r0:r1, c0:c1].copy())
    return sub if sub else None

def apply_op_to_subgrids(g: NGrid, op: Callable, bg: int = 0) -> NGrid:
    """구분선으로 분할 후 각 서브그리드에 op 적용, 재조합."""
    H, W = g.shape
    sep_rows = [r for r in range(H) if len(set(g[r].tolist()))==1]
    sep_cols = [c for c in range(W) if len(set(g[:,c].tolist()))==1]
    if not sep_rows and not sep_cols:
        return op(g)  # 분할 없으면 전체 적용

    row_cuts = sorted(set([-1]+sep_rows+[H]))
    col_cuts = sorted(set([-1]+sep_cols+[W]))

    # 격자를 재구성
    result = g.copy()
    for i in range(len(row_cuts)-1):
        r0 = row_cuts[i]+1; r1 = row_cuts[i+1]
        if r0 >= r1: continue
        for j in range(len(col_cuts)-1):
            c0 = col_cuts[j]+1; c1 = col_cuts[j+1]
            if c0 >= c1: continue
            try:
                sub = g[r0:r1, c0:c1].copy()
                transformed = op(sub)
                if transformed.shape == sub.shape:
                    result[r0:r1, c0:c1] = transformed
            except Exception:
                pass
    return result

# ── 패턴 분석 헬퍼 ──────────────────────────────────────────────────────────────

def find_periodic_period(g: NGrid) -> tuple[int,int] | None:
    """격자가 주기적 패턴을 가지면 (row_period, col_period) 반환."""
    H, W = g.shape
    for rp in range(1, H//2+1):
        if H % rp == 0:
            if all(np.array_equal(g[r], g[r%rp]) for r in range(H)):
                for cp in range(1, W//2+1):
                    if W % cp == 0:
                        if all(np.array_equal(g[:,c], g[:,c%cp]) for c in range(W)):
                            return (rp, cp)
    return None

def extract_pattern(g: NGrid) -> NGrid:
    """주기적 패턴의 최소 단위를 추출."""
    p = find_periodic_period(g)
    if p: return g[:p[0], :p[1]].copy()
    return g.copy()

# ── 행/열 정렬 ──────────────────────────────────────────────────────────────────

def sort_rows_by_sum(g: NGrid) -> NGrid:
    """행을 합계 오름차순으로 정렬."""
    idx = np.argsort(g.sum(axis=1))
    return g[idx]

def sort_cols_by_sum(g: NGrid) -> NGrid:
    idx = np.argsort(g.sum(axis=0))
    return g[:, idx]

def sort_rows_descending(g: NGrid) -> NGrid:
    idx = np.argsort(-g.sum(axis=1))
    return g[idx]


print("✓ Phase 2b 로드 완료 (확장 DSL — 오브젝트·대칭·패턴 연산자)")
'''

# ── Phase 3: Shape Predictor ───────────────────────────────────────────────────
CELL_PHASE3 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 3: 출력 크기 예측기
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ShapeRule:
    kind: str
    fixed: tuple[int,int] | None
    scale: tuple[Fraction,Fraction] | None


class ShapePredictor:
    def analyze(self, task: Task) -> ShapeRule:
        pairs = [(p.input, p.output) for p in task.train if p.output is not None]
        if not pairs: return ShapeRule("unknown", None, None)

        in_shapes  = [(len(i),   len(i[0])   if i   else 0) for i,_ in pairs]
        out_shapes = [(len(o),   len(o[0])   if o   else 0) for _,o in pairs]

        if len(set(out_shapes)) == 1 and len(set(in_shapes)) > 1:
            return ShapeRule("fixed", out_shapes[0], None)
        if all(i==o for i,o in zip(in_shapes, out_shapes)):
            return ShapeRule("identity", None, None)
        row_scales = [Fraction(o[0], i[0]) for i,o in zip(in_shapes, out_shapes) if i[0]>0]
        col_scales = [Fraction(o[1], i[1]) for i,o in zip(in_shapes, out_shapes) if i[1]>0]
        if len(set(row_scales))==1 and len(set(col_scales))==1:
            return ShapeRule("scale", None, (row_scales[0], col_scales[0]))
        if all(i[0]==o[1] and i[1]==o[0] for i,o in zip(in_shapes, out_shapes)):
            return ShapeRule("transpose", None, None)
        if all(o[0]<=i[0] and o[1]<=i[1] for i,o in zip(in_shapes, out_shapes)):
            if len(set(out_shapes))==1:
                return ShapeRule("fixed", out_shapes[0], None)
        mode = Counter(out_shapes).most_common(1)[0][0]
        return ShapeRule("fallback_mode", mode, None)

    def predict(self, task: Task, test_input: Grid) -> tuple[int,int] | None:
        rule = self.analyze(task)
        H, W = len(test_input), len(test_input[0]) if test_input else 1
        if rule.kind == "identity":      return (H, W)
        if rule.kind in ("fixed","fallback_mode"): return rule.fixed
        if rule.kind == "transpose":     return (W, H)
        if rule.kind == "scale" and rule.scale:
            rs, cs = rule.scale
            return (int(H*rs), int(W*cs))
        return None


_shape_predictor = ShapePredictor()
print("✓ Phase 3 로드 완료 (ShapePredictor)")
'''

# ── Phase 4: Operation Library ─────────────────────────────────────────────────
CELL_PHASE4 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 4: 연산자 라이브러리 — 고정 + 학습 기반
# ═══════════════════════════════════════════════════════════════════

FIXED_OPS: list[tuple[str, Callable]] = [
    # 8방향 대칭
    ("rot0",     lambda g: g.copy()),
    ("rot90",    lambda g: np.rot90(g,1).copy()),
    ("rot180",   lambda g: np.rot90(g,2).copy()),
    ("rot270",   lambda g: np.rot90(g,3).copy()),
    ("flip_lr",  lambda g: np.fliplr(g).copy()),
    ("flip_ud",  lambda g: np.flipud(g).copy()),
    ("flip_d0",  lambda g: g.T.copy()),
    ("flip_d1",  lambda g: np.rot90(g.T,2).copy()),
    # 스케일
    ("scale2x2", lambda g: scale_up(g,2,2)),
    ("scale3x3", lambda g: scale_up(g,3,3)),
    ("scale2x1", lambda g: scale_up(g,2,1)),
    ("scale1x2", lambda g: scale_up(g,1,2)),
    ("scale3x1", lambda g: scale_up(g,3,1)),
    ("scale1x3", lambda g: scale_up(g,1,3)),
    # 타일
    ("tile22",   lambda g: tile(g,2,2)),
    ("tile12",   lambda g: tile(g,1,2)),
    ("tile21",   lambda g: tile(g,2,1)),
    ("tile13",   lambda g: tile(g,1,3)),
    ("tile31",   lambda g: tile(g,3,1)),
    ("tile33",   lambda g: tile(g,3,3)),
    # 크롭
    ("crop_bg0", lambda g: crop_to_content(g,0)),
    # 중력
    ("grav_d",   lambda g: gravity_down(g,0)),
    ("grav_u",   lambda g: gravity_up(g,0)),
    ("grav_l",   lambda g: gravity_left(g,0)),
    ("grav_r",   lambda g: gravity_right(g,0)),
    # 색상
    ("norm_col", lambda g: normalize_colors(g)[0]),
    ("sort_col", lambda g: sort_colors_by_frequency(g,0)),
    ("keep_mc",  lambda g: keep_most_common_color(g,0)),
    ("keep_lc",  lambda g: keep_least_common_color(g,0)),
    ("invert01", lambda g: invert_binary(g,0)),
    # 오브젝트
    ("keep_lg1", lambda g: keep_n_largest(g,1,0)),
    ("keep_lg2", lambda g: keep_n_largest(g,2,0)),
    ("keep_sm1", lambda g: keep_n_smallest(g,1,0)),
    ("keep_sm2", lambda g: keep_n_smallest(g,2,0)),
    ("rm_noise1",lambda g: remove_small_objects(g,1,0)),
    ("rm_noise2",lambda g: remove_small_objects(g,2,0)),
    ("col_size", lambda g: color_objects_by_size(g,0)),
    ("mark_obj", lambda g: mark_unique_objects(g,0)),
    ("fill_hole",lambda g: fill_holes(g,-1,0)),
    ("center",   lambda g: center_content(g,0)),
    ("outline",  lambda g: outline_objects(g,0)),
    ("fill_out", lambda g: fill_outer_region(g,0,0)),
    # 대칭 완성
    ("sym_h",    lambda g: complete_symmetry_h(g,0)),
    ("sym_v",    lambda g: complete_symmetry_v(g,0)),
    ("sym_r180", lambda g: complete_symmetry_rot180(g,0)),
    # 미러 확장
    ("mir_r",    lambda g: mirror_extend_right(g)),
    ("mir_l",    lambda g: mirror_extend_left(g)),
    ("mir_d",    lambda g: mirror_extend_down(g)),
    ("mir_u",    lambda g: mirror_extend_up(g)),
    # 테두리
    ("add_bdr",  lambda g: add_border(g,None,0)),
    ("rm_bdr",   lambda g: remove_border(g)),
    # 패턴
    ("tile_cnt", lambda g: tile_from_content(g,0)),
    ("ext_pat",  lambda g: extract_pattern(g)),
    # 행/열 정렬
    ("sort_r",   lambda g: sort_rows_by_sum(g)),
    ("sort_c",   lambda g: sort_cols_by_sum(g)),
    ("sort_rd",  lambda g: sort_rows_descending(g)),
    # 서브그리드 각 연산 적용
    ("sub_rot90",  lambda g: apply_op_to_subgrids(g, lambda x: np.rot90(x,1))),
    ("sub_flipl",  lambda g: apply_op_to_subgrids(g, lambda x: np.fliplr(x))),
    ("sub_norm",   lambda g: apply_op_to_subgrids(g, lambda x: normalize_colors(x)[0])),
]


# ── 학습 기반 연산자 생성 ─────────────────────────────────────────────────────────

def _learn_exact_color_map(task: Task) -> dict[int,int] | None:
    """Train 쌍 전체에서 일관된 색상 매핑 학습."""
    all_maps: list[dict] = []
    for p in task.train:
        if p.output is None: continue
        inp, out = g2n(p.input), g2n(p.output)
        if inp.shape != out.shape: return None  # 크기 다르면 불가
        cmap: dict[int,int] = {}
        for c in np.unique(inp):
            out_pixels = out[inp == c]
            uniq = np.unique(out_pixels)
            if len(uniq) == 1:
                cmap[int(c)] = int(uniq[0])
            else:
                return None  # 비결정적 매핑
        all_maps.append(cmap)
    if not all_maps: return None
    first = all_maps[0]
    if not all(m == first for m in all_maps): return None
    if all(k == v for k, v in first.items()): return None  # identity 제외
    return first


def _check_direct_fixed_op(task: Task) -> list[tuple[str, Callable]]:
    """고정 연산자 중 Train을 100% 통과하는 것을 즉시 찾는다."""
    train_in  = [g2n(p.input)  for p in task.train]
    train_out = [g2n(p.output) for p in task.train if p.output is not None]
    if len(train_in) != len(train_out): return []

    found = []
    for name, fn in FIXED_OPS:
        if name == "rot0": continue
        try:
            preds = [fn(g) for g in train_in]
            if all(np.array_equal(p, t) for p,t in zip(preds, train_out)):
                found.append((f"direct_{name}", fn))
        except Exception:
            pass
    return found


def _learn_scale_factors(task: Task) -> list[tuple[str, Callable]]:
    """Train에서 스케일 배율(정수 배율)을 학습."""
    pairs = [(g2n(p.input), g2n(p.output)) for p in task.train if p.output is not None]
    if not pairs: return []
    ops = []
    in_h, in_w = pairs[0][0].shape
    out_h, out_w = pairs[0][1].shape
    if in_h == 0 or in_w == 0: return []
    if out_h % in_h == 0 and out_w % in_w == 0:
        ry, rx = out_h // in_h, out_w // in_w
        if ry <= 5 and rx <= 5 and (ry, rx) != (1, 1):
            name = f"scale_{ry}x{rx}"
            ops.append((name, lambda g, r=ry, c=rx: scale_up(g, r, c)))
    return ops


def _learn_output_size_ops(task: Task) -> list[tuple[str, Callable]]:
    """ShapePredictor 예측 크기로 확장/크롭 연산자 생성."""
    if not task.test_inputs: return []
    pred_shape = _shape_predictor.predict(task, task.test_inputs[0])
    if not pred_shape: return []
    H, W = pred_shape
    ops = []
    for p in task.train:
        ih, iw = len(p.input), len(p.input[0]) if p.input else 1
        if (ih, iw) == (H, W): return []  # identity → skip
    ops.append((f"ext_{H}x{W}", lambda g, h=H, w=W: extend_to_shape(g, h, w)))
    ops.append((f"pad_{H}x{W}", lambda g, h=H, w=W: pad_to_shape(g, h, w)))
    # crop to predicted size (top-left corner)
    ops.append((f"crop_{H}x{W}", lambda g, h=H, w=W: g[:h, :w].copy()
                                 if g.shape[0]>=h and g.shape[1]>=w else g.copy()))
    return ops


def build_task_ops(task: Task) -> list[tuple[str, Callable]]:
    """전체 연산자 목록 구성: 고정 + Task 적응형 + 학습 기반."""
    ops = list(FIXED_OPS)

    # 1. 직접 해: 고정 연산자 중 100% 통과하는 것 (앞에 배치)
    direct = _check_direct_fixed_op(task)
    if direct:
        ops = direct + ops

    # 2. 학습된 색상 매핑
    cmap = _learn_exact_color_map(task)
    if cmap:
        ops.insert(0, ("learned_cmap", lambda g, m=cmap: recolor(g, m)))

    # 3. 학습된 스케일
    scale_ops = _learn_scale_factors(task)
    for op in scale_ops:
        ops.insert(0, op)

    # 4. 출력 크기 예측 기반 ops
    ops.extend(_learn_output_size_ops(task))

    # 5. 등장 색상 기반 swap 연산자
    color_set: set[int] = set()
    for p in task.train:
        for row in p.input: color_set.update(row)
        if p.output:
            for row in p.output: color_set.update(row)
    colors = sorted(color_set)
    for i, c1 in enumerate(colors):
        for c2 in colors[i+1:]:
            ops.append((f"swap_{c1}_{c2}",
                        (lambda a,b: (lambda g: swap_colors(g,a,b)))(c1,c2)))
    for c in colors:
        if c == 0: continue
        ops.append((f"keep_{c}",
                    (lambda a: (lambda g: np.where(g==a, g, np.int8(0)).astype(np.int8)))(c)))
        ops.append((f"rm_{c}",
                    (lambda a: (lambda g: replace_color(g, a, 0)))(c)))
        ops.append((f"rebg_{c}",
                    (lambda b: (lambda g: swap_colors(g, 0, b)))(c)))

    return ops


_OPS_CACHE: dict[str, list] = {}   # task_id → ops list


print(f"✓ Phase 4 로드 완료 (고정 {len(FIXED_OPS)}개 + Task 적응형 연산자)")
'''

# ── Phase 5: 2단계 Beam Search ─────────────────────────────────────────────────
CELL_PHASE5 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 5: 2단계 빔서치 엔진
# ═══════════════════════════════════════════════════════════════════

def pixel_score(pred: NGrid, target: NGrid) -> float:
    if pred.shape != target.shape: return 0.0
    return float(np.mean(pred == target))

def score_all(preds: list[NGrid], targets: list[NGrid]) -> float:
    if not preds or not targets: return 0.0
    scores = [pixel_score(p, t) for p, t in zip(preds, targets)]
    return float(np.mean(scores))

def shape_score(pred: NGrid, target: NGrid) -> float:
    """크기 일치 여부를 추가 가산점으로 반환 (0 or 0.3)."""
    return 0.3 if pred.shape == target.shape else 0.0

def combined_score(preds: list[NGrid], targets: list[NGrid]) -> float:
    """픽셀 점수 + 크기 일치 보너스."""
    if not preds or not targets: return 0.0
    scores = []
    for p, t in zip(preds, targets):
        px = pixel_score(p, t)
        sh = shape_score(p, t)
        scores.append(px + (0.0 if px > 0 else sh * 0.3))
    return float(np.mean(scores))


# ── 빔 상태 ──────────────────────────────────────────────────────────────────────
# (score: float, op_seq: list[str], grids: list[NGrid])
BeamState = tuple[float, list[str], list[NGrid]]


def _single_beam_pass(
    beam: list[BeamState],
    ops: list[tuple[str, Callable]],
    targets: list[NGrid],
    beam_width: int,
    t0: float,
    timeout: float,
) -> tuple[list[BeamState], list[tuple[list[str], list[NGrid]]]]:
    """단일 깊이 빔서치 스텝. (새 빔, 완전해 목록) 반환."""
    solutions: list[tuple[list[str], list[NGrid]]] = []
    candidates: list[BeamState] = []

    for score, op_seq, cur_grids in beam:
        if time.perf_counter()-t0 > timeout:
            break
        for op_name, op_fn in ops:
            try:
                new_grids = [op_fn(g) for g in cur_grids]
                if any(g.size == 0 for g in new_grids): continue
                s = score_all(new_grids, targets)
                new_seq = op_seq + [op_name]
                if s >= 1.0 - 1e-9:
                    solutions.append((new_seq, new_grids))
                else:
                    candidates.append((s, new_seq, new_grids))
            except Exception:
                continue

    # 중복 제거 및 상위 beam_width 유지
    candidates.sort(key=lambda x: -x[0])
    seen: set[tuple] = set()
    unique: list[BeamState] = []
    for s, seq, grids in candidates:
        key = (round(s, 4), tuple(seq[-2:]))
        if key not in seen:
            seen.add(key); unique.append((s, seq, grids))
        if len(unique) >= beam_width:
            break

    return unique, solutions


class BeamSearchEngine:
    """
    2단계 빔서치 엔진.

    Stage 1: 글로벌 빔서치 (깊이 1~max_depth)
      - 직접 해(direct op)를 먼저 시도
      - score=1.0 달성 시 즉시 반환 (Early Stopping)

    Stage 2: 파인튜닝 (Stage 1 최고 점수 ≥ 0.5인 경우)
      - Stage 1 최고 상태에서 추가 2단계 탐색
      - 더 넓은 빔 너비로 정밀 탐색
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
            return BeamResult(task.task_id, False, [], None, 0.0, "pair mismatch")

        # 캐시된 ops 또는 새로 빌드
        if task.task_id not in _OPS_CACHE:
            _OPS_CACHE[task.task_id] = build_task_ops(task)
        ops = _OPS_CACHE[task.task_id]

        # identity 점수
        init_score = score_all(train_in, train_out)
        if init_score >= 1.0 - 1e-9:
            return BeamResult(task.task_id, True, [], train_in, 1.0, "identity")

        # ── Stage 1: 글로벌 빔서치 ──────────────────────────────────────────────
        beam: list[BeamState] = [(init_score, [], list(train_in))]
        best: BeamState       = beam[0]
        all_solutions: list[tuple[list[str], list[NGrid]]] = []

        for depth in range(self.max_depth):
            if time.perf_counter()-t0 > self.timeout*0.7:
                break
            beam, solutions = _single_beam_pass(
                beam, ops, train_out, self.beam_width, t0, self.timeout*0.7
            )
            all_solutions.extend(solutions)
            if solutions: break      # 해 발견 → Stage 1 완료
            if beam and beam[0][0] > best[0]:
                best = beam[0]

        # ── Stage 2: 파인튜닝 ────────────────────────────────────────────────────
        if not all_solutions and best[0] >= 0.5 and time.perf_counter()-t0 < self.timeout:
            # 최고 상태에서 더 넓은 빔으로 추가 2단계
            fine_beam: list[BeamState] = [best]
            fine_width = min(self.beam_width * 2, 40)
            for _ in range(2):
                if time.perf_counter()-t0 > self.timeout:
                    break
                fine_beam, solutions = _single_beam_pass(
                    fine_beam, ops, train_out, fine_width, t0, self.timeout
                )
                all_solutions.extend(solutions)
                if solutions: break
                if fine_beam and fine_beam[0][0] > best[0]:
                    best = fine_beam[0]

        elapsed = time.perf_counter()-t0

        if all_solutions:
            all_solutions.sort(key=lambda x: len(x[0]))  # 짧은 시퀀스 우선
            seq1, grids1 = all_solutions[0]
            seq2, grids2 = (all_solutions[1] if len(all_solutions)>1
                            else (seq1, grids1))
            return BeamResult(task.task_id, True, seq1, grids1, 1.0, "",
                              second_seq=seq2, second_grids=grids2)

        s, seq, grids = best
        return BeamResult(task.task_id, False, seq, grids, s,
                          f"partial {s:.3f}")

    def apply_sequence(self, seq: list[str], inputs: list[Grid],
                       task: Task) -> list[NGrid]:
        if task.task_id not in _OPS_CACHE:
            _OPS_CACHE[task.task_id] = build_task_ops(task)
        ops_dict = {n: f for n,f in _OPS_CACHE[task.task_id]}
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
        st = "✓" if self.success else f"~{self.score:.3f}"
        return f"[{st}] {self.task_id}  {self.seq1[:3]}{'…' if len(self.seq1)>3 else ''}"


_beam_engine = BeamSearchEngine()
print(f"✓ Phase 5 로드 완료 (2단계 BeamSearchEngine  beam={BEAM_WIDTH}  depth={MAX_DEPTH})")
'''

# ── Phase 6: Solver ────────────────────────────────────────────────────────────
CELL_PHASE6 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 6: ARC Solver + 병렬 검증 파이프라인
# ═══════════════════════════════════════════════════════════════════

class HeuristicFallback:
    @staticmethod
    def best_guess(task: Task, test_inp: Grid) -> Grid:
        pred_shape = _shape_predictor.predict(task, test_inp)
        H = pred_shape[0] if pred_shape else len(test_inp)
        W = pred_shape[1] if pred_shape else (len(test_inp[0]) if test_inp else 1)
        cnt: Counter = Counter()
        for p in task.train:
            if p.output:
                for row in p.output: cnt.update(row)
        dominant = cnt.most_common(1)[0][0] if cnt else 0
        return [[int(dominant)]*W for _ in range(H)]

    @staticmethod
    def copy_input(inp: Grid) -> Grid:
        return copy.deepcopy(inp)


class ARCSolver:
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
        except: return fallback

    def _build_predictions(self, task: Task, result: BeamResult) -> list[Prediction]:
        preds = []
        for inp in task.test_inputs:
            fb1 = self.fallback.best_guess(task, inp)
            fb2 = self.fallback.copy_input(inp)
            if result.success and result.grids1 is not None:
                test_g1 = self.engine.apply_sequence(result.seq1, [inp], task)
                a1 = self._ngrid_to_list(test_g1[0] if test_g1 else None, fb1)
                if result.second_seq and result.second_seq != result.seq1:
                    test_g2 = self.engine.apply_sequence(result.second_seq, [inp], task)
                    a2 = self._ngrid_to_list(test_g2[0] if test_g2 else None, fb2)
                else:
                    a2 = fb2
            elif result.grids1 is not None and result.score >= 0.5:
                test_g1 = self.engine.apply_sequence(result.seq1, [inp], task)
                a1 = self._ngrid_to_list(test_g1[0] if test_g1 else None, fb1)
                a2 = fb1
            else:
                a1 = fb1; a2 = fb2
            preds.append(Prediction(attempt_1=a1, attempt_2=a2))
        return preds

    def solve_all(self, tasks: dict[TaskId, Task],
                  progress_fn=None) -> tuple[dict, dict]:
        all_preds: dict[TaskId, list[Prediction]] = {}
        all_results: dict[TaskId, BeamResult]     = {}

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {ex.submit(self.solve_task, task): tid
                        for tid, task in tasks.items()}
                for fut in as_completed(futs):
                    tid = futs[fut]
                    try:
                        _, preds, res = fut.result()
                    except Exception as e:
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
print(f"training={len(train_tasks)}  eval={len(eval_tasks)}  test={len(test_tasks)}")
print("✓ Test 1 통과")
'''

CELL_TEST2 = '''\
# ── Test 2: 확장 DSL 검증 ─────────────────────────────────────────────────────
g = g2n([[0,1,0],[1,1,1],[0,1,0]])

# 오브젝트 연산
objs = find_objects(g, 0, 4)
print(f"오브젝트 수: {len(objs)}")
assert len(objs) == 1   # 십자 모양 = 1개
lg = keep_n_largest(g, 1)
assert np.array_equal(lg, g)
print("✓ keep_n_largest")

# 구멍 채우기
g_ring = g2n([[1,1,1],[1,0,1],[1,1,1]])
filled = fill_holes(g_ring, -1, 0)
assert filled[1,1] == 1, f"구멍 색상 = {filled[1,1]}"
print("✓ fill_holes")

# 대칭 완성
g_half = g2n([[1,0,0],[0,1,0],[0,0,0]])
sym = complete_symmetry_h(g_half, 0)
assert sym[0,2] == 1
print("✓ complete_symmetry_h")

# 미러 확장
g2 = g2n([[1,2],[3,4]])
mr = mirror_extend_right(g2)
assert mr.shape == (2, 4)
assert mr[0,2] == 2 and mr[0,3] == 1
print("✓ mirror_extend_right")

# add_border / remove_border
g3 = g2n([[5]])
bordered = add_border(g3, 1)
assert bordered.shape == (3,3)
assert bordered[0,0] == 1 and bordered[1,1] == 5
removed = remove_border(bordered)
assert np.array_equal(removed, g3)
print("✓ add_border / remove_border")

# 색상 정렬
g4 = g2n([[2,2,1],[2,1,3]])
sc = sort_colors_by_frequency(g4, 0)
# color2(가장많음)→1, color1→2, color3→3
assert sc[0,0] == 1 and sc[1,2] == 3
print("✓ sort_colors_by_frequency")

# 패턴 추출
g5 = g2n([[1,2,1,2],[3,4,3,4],[1,2,1,2],[3,4,3,4]])
pp = find_periodic_period(g5)
assert pp == (2,2), f"period={pp}"
pat = extract_pattern(g5)
assert pat.shape == (2,2)
print("✓ find_periodic_period / extract_pattern")

# 격자 분할
gsep = g2n([[1,2,0,3,4],[5,6,0,7,8],[0,0,0,0,0],[1,1,0,2,2]])
subs = split_by_separator(gsep, 0)
assert subs is not None and len(subs) == 4
print(f"✓ split_by_separator ({len(subs)} subs)")

print("\\n✓ Test 2 통과")
'''

CELL_TEST3 = '''\
# ── Test 3: 학습 기반 연산자 검증 ─────────────────────────────────────────────
# 색상 매핑 학습 테스트
task_cm = Task("_cm", [
    Pair([[1,2,3],[1,2,3]], [[4,5,6],[4,5,6]]),
    Pair([[1,1,2],[3,3,1]], [[4,4,5],[6,6,4]]),
], [[[1,2,3]]])

cmap = _learn_exact_color_map(task_cm)
assert cmap == {1:4, 2:5, 3:6}, f"learned cmap={cmap}"
print(f"✓ _learn_exact_color_map: {cmap}")

# 직접 연산자 탐지 테스트 (scale2x2)
task_sc = Task("_sc", [
    Pair([[1,2],[3,4]],
         [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]),
    Pair([[5,6],[7,8]],
         [[5,5,6,6],[5,5,6,6],[7,7,8,8],[7,7,8,8]]),
], [[[0,1],[2,3]]])
direct = _check_direct_fixed_op(task_sc)
assert len(direct) > 0, "scale2x2 직접 해를 찾지 못함"
print(f"✓ _check_direct_fixed_op: {[n for n,_ in direct][:3]}")

# ShapePredictor
sp = ShapePredictor()
assert sp.predict(task_sc, [[0,0],[0,0]]) == (4, 4)
print("✓ ShapePredictor scale2x2")

print("\\n✓ Test 3 통과")
'''

CELL_TEST4 = '''\
# ── Test 4: 2단계 빔서치 + 학습 기반 ──────────────────────────────────────────
engine = BeamSearchEngine(beam_width=15, max_depth=3, timeout=20.0)

# (a) 색상 매핑 학습으로 즉시 풀이
task_cm = Task("_cm2", [
    Pair([[1,0,1],[0,2,0]], [[3,0,3],[0,4,0]]),
    Pair([[1,1,0],[2,0,2]], [[3,3,0],[4,0,4]]),
], [[[1,2,1]]])
res_cm = engine.solve(task_cm)
print(f"[color_map] success={res_cm.success} score={res_cm.score:.3f} seq={res_cm.seq1}")
assert res_cm.success, f"색상 매핑 태스크 실패: {res_cm.error}"
print("✓ 학습 색상 매핑 풀이")

# (b) 회전 태스크
task_rot = Task("_rot2", [
    Pair([[1,2,3],[4,5,6],[7,8,9]],
         [[7,4,1],[8,5,2],[9,6,3]]),
    Pair([[0,1,0],[1,0,1],[0,1,0]],
         [[0,1,0],[1,0,1],[0,1,0]]),
], [[[1,0,0],[0,1,0],[0,0,1]]])
res_rot = engine.solve(task_rot)
print(f"[rotation] success={res_rot.success} score={res_rot.score:.3f} seq={res_rot.seq1}")
assert res_rot.score >= 0.5
print("✓ 회전 태스크 탐색")

# (c) 스케일 태스크
task_sc = Task("_sc2", [
    Pair([[1,2],[3,4]], [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]),
    Pair([[5,6],[7,8]], [[5,5,6,6],[5,5,6,6],[7,7,8,8],[7,7,8,8]]),
], [[[0,1],[2,3]]])
res_sc = engine.solve(task_sc)
print(f"[scale]    success={res_sc.success} score={res_sc.score:.3f} seq={res_sc.seq1}")
assert res_sc.success, f"스케일 태스크 실패: {res_sc.error}"
print("✓ 스케일 태스크 완전 풀이")

print("\\n✓ Test 4 통과")
'''

CELL_TEST5 = '''\
# ── Test 5: E2E 파이프라인 (10개 Task) ────────────────────────────────────────
try:
    from tqdm.notebook import tqdm as _tqdm
except ImportError:
    from tqdm import tqdm as _tqdm

_sample = dict(list(train_tasks.items())[:10])   # training 으로 검증
_solver_test = ARCSolver(beam_width=10, max_depth=3, timeout=20.0, max_workers=1)
_all_preds: dict = {}; _all_res: dict = {}
_solved = 0

for _tid, _task in _tqdm(_sample.items(), desc="E2E Test", unit="task"):
    _, _preds, _res = _solver_test.solve_task(_task)
    _all_preds[_tid] = _preds; _all_res[_tid] = _res
    if _res.success: _solved += 1

_scores = ARCEvaluator().evaluate(_all_preds, _sample)
_path = ARCSubmissionWriter().save(_all_preds, "submission_test.json")

print(f"\\n10-task 샘플  solved={_solved}/10  accuracy={_scores[\'overall_score\']:.4f}")
for _tid, _r in _all_res.items():
    _st = "✓" if _r.success else f"~{_r.score:.2f}"
    _seq_str = str(_r.seq1[:3]).replace("direct_","d:")
    print(f"  [{_st}] {_tid}  {_seq_str}")

assert _path.exists()
print("\\n✓ Test 5 통과 — E2E 파이프라인 정상")
'''

CELL_CONFIG = '''\
# ⚙️ 설정
SPLIT             = "evaluation"    # "training" | "evaluation" | "test"
MAX_TASKS         = None            # None = 전체
OUTPUT_FILE       = "submission.json"

RUN_BEAM_WIDTH    = 20
RUN_MAX_DEPTH     = 4
RUN_TASK_TIMEOUT  = 90.0
RUN_MAX_WORKERS   = max(1, os.cpu_count() or 1)

print(f"설정: split={SPLIT}  beam={RUN_BEAM_WIDTH}  depth={RUN_MAX_DEPTH}"
      f"  timeout={RUN_TASK_TIMEOUT}s  workers={RUN_MAX_WORKERS}")
'''

CELL_RUN = '''\
# 🚀 메인 실행
try:
    from tqdm.notebook import tqdm as _ntqdm
except ImportError:
    from tqdm import tqdm as _ntqdm

_loader = ARCDataLoader()
_tasks  = {"training":   _loader.load_training,
           "evaluation": _loader.load_evaluation,
           "test":       _loader.load_test}[SPLIT]()
if MAX_TASKS:
    _tasks = dict(list(_tasks.items())[:MAX_TASKS])
print(f"✓ {len(_tasks)} tasks 로드 ({SPLIT})")

_run_solver = ARCSolver(
    beam_width=RUN_BEAM_WIDTH,
    max_depth=RUN_MAX_DEPTH,
    timeout=RUN_TASK_TIMEOUT,
    max_workers=RUN_MAX_WORKERS,
)

_all_preds: dict = {}; _all_res: dict = {}
_total_start = time.perf_counter()

with _ntqdm(total=len(_tasks), desc="Beam Search", unit="task") as _pbar:
    def _upd(n): _pbar.update(n)
    _all_preds, _all_res = _run_solver.solve_all(_tasks, progress_fn=_upd)

_elapsed = time.perf_counter()-_total_start

_eval_scores = None
if SPLIT in ("training","evaluation"):
    _eval_scores = ARCEvaluator().evaluate(_all_preds, _tasks)

_saved = ARCSubmissionWriter().save(_all_preds, OUTPUT_FILE)

_tot    = len(_all_res)
_solved = sum(1 for r in _all_res.values() if r.success)
_hi     = sum(1 for r in _all_res.values() if not r.success and r.score >= 0.9)
_mid    = sum(1 for r in _all_res.values() if not r.success and 0.5 <= r.score < 0.9)
_low    = _tot - _solved - _hi - _mid

print()
print("="*64)
print(" ARC-AGI-2 Beam Search v2 실행 요약")
print("="*64)
print(f"  총 tasks      : {_tot}")
print(f"  처리 시간     : {_elapsed:.1f}s  ({_elapsed/_tot:.1f}s/task)")
print()
print(f"  ✓ 완전 풀이   : {_solved:4d}  ({_solved/_tot*100:.1f}%)")
print(f"  ≥0.9 근접     : {_hi:4d}  ({_hi/_tot*100:.1f}%)")
print(f"  0.5~0.9 부분  : {_mid:4d}  ({_mid/_tot*100:.1f}%)")
print(f"  <0.5 폴백     : {_low:4d}  ({_low/_tot*100:.1f}%)")
if _eval_scores:
    _pf = sum(1 for s in _eval_scores["task_scores"].values() if s==1.0)
    print()
    print(f"  평가 점수(overall) : {_eval_scores[\'overall_score\']:.4f}")
    print(f"  완벽 task          : {_pf}/{_tot}")
print("="*64)
print(f"\\n제출 파일: {_saved}")
'''

# ══════════════════════════════════════════════════════════════════════════════
cells = [
    md_cell(MD_TITLE),
    md_cell("## 0. 패키지 설치"),
    code_cell(CELL_INSTALL),
    md_cell("## 1. 임포트 및 전역 설정"),
    code_cell(CELL_IMPORTS),
    md_cell("## Phase 1: 데이터 파이프라인"),
    code_cell(CELL_PHASE1),
    md_cell("## Phase 2: NumPy DSL 기본 프리미티브"),
    code_cell(CELL_PHASE2),
    md_cell("## Phase 2b: 확장 DSL (오브젝트·대칭·패턴·합성)"),
    code_cell(CELL_PHASE2B),
    md_cell("## Phase 3: 출력 크기 예측기"),
    code_cell(CELL_PHASE3),
    md_cell("## Phase 4: 연산자 라이브러리 (고정 + 학습 기반)"),
    code_cell(CELL_PHASE4),
    md_cell("## Phase 5: 2단계 빔서치 엔진"),
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
print(f"✓ 노트북 생성: {out}  ({out.stat().st_size//1024} KB)")
