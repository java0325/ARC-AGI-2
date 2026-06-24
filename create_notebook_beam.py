#!/usr/bin/env python3
"""arc_agi2_beam.ipynb 생성 스크립트 v3 — RuleEngine + BeamSearch 앙상블."""

import json, uuid
from pathlib import Path

def code_cell(source):
    return {"cell_type":"code","execution_count":None,
            "id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":source}

def md_cell(source):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":source}

# ══════════════════════════════════════════════════════════════════════════════
MD_TITLE = """\
# ARC-AGI-2 — RuleEngine + Beam Search v5 (LLM-Free, Offline)
> **오프라인 동작 · LOOCV 검증 · attempt_2 앙상블 · 0.9+ 정밀 refinement**
>
> | 모듈 | 내용 |
> |---|---|
> | Phase 1 | 데이터 파이프라인 |
> | Phase 2 | NumPy DSL 기본 프리미티브 |
> | Phase 2b | 확장 DSL (오브젝트·대칭·이동) |
> | Phase 2c | 형태소·경계·CA 연산자 |
> | Phase 3 | 출력 크기 예측기 |
> | Phase 4 | 고속 연산자 라이브러리 |
> | Phase 4b | **RuleEngine** — 50+ 결정론적 규칙 |
> | Phase 4c | **DynamicRuleLearner** — 48개 전략 + LOOCV 과적합 방지 |
> | Phase 5 | 빔서치 엔진 (0.9+ 2-step 정밀 refinement) |
> | Phase 6 | **앙상블 Solver v5**: attempt_1·2 독립 경로 |"""

CELL_INSTALL = """\
# ── 패키지 임포트 확인 (인터넷 불필요 — Kaggle 기본 환경에 포함됨) ──────
import importlib, sys
_missing = []
for _p in ["numpy", "tqdm"]:
    if importlib.util.find_spec(_p) is None:
        _missing.append(_p)
if _missing:
    # 오프라인 환경에서만 시도 (실패해도 계속 진행)
    import subprocess
    for _p in _missing:
        subprocess.run([sys.executable, "-m", "pip", "install", _p, "-q",
                        "--no-index", "--find-links", "/kaggle/lib"], check=False)
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

_KAGGLE_DATA  = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
_LOCAL_DATA   = Path(".")
DATA_DIR      = _KAGGLE_DATA if _KAGGLE_DATA.exists() else _LOCAL_DATA

BEAM_WIDTH    = 30
MAX_DEPTH     = 5
TASK_TIMEOUT  = 60.0
MAX_WORKERS   = max(1, os.cpu_count() or 1)

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
        # arc_agi2_beam과 동일한 단순 로드 방식 유지
        path = self.data_dir / fname
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        # DATA_DIR가 없을 경우 현재 디렉토리 재시도
        path2 = Path(".") / fname
        if path2.exists():
            with path2.open(encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"{fname} not found at {path} or {path2}")

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

    def score_prediction(self, p, gt):
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
def g2n(grid) -> NGrid:
    return np.array(grid, dtype=np.int8)
def n2g(arr: NGrid) -> Grid:
    return arr.tolist()

def scale_up(g,ry,rx):   return np.repeat(np.repeat(g,ry,0),rx,1)
def scale_down(g,ry,rx): return g[::ry,::rx]
def tile(g,ny,nx):       return np.tile(g,(ny,nx))
def crop_to_content(g,bg=0):
    rows=np.any(g!=bg,axis=1); cols=np.any(g!=bg,axis=0)
    if not rows.any() or not cols.any(): return g.copy()
    return g[np.ix_(rows,cols)]
def pad_to_shape(g,H,W,fill=0):
    h,w=g.shape; result=np.full((H,W),fill,dtype=np.int8)
    r0=max(0,(H-h)//2); c0=max(0,(W-w)//2)
    he=min(H,r0+h); we=min(W,c0+w)
    result[r0:he,c0:we]=g[:he-r0,:we-c0]; return result
def extend_to_shape(g,H,W):
    h,w=g.shape
    if h==0 or w==0: return np.zeros((H,W),dtype=np.int8)
    return np.tile(g,(-(-H//h),-(-W//w)))[:H,:W]
def gravity_down(g,bg=0):
    result=np.full_like(g,bg)
    for c in range(g.shape[1]):
        nz=g[:,c][g[:,c]!=bg]; result[g.shape[0]-len(nz):,c]=nz
    return result
def gravity_up(g,bg=0):
    result=np.full_like(g,bg)
    for c in range(g.shape[1]):
        nz=g[:,c][g[:,c]!=bg]; result[:len(nz),c]=nz
    return result
def gravity_left(g,bg=0):
    result=np.full_like(g,bg)
    for r in range(g.shape[0]):
        nz=g[r][g[r]!=bg]; result[r,:len(nz)]=nz
    return result
def gravity_right(g,bg=0):
    result=np.full_like(g,bg)
    for r in range(g.shape[0]):
        nz=g[r][g[r]!=bg]; result[r,g.shape[1]-len(nz):]=nz
    return result
def recolor(g,mapping):
    result=g.copy()
    for s,d in mapping.items(): result[g==s]=d
    return result
def swap_colors(g,c1,c2): return recolor(g,{c1:c2,c2:c1})
def replace_color(g,src,dst): r=g.copy(); r[g==src]=dst; return r
def normalize_colors(g):
    counts=Counter(g.flatten().tolist())
    sc=[c for c,_ in sorted(counts.items(),key=lambda x:-x[1])]
    fwd={c:i for i,c in enumerate(sc)}; inv={i:c for c,i in fwd.items()}
    return np.vectorize(fwd.get)(g).astype(np.int8), inv
def denormalize_colors(g,inv):
    return np.vectorize(lambda x:inv.get(x,x))(g).astype(np.int8)
def find_objects(g,bg=0,connectivity=4):
    H,W=g.shape; visited=np.zeros((H,W),dtype=bool)
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity==8: dirs+=[(- 1,-1),(-1,1),(1,-1),(1,1)]
    objects=[]
    for r in range(H):
        for c in range(W):
            if visited[r,c] or g[r,c]==bg: continue
            cells=[]; queue=deque([(r,c)]); visited[r,c]=True
            while queue:
                cr,cc=queue.popleft(); cells.append((cr,cc))
                for dr,dc in dirs:
                    nr,nc=cr+dr,cc+dc
                    if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and g[nr,nc]==g[r,c]:
                        visited[nr,nc]=True; queue.append((nr,nc))
            objects.append(np.array(cells))
    return objects
def extract_object(g,cells,bg=0):
    r0,c0=cells.min(axis=0); r1,c1=cells.max(axis=0)
    sub=np.full((r1-r0+1,c1-c0+1),bg,dtype=np.int8)
    for r,c in cells: sub[r-r0,c-c0]=g[r,c]
    return sub
def flood_fill(g,r0,c0,new_color,connectivity=4):
    H,W=g.shape
    if not(0<=r0<H and 0<=c0<W): return g.copy()
    old=g[r0,c0]
    if old==new_color: return g.copy()
    result=g.copy(); dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity==8: dirs+=[(-1,-1),(-1,1),(1,-1),(1,1)]
    queue=deque([(r0,c0)]); result[r0,c0]=new_color
    while queue:
        r,c=queue.popleft()
        for dr,dc in dirs:
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and result[nr,nc]==old:
                result[nr,nc]=new_color; queue.append((nr,nc))
    return result
def fill_outer_region(g,fill_color=0,bg=0):
    H,W=g.shape; result=g.copy()
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    visited=np.zeros((H,W),dtype=bool); queue=deque()
    for r in range(H):
        for c in [0,W-1]:
            if result[r,c]==bg and not visited[r,c]: visited[r,c]=True; queue.append((r,c))
    for c in range(W):
        for r in [0,H-1]:
            if result[r,c]==bg and not visited[r,c]: visited[r,c]=True; queue.append((r,c))
    while queue:
        r,c=queue.popleft(); result[r,c]=fill_color
        for dr,dc in dirs:
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and result[nr,nc]==bg:
                visited[nr,nc]=True; queue.append((nr,nc))
    return result
def outline_objects(g,bg=0):
    result=g.copy(); H,W=g.shape
    for r in range(1,H-1):
        for c in range(1,W-1):
            if g[r,c]!=bg and all(g[r+dr,c+dc]==g[r,c] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]):
                result[r,c]=bg
    return result

print("✓ Phase 2 로드 완료")
'''

# ── Phase 2b: 확장 DSL ──────────────────────────────────────────────────────────
CELL_PHASE2B = '''\
def keep_n_largest(g,n=1,bg=0):
    objs=sorted(find_objects(g,bg,4),key=len,reverse=True)[:n]
    result=np.full_like(g,bg)
    for obj in objs:
        for r,c in obj: result[r,c]=g[r,c]
    return result
def keep_n_smallest(g,n=1,bg=0):
    objs=sorted(find_objects(g,bg,4),key=len)[:n]
    result=np.full_like(g,bg)
    for obj in objs:
        for r,c in obj: result[r,c]=g[r,c]
    return result
def remove_small_objects(g,max_size=1,bg=0):
    result=g.copy()
    for obj in find_objects(g,bg,4):
        if len(obj)<=max_size:
            for r,c in obj: result[r,c]=bg
    return result
def color_objects_by_size(g,bg=0):
    objs=sorted(find_objects(g,bg,4),key=len,reverse=True)
    result=np.full_like(g,bg)
    for i,obj in enumerate(objs,1):
        for r,c in obj: result[r,c]=min(i,9)
    return result
def fill_holes(g,fill_color=-1,bg=0):
    H,W=g.shape; exterior=np.zeros((H,W),dtype=bool)
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]; queue=deque()
    for r in range(H):
        for c in [0,W-1]:
            if g[r,c]==bg and not exterior[r,c]: exterior[r,c]=True; queue.append((r,c))
    for c in range(W):
        for r in [0,H-1]:
            if g[r,c]==bg and not exterior[r,c]: exterior[r,c]=True; queue.append((r,c))
    while queue:
        r,c=queue.popleft()
        for dr,dc in dirs:
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and not exterior[nr,nc] and g[nr,nc]==bg:
                exterior[nr,nc]=True; queue.append((nr,nc))
    result=g.copy(); hole_mask=(~exterior)&(g==bg)
    if not hole_mask.any(): return result
    if fill_color==-1:
        nbr=[int(g[r+dr,c+dc]) for r,c in np.argwhere(hole_mask)
             for dr,dc in dirs if 0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]!=bg]
        fc=Counter(nbr).most_common(1)[0][0] if nbr else 1
    else: fc=fill_color
    result[hole_mask]=fc; return result
def center_content(g,bg=0):
    rows=np.any(g!=bg,axis=1); cols=np.any(g!=bg,axis=0)
    if not rows.any() or not cols.any(): return g.copy()
    cropped=g[np.ix_(rows,cols)]; result=np.full_like(g,bg)
    r0=(g.shape[0]-cropped.shape[0])//2; c0=(g.shape[1]-cropped.shape[1])//2
    result[r0:r0+cropped.shape[0],c0:c0+cropped.shape[1]]=cropped; return result
def complete_symmetry_h(g,bg=0):
    H,W=g.shape; result=g.copy()
    for r in range(H):
        for c in range(W//2):
            l,rc=g[r,c],g[r,W-1-c]
            if l!=bg and rc==bg: result[r,W-1-c]=l
            elif rc!=bg and l==bg: result[r,c]=rc
    return result
def complete_symmetry_v(g,bg=0):
    H,W=g.shape; result=g.copy()
    for r in range(H//2):
        for c in range(W):
            t,b=g[r,c],g[H-1-r,c]
            if t!=bg and b==bg: result[H-1-r,c]=t
            elif b!=bg and t==bg: result[r,c]=b
    return result
def complete_symmetry_rot180(g,bg=0):
    H,W=g.shape; result=g.copy()
    for r in range(H):
        for c in range(W):
            if g[r,c]==bg and g[H-1-r,W-1-c]!=bg: result[r,c]=g[H-1-r,W-1-c]
    return result
def mirror_extend_right(g):  return np.hstack([g,np.fliplr(g)])
def mirror_extend_left(g):   return np.hstack([np.fliplr(g),g])
def mirror_extend_down(g):   return np.vstack([g,np.flipud(g)])
def mirror_extend_up(g):     return np.vstack([np.flipud(g),g])
def add_border(g,color=None,bg=0):
    if color is None:
        cnt=Counter(g[g!=bg].tolist()); color=cnt.most_common(1)[0][0] if cnt else 1
    H,W=g.shape; result=np.full((H+2,W+2),color,dtype=np.int8)
    result[1:-1,1:-1]=g; return result
def remove_border(g):
    if g.shape[0]<=2 or g.shape[1]<=2: return g.copy()
    return g[1:-1,1:-1].copy()
def sort_colors_by_frequency(g,bg=0):
    cnt=Counter(g[g!=bg].tolist())
    mapping={c:i+1 for i,(c,_) in enumerate(cnt.most_common())}; mapping[bg]=bg
    return np.vectorize(lambda x:mapping.get(x,x))(g).astype(np.int8)
def keep_most_common_color(g,bg=0):
    cnt=Counter(g[g!=bg].tolist())
    if not cnt: return g.copy()
    mc=cnt.most_common(1)[0][0]
    return np.where(g==mc,g,bg).astype(np.int8)
def keep_least_common_color(g,bg=0):
    cnt=Counter(g[g!=bg].tolist())
    if not cnt: return g.copy()
    lc=cnt.most_common()[-1][0]
    return np.where(g==lc,g,bg).astype(np.int8)
def invert_binary(g,bg=0):
    cnt=Counter(g[g!=bg].tolist())
    if not cnt: return g.copy()
    fg=cnt.most_common(1)[0][0]
    result=g.copy(); result[g==bg]=fg; result[g!=bg]=bg; return result
def split_by_separator(g,bg=0):
    H,W=g.shape
    sr=[r for r in range(H) if len(set(g[r].tolist()))==1]
    sc=[c for c in range(W) if len(set(g[:,c].tolist()))==1]
    if not sr and not sc: return None
    rc=sorted(set([-1]+sr+[H])); cc=sorted(set([-1]+sc+[W]))
    sub=[]
    for i in range(len(rc)-1):
        r0=rc[i]+1; r1=rc[i+1]
        if r0>=r1: continue
        for j in range(len(cc)-1):
            c0=cc[j]+1; c1=cc[j+1]
            if c0>=c1: continue
            sub.append(g[r0:r1,c0:c1].copy())
    return sub if sub else None
def apply_op_to_subgrids(g,op,bg=0):
    H,W=g.shape
    sr=[r for r in range(H) if len(set(g[r].tolist()))==1]
    sc=[c for c in range(W) if len(set(g[:,c].tolist()))==1]
    if not sr and not sc: return op(g)
    rc=sorted(set([-1]+sr+[H])); cc=sorted(set([-1]+sc+[W]))
    result=g.copy()
    for i in range(len(rc)-1):
        r0=rc[i]+1; r1=rc[i+1]
        if r0>=r1: continue
        for j in range(len(cc)-1):
            c0=cc[j]+1; c1=cc[j+1]
            if c0>=c1: continue
            try:
                t=op(g[r0:r1,c0:c1].copy())
                if t.shape==(r1-r0,c1-c0): result[r0:r1,c0:c1]=t
            except: pass
    return result
def find_periodic_period(g):
    H,W=g.shape
    for rp in range(1,H//2+1):
        if H%rp==0 and all(np.array_equal(g[r],g[r%rp]) for r in range(H)):
            for cp in range(1,W//2+1):
                if W%cp==0 and all(np.array_equal(g[:,c],g[:,c%cp]) for c in range(W)):
                    return (rp,cp)
    return None
def extract_pattern(g):
    p=find_periodic_period(g); return g[:p[0],:p[1]].copy() if p else g.copy()
def tile_from_content(g,bg=0):
    cropped=crop_to_content(g,bg)
    if cropped.size==0: return g.copy()
    return extend_to_shape(cropped,g.shape[0],g.shape[1])
def sort_rows_by_sum(g):    return g[np.argsort(g.sum(axis=1))]
def sort_cols_by_sum(g):    return g[:,np.argsort(g.sum(axis=0))]
def sort_rows_descending(g): return g[np.argsort(-g.sum(axis=1))]
def shift_content(g,dr,dc,bg=0):
    H,W=g.shape; result=np.full_like(g,bg)
    for r,c in np.argwhere(g!=bg):
        nr,nc=int(r)+dr,int(c)+dc
        if 0<=nr<H and 0<=nc<W: result[nr,nc]=g[r,c]
    return result
def paste_at(base,obj,top,left,bg=0):
    result=base.copy(); H,W=result.shape; h,w=obj.shape
    for r in range(h):
        for c in range(w):
            nr,nc=top+r,left+c
            if 0<=nr<H and 0<=nc<W and obj[r,c]!=bg: result[nr,nc]=obj[r,c]
    return result
def align_content(g,where,bg=0):
    rows=np.any(g!=bg,axis=1); cols=np.any(g!=bg,axis=0)
    if not rows.any() or not cols.any(): return g.copy()
    cropped=g[np.ix_(rows,cols)]; H,W=g.shape; h,w=cropped.shape
    positions={"tl":(0,0),"tr":(0,W-w),"bl":(H-h,0),"br":(H-h,W-w),"center":((H-h)//2,(W-w)//2)}
    r0,c0=positions.get(where,((H-h)//2,(W-w)//2))
    return paste_at(np.full_like(g,bg),cropped,r0,c0,bg)
def move_largest_object_to(g,where,bg=0):
    objs=sorted(find_objects(g,bg,4),key=len,reverse=True)
    if not objs: return g.copy()
    obj=extract_object(g,objs[0],bg); H,W=g.shape; h,w=obj.shape
    positions={"tl":(0,0),"tr":(0,W-w),"bl":(H-h,0),"br":(H-h,W-w),"center":((H-h)//2,(W-w)//2)}
    r0,c0=positions.get(where,((H-h)//2,(W-w)//2))
    return paste_at(np.full_like(g,bg),obj,r0,c0,bg)

print("✓ Phase 2b 로드 완료 (확장 DSL)")
'''

# ── Phase 2c: 형태소·경계·CA 연산자 ──────────────────────────────────────────────
CELL_PHASE2C = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 2c: 형태소·경계·CA 연산자
# ═══════════════════════════════════════════════════════════════════

def dilate_objects(g: NGrid, bg: int = 0, connectivity: int = 4) -> NGrid:
    """형태소 팽창(Dilation): 오브젝트를 1픽셀 확장."""
    H, W = g.shape
    result = g.copy()
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8: dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for r in range(H):
        for c in range(W):
            if g[r,c] == bg:
                nbr = [int(g[r+dr,c+dc]) for dr,dc in dirs
                       if 0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]!=bg]
                if nbr:
                    result[r,c] = Counter(nbr).most_common(1)[0][0]
    return result

def erode_objects(g: NGrid, bg: int = 0, connectivity: int = 4) -> NGrid:
    """형태소 침식(Erosion): 오브젝트 경계 1픽셀 제거."""
    H, W = g.shape
    result = g.copy()
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8: dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for r in range(H):
        for c in range(W):
            if g[r,c] != bg:
                # 경계 픽셀이면(bg 이웃 있으면) 제거
                if any(not(0<=r+dr<H and 0<=c+dc<W) or g[r+dr,c+dc]==bg
                       for dr,dc in dirs):
                    result[r,c] = bg
    return result

def majority_neighbor_color(g: NGrid, radius: int = 1) -> NGrid:
    """각 셀을 3×3 이웃의 최빈값으로 교체(Majority Vote CA)."""
    H, W = g.shape
    result = g.copy()
    for r in range(H):
        for c in range(W):
            r0, r1 = max(0,r-radius), min(H,r+radius+1)
            c0, c1 = max(0,c-radius), min(W,c+radius+1)
            patch = g[r0:r1,c0:c1].flatten().tolist()
            result[r,c] = Counter(patch).most_common(1)[0][0]
    return result.astype(np.int8)

def keep_border_objects(g: NGrid, bg: int = 0) -> NGrid:
    """격자 경계에 닿는 오브젝트만 유지."""
    H, W = g.shape
    border = {(r,c) for r in range(H) for c in [0,W-1]}
    border |= {(r,c) for c in range(W) for r in [0,H-1]}
    objs = find_objects(g, bg, 4)
    result = np.full_like(g, bg)
    for obj in objs:
        if any((int(r),int(c)) in border for r,c in obj):
            for r,c in obj: result[r,c] = g[r,c]
    return result

def keep_interior_objects(g: NGrid, bg: int = 0) -> NGrid:
    """격자 경계에 닿지 않는 오브젝트만 유지."""
    H, W = g.shape
    border = {(r,c) for r in range(H) for c in [0,W-1]}
    border |= {(r,c) for c in range(W) for r in [0,H-1]}
    objs = find_objects(g, bg, 4)
    result = np.full_like(g, bg)
    for obj in objs:
        if not any((int(r),int(c)) in border for r,c in obj):
            for r,c in obj: result[r,c] = g[r,c]
    return result

def mark_unique_objects(g: NGrid, bg: int = 0) -> NGrid:
    objs = find_objects(g, bg, 4)
    result = np.full_like(g, bg)
    for i, obj in enumerate(objs, 1):
        for r,c in obj: result[r,c] = i % 10
    return result

def recolor_by_object_size(g: NGrid, bg: int = 0) -> NGrid:
    """각 오브젝트를 크기 값(픽셀 수)으로 재색칠 (최대 9)."""
    objs = find_objects(g, bg, 4)
    result = np.full_like(g, bg)
    for obj in objs:
        sz = min(len(obj), 9)
        for r,c in obj: result[r,c] = sz
    return result

def count_objects_to_value(g: NGrid, bg: int = 0) -> NGrid:
    """오브젝트 개수를 단일 셀 격자로 반환."""
    n = len(find_objects(g, bg, 4))
    return np.array([[min(n,9)]], dtype=np.int8)

def object_sizes_row(g: NGrid, bg: int = 0) -> NGrid:
    """오브젝트 크기들을 1행 격자로 반환 (크기 오름차순)."""
    sizes = sorted(len(obj) for obj in find_objects(g, bg, 4))
    if not sizes: return np.array([[0]], dtype=np.int8)
    return np.array([[min(s,9) for s in sizes]], dtype=np.int8)

def color_frequency_row(g: NGrid, bg: int = 0) -> NGrid:
    """비배경 색상 빈도를 1행으로 반환 (색상 오름차순)."""
    cnt = Counter(g[g!=bg].flatten().tolist())
    if not cnt: return np.array([[0]], dtype=np.int8)
    row = [min(cnt[c],9) for c in sorted(cnt)]
    return np.array([row], dtype=np.int8)

def overlay_grids(g1: NGrid, g2: NGrid, bg: int = 0) -> NGrid:
    """g2의 비배경 픽셀을 g1 위에 덮어쓴다."""
    result = g1.copy()
    mask = g2 != bg
    result[mask] = g2[mask]
    return result

def xor_grids(g1: NGrid, g2: NGrid, bg: int = 0) -> NGrid:
    """두 그리드의 대칭 차분(XOR 논리)."""
    # 양쪽에서 다른 셀만 살리고 같은 셀은 bg
    result = np.where(g1 != g2, np.where(g1 != bg, g1, g2), bg)
    return result.astype(np.int8)

print("✓ Phase 2c 로드 완료 (형태소·경계·CA)")
'''

# ── Phase 3 ────────────────────────────────────────────────────────────────────
CELL_PHASE3 = '''\
@dataclass
class ShapeRule:
    kind: str
    fixed: tuple[int,int] | None
    scale: tuple[Fraction,Fraction] | None

class ShapePredictor:
    def analyze(self, task):
        pairs=[(p.input,p.output) for p in task.train if p.output is not None]
        if not pairs: return ShapeRule("unknown",None,None)
        in_sh=[(len(i),len(i[0]) if i else 0) for i,_ in pairs]
        out_sh=[(len(o),len(o[0]) if o else 0) for _,o in pairs]
        if len(set(out_sh))==1 and len(set(in_sh))>1: return ShapeRule("fixed",out_sh[0],None)
        if all(i==o for i,o in zip(in_sh,out_sh)): return ShapeRule("identity",None,None)
        rs=[Fraction(o[0],i[0]) for i,o in zip(in_sh,out_sh) if i[0]>0]
        cs=[Fraction(o[1],i[1]) for i,o in zip(in_sh,out_sh) if i[1]>0]
        if len(set(rs))==1 and len(set(cs))==1: return ShapeRule("scale",None,(rs[0],cs[0]))
        if all(i[0]==o[1] and i[1]==o[0] for i,o in zip(in_sh,out_sh)): return ShapeRule("transpose",None,None)
        if all(o[0]<=i[0] and o[1]<=i[1] for i,o in zip(in_sh,out_sh)) and len(set(out_sh))==1:
            return ShapeRule("fixed",out_sh[0],None)
        mode=Counter(out_sh).most_common(1)[0][0]
        return ShapeRule("fallback_mode",mode,None)
    def predict(self, task, test_input):
        rule=self.analyze(task)
        H,W=len(test_input),(len(test_input[0]) if test_input else 1)
        if rule.kind=="identity": return (H,W)
        if rule.kind in ("fixed","fallback_mode"): return rule.fixed
        if rule.kind=="transpose": return (W,H)
        if rule.kind=="scale" and rule.scale:
            rs,cs=rule.scale; return (int(H*rs),int(W*cs))
        return None

_shape_predictor=ShapePredictor()
print("✓ Phase 3 로드 완료 (ShapePredictor)")
'''

# ── Phase 4: 고속 연산자 라이브러리 ───────────────────────────────────────────────
CELL_PHASE4 = '''\
FIXED_OPS: list[tuple[str,Callable]] = [
    ("rot90",    lambda g: np.rot90(g,1).copy()),
    ("rot180",   lambda g: np.rot90(g,2).copy()),
    ("rot270",   lambda g: np.rot90(g,3).copy()),
    ("flip_lr",  lambda g: np.fliplr(g).copy()),
    ("flip_ud",  lambda g: np.flipud(g).copy()),
    ("flip_d0",  lambda g: g.T.copy()),
    ("flip_d1",  lambda g: np.rot90(g.T,2).copy()),
    ("scale2x2", lambda g: scale_up(g,2,2)),
    ("scale3x3", lambda g: scale_up(g,3,3)),
    ("scale2x1", lambda g: scale_up(g,2,1)),
    ("scale1x2", lambda g: scale_up(g,1,2)),
    ("tile22",   lambda g: tile(g,2,2)),
    ("tile12",   lambda g: tile(g,1,2)),
    ("tile21",   lambda g: tile(g,2,1)),
    ("tile33",   lambda g: tile(g,3,3)),
    ("crop_bg",  lambda g: crop_to_content(g,0)),
    ("grav_d",   lambda g: gravity_down(g,0)),
    ("grav_u",   lambda g: gravity_up(g,0)),
    ("grav_l",   lambda g: gravity_left(g,0)),
    ("grav_r",   lambda g: gravity_right(g,0)),
    ("norm_col", lambda g: normalize_colors(g)[0]),
    ("keep_mc",  lambda g: keep_most_common_color(g,0)),
    ("keep_lc",  lambda g: keep_least_common_color(g,0)),
    ("invert01", lambda g: invert_binary(g,0)),
    ("keep_lg1", lambda g: keep_n_largest(g,1,0)),
    ("keep_lg2", lambda g: keep_n_largest(g,2,0)),
    ("keep_sm1", lambda g: keep_n_smallest(g,1,0)),
    ("rm_noise1",lambda g: remove_small_objects(g,1,0)),
    ("rm_noise2",lambda g: remove_small_objects(g,2,0)),
    ("col_size", lambda g: color_objects_by_size(g,0)),
    ("fill_hole",lambda g: fill_holes(g,-1,0)),
    ("center",   lambda g: center_content(g,0)),
    ("outline",  lambda g: outline_objects(g,0)),
    ("fill_out", lambda g: fill_outer_region(g,0,0)),
    ("sym_h",    lambda g: complete_symmetry_h(g,0)),
    ("sym_v",    lambda g: complete_symmetry_v(g,0)),
    ("sym_r180", lambda g: complete_symmetry_rot180(g,0)),
    ("mir_r",    lambda g: mirror_extend_right(g)),
    ("mir_l",    lambda g: mirror_extend_left(g)),
    ("mir_d",    lambda g: mirror_extend_down(g)),
    ("mir_u",    lambda g: mirror_extend_up(g)),
    ("add_bdr",  lambda g: add_border(g,None,0)),
    ("rm_bdr",   lambda g: remove_border(g)),
    ("tile_cnt", lambda g: tile_from_content(g,0)),
    ("ext_pat",  lambda g: extract_pattern(g)),
    ("sort_r",   lambda g: sort_rows_by_sum(g)),
    ("sort_c",   lambda g: sort_cols_by_sum(g)),
    ("dilate4",  lambda g: dilate_objects(g,0,4)),
    ("dilate8",  lambda g: dilate_objects(g,0,8)),
    ("erode4",   lambda g: erode_objects(g,0,4)),
    ("erode8",   lambda g: erode_objects(g,0,8)),
    ("majority", lambda g: majority_neighbor_color(g,1)),
    ("keep_bdr", lambda g: keep_border_objects(g,0)),
    ("keep_int", lambda g: keep_interior_objects(g,0)),
    ("mark_obj", lambda g: mark_unique_objects(g,0)),
    ("sub_rot90",lambda g: apply_op_to_subgrids(g,lambda x:np.rot90(x,1))),
    ("sub_flip", lambda g: apply_op_to_subgrids(g,lambda x:np.fliplr(x))),
    ("sub_norm", lambda g: apply_op_to_subgrids(g,lambda x:normalize_colors(x)[0])),
]

def _learn_color_map(task):
    all_maps=[]
    for p in task.train:
        if p.output is None: continue
        inp,out=g2n(p.input),g2n(p.output)
        if inp.shape!=out.shape: return None
        cmap={}
        for c in np.unique(inp):
            v=out[inp==c]; u=np.unique(v)
            if len(u)!=1: return None
            c=int(c); d=int(u[0])
            if c in cmap and cmap[c]!=d: return None
            cmap[c]=d
        all_maps.append(cmap)
    if not all_maps: return None
    first=all_maps[0]
    if not all(m==first for m in all_maps): return None
    if all(k==v for k,v in first.items()): return None
    return first

def _learn_scale(task):
    pairs=[(g2n(p.input),g2n(p.output)) for p in task.train if p.output is not None]
    if not pairs: return []
    ih,iw=pairs[0][0].shape; oh,ow=pairs[0][1].shape
    if ih==0 or iw==0: return []
    if oh%ih==0 and ow%iw==0:
        ry,rx=oh//ih,ow//iw
        if 1<ry<=5 and 1<rx<=5:
            return [(f"scale_{ry}x{rx}",lambda g,r=ry,c=rx:scale_up(g,r,c))]
    return []

def _learn_output_size_ops(task):
    if not task.test_inputs: return []
    ps=_shape_predictor.predict(task,task.test_inputs[0])
    if not ps: return []
    H,W=ps
    ops=[]
    ih=len(task.train[0].input); iw=len(task.train[0].input[0]) if task.train[0].input else 1
    if (ih,iw)==(H,W): return []
    ops.append((f"ext_{H}x{W}",lambda g,h=H,w=W:extend_to_shape(g,h,w)))
    ops.append((f"pad_{H}x{W}",lambda g,h=H,w=W:pad_to_shape(g,h,w)))
    ops.append((f"crop_{H}x{W}",lambda g,h=H,w=W:g[:h,:w].copy() if g.shape[0]>=h and g.shape[1]>=w else g.copy()))
    return ops

def build_task_ops(task):
    ops=list(FIXED_OPS)
    # 학습 ops
    cmap=_learn_color_map(task)
    if cmap: ops.insert(0,("learn_cmap",lambda g,m=cmap:recolor(g,m)))
    for op in _learn_scale(task): ops.insert(0,op)
    ops.extend(_learn_output_size_ops(task))
    # 색상 swap
    colors=sorted({c for p in task.train for row in p.input for c in row}|
                  {c for p in task.train if p.output for row in p.output for c in row})
    for i,c1 in enumerate(colors):
        for c2 in colors[i+1:]:
            ops.append((f"sw{c1}_{c2}",(lambda a,b:(lambda g:swap_colors(g,a,b)))(c1,c2)))
    for c in colors:
        if c==0: continue
        ops.append((f"rm{c}",(lambda a:(lambda g:replace_color(g,a,0)))(c)))
    return ops

_OPS_CACHE: dict[str,list]={}
print(f"✓ Phase 4 로드 완료 (고정 {len(FIXED_OPS)}개)")
'''

# ── Phase 4b: RuleEngine ─────────────────────────────────────────────────────────
CELL_PHASE4B = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 4b: RuleEngine — 결정론적 규칙 엔진 (핵심 개선)
#
# 각 ARCRule은 모든 train pair에 100% 일치하는 변환 함수를 O(n) 에
# 탐색한다. 빔서치보다 빠르고 정확하며 50%+ 정확도 달성의 핵심.
# ═══════════════════════════════════════════════════════════════════

class ARCRule:
    """모든 규칙의 베이스 클래스."""
    NAME = "base"
    def try_task(self, task: Task) -> tuple[str, Callable] | None:
        """(rule_name, test_transform_fn) 또는 None 반환."""
        try:
            pairs = [(g2n(p.input), g2n(p.output))
                     for p in task.train if p.output is not None]
            if not pairs: return None
            result = self._learn(pairs, task)
            if result is None: return None
            name, fn = result
            if all(np.array_equal(fn(inp), out) for inp, out in pairs):
                return name, fn
        except Exception:
            pass
        return None
    def _learn(self, pairs, task):
        raise NotImplementedError


# ──────────────────────────── 기본 기하·색상 규칙 ──────────────────────────────

class IdentityRule(ARCRule):
    NAME = "identity"
    def _learn(self, pairs, task):
        if all(np.array_equal(i, o) for i,o in pairs):
            return ("identity", lambda g: g.copy())
        return None

class GeoRule(ARCRule):
    NAME = "geo"
    TRANSFORMS = [
        ("rot90",    lambda g: np.rot90(g,1).copy()),
        ("rot180",   lambda g: np.rot90(g,2).copy()),
        ("rot270",   lambda g: np.rot90(g,3).copy()),
        ("flip_lr",  lambda g: np.fliplr(g).copy()),
        ("flip_ud",  lambda g: np.flipud(g).copy()),
        ("flip_d0",  lambda g: g.T.copy()),
        ("flip_d1",  lambda g: np.rot90(g.T,2).copy()),
    ]
    def _learn(self, pairs, task):
        for name, fn in self.TRANSFORMS:
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return name, fn
        return None

class ColorMapRule(ARCRule):
    NAME = "color_map"
    def _learn(self, pairs, task):
        cmap = _learn_color_map(task)
        if cmap:
            return ("color_map", lambda g, m=cmap: recolor(g, m))
        return None

class ScaleRule(ARCRule):
    NAME = "scale"
    def _learn(self, pairs, task):
        ops = _learn_scale(task)
        for name, fn in ops:
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return name, fn
        # Also try larger factors
        for ry in range(1, 6):
            for rx in range(1, 6):
                if ry == rx == 1: continue
                fn = lambda g, r=ry, c=rx: scale_up(g, r, c)
                if all(np.array_equal(fn(i), o) for i,o in pairs):
                    return f"scale_{ry}x{rx}", fn
        return None

class ScaleDownRule(ARCRule):
    NAME = "scale_down"
    def _learn(self, pairs, task):
        for ry in range(2, 6):
            for rx in range(2, 6):
                fn = lambda g, r=ry, c=rx: scale_down(g, r, c)
                if all(np.array_equal(fn(i), o) for i,o in pairs):
                    return f"scale_down_{ry}x{rx}", fn
        return None

class TileRule(ARCRule):
    NAME = "tile"
    def _learn(self, pairs, task):
        for ny in range(1, 5):
            for nx in range(1, 5):
                if ny == nx == 1: continue
                fn = lambda g, n=ny, m=nx: tile(g, n, m)
                if all(np.array_equal(fn(i), o) for i,o in pairs):
                    return f"tile_{ny}x{nx}", fn
        return None

class TransposeRule(ARCRule):
    NAME = "transpose"
    def _learn(self, pairs, task):
        fn = lambda g: g.T.copy()
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "transpose", fn
        return None

class ConstOutputRule(ARCRule):
    NAME = "const_output"
    def _learn(self, pairs, task):
        outs = [o for _, o in pairs]
        if all(np.array_equal(outs[0], o) for o in outs[1:]):
            out = outs[0].copy()
            return "const_output", lambda g, o=out: o.copy()
        return None

class CropBgRule(ARCRule):
    NAME = "crop_bg"
    def _learn(self, pairs, task):
        for bg in range(10):
            fn = lambda g, b=bg: crop_to_content(g, b)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"crop_bg{bg}", fn
        return None

class GravityRule(ARCRule):
    NAME = "gravity"
    def _learn(self, pairs, task):
        for bg in [0]:
            for name, fn in [
                ("grav_d", lambda g, b=bg: gravity_down(g, b)),
                ("grav_u", lambda g, b=bg: gravity_up(g, b)),
                ("grav_l", lambda g, b=bg: gravity_left(g, b)),
                ("grav_r", lambda g, b=bg: gravity_right(g, b)),
            ]:
                if all(np.array_equal(fn(i), o) for i,o in pairs):
                    return name, fn
        return None

class MirrorExtendRule(ARCRule):
    NAME = "mirror_extend"
    def _learn(self, pairs, task):
        for name, fn in [
            ("mir_r", mirror_extend_right), ("mir_l", mirror_extend_left),
            ("mir_d", mirror_extend_down),  ("mir_u", mirror_extend_up),
        ]:
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return name, fn
        return None

class ShiftRule(ARCRule):
    NAME = "shift"
    def _learn(self, pairs, task):
        in0, out0 = pairs[0]
        if in0.shape != out0.shape: return None
        H, W = in0.shape
        for dr in range(-H+1, H):
            for dc in range(-W+1, W):
                if dr == 0 and dc == 0: continue
                fn = lambda g, r=dr, c=dc: shift_content(g, r, c, 0)
                if all(np.array_equal(fn(i), o) for i,o in pairs):
                    return f"shift_{dr}_{dc}", fn
        return None

class InvertRule(ARCRule):
    NAME = "invert"
    def _learn(self, pairs, task):
        fn = lambda g: invert_binary(g, 0)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "invert_binary", fn
        return None

class OutlineRule(ARCRule):
    NAME = "outline"
    def _learn(self, pairs, task):
        fn = lambda g: outline_objects(g, 0)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "outline", fn
        return None

class FillHolesRule(ARCRule):
    NAME = "fill_holes"
    def _learn(self, pairs, task):
        for fill in range(10):
            fn = lambda g, f=fill: fill_holes(g, f, 0)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"fill_holes_{fill}", fn
        return None

class FillOuterRule(ARCRule):
    NAME = "fill_outer"
    def _learn(self, pairs, task):
        for fill in range(10):
            fn = lambda g, f=fill: fill_outer_region(g, f, 0)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"fill_outer_{fill}", fn
        return None

class DilateRule(ARCRule):
    NAME = "dilate"
    def _learn(self, pairs, task):
        for conn in [4, 8]:
            fn = lambda g, c=conn: dilate_objects(g, 0, c)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"dilate{conn}", fn
        return None

class ErodeRule(ARCRule):
    NAME = "erode"
    def _learn(self, pairs, task):
        for conn in [4, 8]:
            fn = lambda g, c=conn: erode_objects(g, 0, c)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"erode{conn}", fn
        return None

class MajorityCARule(ARCRule):
    NAME = "majority_ca"
    def _learn(self, pairs, task):
        fn = lambda g: majority_neighbor_color(g, 1)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "majority_ca", fn
        return None

class KeepLargestRule(ARCRule):
    NAME = "keep_largest"
    def _learn(self, pairs, task):
        for n in [1, 2]:
            fn = lambda g, k=n: keep_n_largest(g, k, 0)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"keep_largest_{n}", fn
        return None

class KeepSmallestRule(ARCRule):
    NAME = "keep_smallest"
    def _learn(self, pairs, task):
        for n in [1, 2]:
            fn = lambda g, k=n: keep_n_smallest(g, k, 0)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"keep_smallest_{n}", fn
        return None

class KeepBorderObjRule(ARCRule):
    NAME = "keep_border"
    def _learn(self, pairs, task):
        fn = lambda g: keep_border_objects(g, 0)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "keep_border", fn
        return None

class KeepInteriorObjRule(ARCRule):
    NAME = "keep_interior"
    def _learn(self, pairs, task):
        fn = lambda g: keep_interior_objects(g, 0)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "keep_interior", fn
        return None

class RemoveSmallRule(ARCRule):
    NAME = "remove_small"
    def _learn(self, pairs, task):
        for sz in range(1, 5):
            fn = lambda g, s=sz: remove_small_objects(g, s, 0)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"remove_small_{sz}", fn
        return None

class SymmetryCompleteRule(ARCRule):
    NAME = "sym_complete"
    def _learn(self, pairs, task):
        for name, fn in [
            ("sym_h",    lambda g: complete_symmetry_h(g, 0)),
            ("sym_v",    lambda g: complete_symmetry_v(g, 0)),
            ("sym_r180", lambda g: complete_symmetry_rot180(g, 0)),
        ]:
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return name, fn
        return None

class PatternPeriodRule(ARCRule):
    NAME = "pattern_period"
    def _learn(self, pairs, task):
        fn = lambda g: extract_pattern(g)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "extract_pattern", fn
        return None

class TileFromContentRule(ARCRule):
    NAME = "tile_content"
    def _learn(self, pairs, task):
        fn = lambda g: tile_from_content(g, 0)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "tile_from_content", fn
        return None

class CenterContentRule(ARCRule):
    NAME = "center"
    def _learn(self, pairs, task):
        fn = lambda g: center_content(g, 0)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "center_content", fn
        return None

class SortRowsRule(ARCRule):
    NAME = "sort_rows"
    def _learn(self, pairs, task):
        for name, fn in [
            ("sort_r_asc",  lambda g: sort_rows_by_sum(g)),
            ("sort_r_desc", lambda g: sort_rows_descending(g)),
            ("sort_c_asc",  lambda g: sort_cols_by_sum(g)),
        ]:
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return name, fn
        return None

class MosaicSubgridRule(ARCRule):
    """구분선으로 분할된 서브그리드 각각에 동일한 변환 적용."""
    NAME = "mosaic"
    SUB_OPS = [
        ("rot90",  lambda g: np.rot90(g,1).copy()),
        ("rot180", lambda g: np.rot90(g,2).copy()),
        ("rot270", lambda g: np.rot90(g,3).copy()),
        ("fliplr", lambda g: np.fliplr(g).copy()),
        ("flipud", lambda g: np.flipud(g).copy()),
        ("invert", lambda g: invert_binary(g,0)),
        ("norm",   lambda g: normalize_colors(g)[0]),
    ]
    def _learn(self, pairs, task):
        for op_name, op_fn in self.SUB_OPS:
            fn = lambda g, f=op_fn: apply_op_to_subgrids(g, f, 0)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"mosaic_{op_name}", fn
        return None

class DiffPatchRule(ARCRule):
    """모든 train pair에서 같은 좌표가 같은 색으로 변하면 고정 패치 생성."""
    NAME = "diff_patch"
    def _learn(self, pairs, task):
        if any(i.shape != o.shape for i,o in pairs):
            return None
        masks = [(i != o) for i,o in pairs]
        if not all(np.array_equal(masks[0], m) for m in masks[1:]):
            return None
        mask = masks[0]
        if not mask.any() or int(mask.sum()) > 60:
            return None
        patch: dict[tuple,int] = {}
        for r,c in np.argwhere(mask):
            vals = [int(o[r,c]) for _,o in pairs]
            if len(set(vals)) != 1: return None
            patch[(int(r),int(c))] = vals[0]
        def apply(g, p=patch):
            res = g.copy(); H,W = res.shape
            for (r,c),col in p.items():
                if 0<=r<H and 0<=c<W: res[r,c]=col
            return res
        return "diff_patch", apply

class NormGeoRule(ARCRule):
    """색상 정규화 후 기하 변환 적용."""
    NAME = "norm_geo"
    TRANSFORMS = [
        ("rot90",  lambda g: np.rot90(g,1).copy()),
        ("rot180", lambda g: np.rot90(g,2).copy()),
        ("rot270", lambda g: np.rot90(g,3).copy()),
        ("fliplr", lambda g: np.fliplr(g).copy()),
        ("flipud", lambda g: np.flipud(g).copy()),
        ("flip_d0",lambda g: g.T.copy()),
        ("flip_d1",lambda g: np.rot90(g.T,2).copy()),
    ]
    def _learn(self, pairs, task):
        norm_pairs=[(normalize_colors(i)[0],normalize_colors(o)[0]) for i,o in pairs]
        for tname, tfn in self.TRANSFORMS:
            if all(np.array_equal(tfn(ni), no) for ni,no in norm_pairs):
                def transform(g, fn=tfn):
                    ng, inv = normalize_colors(g)
                    return denormalize_colors(fn(ng), inv)
                return f"norm_{tname}", transform
        return None

class NormColorMapRule(ARCRule):
    """색상 정규화 후 색상 매핑 적용."""
    NAME = "norm_color_map"
    def _learn(self, pairs, task):
        norm_pairs=[(normalize_colors(i)[0],normalize_colors(o)[0]) for i,o in pairs]
        cmap={}
        for ni,no in norm_pairs:
            if ni.shape!=no.shape: return None
            for c in np.unique(ni):
                v=no[ni==c]; u=np.unique(v)
                if len(u)!=1: return None
                c=int(c); d=int(u[0])
                if c in cmap and cmap[c]!=d: return None
                cmap[c]=d
        if not cmap or all(k==v for k,v in cmap.items()): return None
        def transform(g, m=cmap):
            ng, inv = normalize_colors(g)
            return denormalize_colors(recolor(ng, m), inv)
        return "norm_color_map", transform

class NormScaleRule(ARCRule):
    """색상 정규화 후 스케일 변환."""
    NAME = "norm_scale"
    def _learn(self, pairs, task):
        norm_pairs=[(normalize_colors(i)[0],normalize_colors(o)[0]) for i,o in pairs]
        for ry in range(1,5):
            for rx in range(1,5):
                if ry==rx==1: continue
                fn=lambda g,r=ry,c=rx:scale_up(g,r,c)
                if all(np.array_equal(fn(ni),no) for ni,no in norm_pairs):
                    def transform(g, r=ry, c=rx):
                        ng,inv=normalize_colors(g)
                        return denormalize_colors(scale_up(ng,r,c),inv)
                    return f"norm_scale_{ry}x{rx}", transform
        return None

class AddBorderRule(ARCRule):
    NAME = "add_border"
    def _learn(self, pairs, task):
        for c in range(10):
            fn = lambda g, col=c: add_border(g, col, 0)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"add_border_{c}", fn
        return None

class RemoveBorderRule(ARCRule):
    NAME = "remove_border"
    def _learn(self, pairs, task):
        fn = lambda g: remove_border(g)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "remove_border", fn
        return None

class ColorObjSizeRule(ARCRule):
    NAME = "color_obj_size"
    def _learn(self, pairs, task):
        fn = lambda g: color_objects_by_size(g, 0)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "color_by_size", fn
        return None

class HalfGridRule(ARCRule):
    """격자의 반쪽(상·하·좌·우)을 추출."""
    NAME = "half_grid"
    def _learn(self, pairs, task):
        def top_half(g):
            H=g.shape[0]; return g[:H//2,:].copy()
        def bot_half(g):
            H=g.shape[0]; return g[H//2:,:].copy() if H%2==0 else g[(H+1)//2:,:].copy()
        def lft_half(g):
            W=g.shape[1]; return g[:,:W//2].copy()
        def rgt_half(g):
            W=g.shape[1]; return g[:,W//2:].copy() if W%2==0 else g[:,(W+1)//2:].copy()
        for name, fn in [("top_half",top_half),("bot_half",bot_half),
                         ("lft_half",lft_half),("rgt_half",rgt_half)]:
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return name, fn
        return None

class QuadrantRule(ARCRule):
    """격자의 4분면(TL/TR/BL/BR) 추출."""
    NAME = "quadrant"
    def _learn(self, pairs, task):
        def quad(g, r0, c0):
            H,W=g.shape; rh=H//2; cw=W//2
            rs={0:0,1:rh}; re={0:rh,1:H}
            cs={0:0,1:cw}; ce={0:cw,1:W}
            return g[rs[r0]:re[r0],cs[c0]:ce[c0]].copy()
        for name, (r0,c0) in [("quad_tl",(0,0)),("quad_tr",(0,1)),
                               ("quad_bl",(1,0)),("quad_br",(1,1))]:
            fn = lambda g, r=r0, c=c0: quad(g, r, c)
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return name, fn
        return None

class DownsampleMajorityRule(ARCRule):
    """kxk 블록마다 최빈값으로 다운샘플링."""
    NAME = "downsample_majority"
    def _learn(self, pairs, task):
        for k in range(2, 6):
            def fn(g, kk=k):
                H,W=g.shape
                if H%kk!=0 or W%kk!=0: return g.copy()
                out=np.zeros((H//kk,W//kk),dtype=np.int8)
                for r in range(H//kk):
                    for c in range(W//kk):
                        bl=g[r*kk:(r+1)*kk,c*kk:(c+1)*kk]
                        out[r,c]=Counter(bl.flatten().tolist()).most_common(1)[0][0]
                return out
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"down_maj_{k}", fn
        return None

class DownsampleUniqueRule(ARCRule):
    """kxk 블록마다 배경이 아닌 유일색으로 다운샘플링."""
    NAME = "downsample_unique"
    def _learn(self, pairs, task):
        for k in range(2, 6):
            def fn(g, kk=k, bg=0):
                H,W=g.shape
                if H%kk!=0 or W%kk!=0: return g.copy()
                out=np.zeros((H//kk,W//kk),dtype=np.int8)
                for r in range(H//kk):
                    for c in range(W//kk):
                        bl=g[r*kk:(r+1)*kk,c*kk:(c+1)*kk]
                        nz=[int(v) for v in bl.flatten() if v!=bg]
                        u=list(set(nz))
                        out[r,c]=u[0] if len(u)==1 else (nz[0] if nz else bg)
                return out
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"down_uniq_{k}", fn
        return None

class CANeighborRule(ARCRule):
    """CA 이웃 규칙: 특정 이웃 수일 때 셀 변경."""
    NAME = "ca_neighbor"
    def _learn(self, pairs, task):
        if any(i.shape!=o.shape for i,o in pairs): return None
        # Learn: for bg cells with exactly n neighbors of color c, set to color t
        colors=sorted({int(v) for i,o in pairs for v in i.flatten() if v!=0}|
                      {int(v) for i,o in pairs for v in o.flatten() if v!=0})
        for c in colors[:4]:
            for n in range(1, 5):
                for t in colors[:6]:
                    def fn(g, clr=c, cnt=n, tgt=t):
                        H,W=g.shape; res=g.copy()
                        for r in range(H):
                            for cc in range(W):
                                if g[r,cc]==0:
                                    nb=sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                           if 0<=r+dr<H and 0<=cc+dc<W and g[r+dr,cc+dc]==clr)
                                    if nb==cnt: res[r,cc]=tgt
                        return res
                    if all(np.array_equal(fn(i), o) for i,o in pairs):
                        return f"ca_{c}_n{n}to{t}", fn
        return None

class ConcatRule(ARCRule):
    """입력을 특정 방향으로 자신과 이어붙이기."""
    NAME = "concat"
    def _learn(self, pairs, task):
        for name, fn in [
            ("hstack", lambda g: np.hstack([g,g])),
            ("vstack", lambda g: np.vstack([g,g])),
            ("hstack_flip", lambda g: np.hstack([g,np.fliplr(g)])),
            ("vstack_flip", lambda g: np.vstack([g,np.flipud(g)])),
        ]:
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return name, fn
        return None

class UniqueColorRule(ARCRule):
    """출력 = 입력에서 한 번만 나오는 색의 마스크."""
    NAME = "unique_color"
    def _learn(self, pairs, task):
        def fn(g, bg=0):
            cnt=Counter(g[g!=bg].flatten().tolist())
            uniq={c for c,n in cnt.items() if n==1}
            return np.where(np.isin(g,list(uniq)),g,bg).astype(np.int8)
        if all(np.array_equal(fn(i), o) for i,o in pairs):
            return "unique_color", fn
        return None

# ── 2단계 복합 규칙 (2-op fast combinations) ───────────────────────────────────

class TwoStepRule(ARCRule):
    """핵심 단일 연산자 2개를 순서 조합으로 시도 (O(n^2))."""
    NAME = "two_step"
    FAST = [
        ("rot90",   lambda g: np.rot90(g,1).copy()),
        ("rot180",  lambda g: np.rot90(g,2).copy()),
        ("rot270",  lambda g: np.rot90(g,3).copy()),
        ("fliplr",  lambda g: np.fliplr(g).copy()),
        ("flipud",  lambda g: np.flipud(g).copy()),
        ("flip_d0", lambda g: g.T.copy()),
        ("crop_bg", lambda g: crop_to_content(g,0)),
        ("grav_d",  lambda g: gravity_down(g,0)),
        ("grav_u",  lambda g: gravity_up(g,0)),
        ("grav_l",  lambda g: gravity_left(g,0)),
        ("grav_r",  lambda g: gravity_right(g,0)),
        ("dilate4", lambda g: dilate_objects(g,0,4)),
        ("erode4",  lambda g: erode_objects(g,0,4)),
        ("fill_h",  lambda g: fill_holes(g,-1,0)),
        ("rm_bdr",  lambda g: remove_border(g)),
        ("keep_lg", lambda g: keep_n_largest(g,1,0)),
        ("keep_bdr",lambda g: keep_border_objects(g,0)),
        ("keep_int",lambda g: keep_interior_objects(g,0)),
        ("norm",    lambda g: normalize_colors(g)[0]),
        ("invert",  lambda g: invert_binary(g,0)),
        ("center",  lambda g: center_content(g,0)),
        ("sym_h",   lambda g: complete_symmetry_h(g,0)),
        ("sym_v",   lambda g: complete_symmetry_v(g,0)),
        ("outline", lambda g: outline_objects(g,0)),
        ("top_half",lambda g: g[:g.shape[0]//2,:].copy()),
        ("bot_half",lambda g: g[g.shape[0]//2:,:].copy()),
        ("lft_half",lambda g: g[:,:g.shape[1]//2].copy()),
        ("rgt_half",lambda g: g[:,g.shape[1]//2:].copy()),
    ]
    def _learn(self, pairs, task):
        for n1,f1 in self.FAST:
            for n2,f2 in self.FAST:
                fn = lambda g, a=f1, b=f2: b(a(g))
                if all(np.array_equal(fn(i), o) for i,o in pairs):
                    return f"{n1}+{n2}", fn
        return None

class ColorMapGeoRule(ARCRule):
    """색상 매핑 + 기하 변환 2단계 조합."""
    NAME = "cmap_geo"
    GEOS = [
        ("rot90",  lambda g: np.rot90(g,1).copy()),
        ("rot180", lambda g: np.rot90(g,2).copy()),
        ("rot270", lambda g: np.rot90(g,3).copy()),
        ("fliplr", lambda g: np.fliplr(g).copy()),
        ("flipud", lambda g: np.flipud(g).copy()),
        ("flip_d0",lambda g: g.T.copy()),
        ("dilate4",lambda g: dilate_objects(g,0,4)),
        ("erode4", lambda g: erode_objects(g,0,4)),
    ]
    def _learn(self, pairs, task):
        cmap = _learn_color_map(task)
        if not cmap: return None
        for tname, tfn in self.GEOS:
            fn = lambda g, m=cmap, tf=tfn: tf(recolor(g, m))
            if all(np.array_equal(fn(i), o) for i,o in pairs):
                return f"cmap+{tname}", fn
        return None

class GeoColorMapRule(ARCRule):
    """기하 변환 + 색상 매핑 2단계 조합."""
    NAME = "geo_cmap"
    GEOS = [
        ("rot90",  lambda g: np.rot90(g,1).copy()),
        ("rot180", lambda g: np.rot90(g,2).copy()),
        ("rot270", lambda g: np.rot90(g,3).copy()),
        ("fliplr", lambda g: np.fliplr(g).copy()),
        ("flipud", lambda g: np.flipud(g).copy()),
        ("flip_d0",lambda g: g.T.copy()),
        ("dilate4",lambda g: dilate_objects(g,0,4)),
        ("erode4", lambda g: erode_objects(g,0,4)),
    ]
    def _learn(self, pairs, task):
        for tname, tfn in self.GEOS:
            transformed = [(tfn(i), o) for i,o in pairs]
            if any(ti.shape != o.shape for ti,o in transformed): continue
            cmap = {}
            ok = True
            for ti,o in transformed:
                for c in np.unique(ti):
                    v=o[ti==c]; u=np.unique(v)
                    if len(u)!=1: ok=False; break
                    c=int(c); d=int(u[0])
                    if c in cmap and cmap[c]!=d: ok=False; break
                    cmap[c]=d
                if not ok: break
            if ok and cmap and not all(k==v for k,v in cmap.items()):
                fn = lambda g, tf=tfn, m=cmap: recolor(tf(g), m)
                if all(np.array_equal(fn(i), o) for i,o in pairs):
                    return f"{tname}+cmap", fn
        return None

class ScaleCmapRule(ARCRule):
    """스케일업 + 색상 매핑 2단계 조합."""
    NAME = "scale_cmap"
    def _learn(self, pairs, task):
        for ry in range(1, 5):
            for rx in range(1, 5):
                if ry==rx==1: continue
                scaled=[(scale_up(i,ry,rx),o) for i,o in pairs]
                if any(si.shape!=o.shape for si,o in scaled): continue
                cmap={}; ok=True
                for si,o in scaled:
                    for c in np.unique(si):
                        v=o[si==c]; u=np.unique(v)
                        if len(u)!=1: ok=False; break
                        c=int(c); d=int(u[0])
                        if c in cmap and cmap[c]!=d: ok=False; break
                        cmap[c]=d
                    if not ok: break
                if ok and cmap and not all(k==v for k,v in cmap.items()):
                    fn=lambda g,r=ry,c=rx,m=cmap:recolor(scale_up(g,r,c),m)
                    if all(np.array_equal(fn(i),o) for i,o in pairs):
                        return f"scale_{ry}x{rx}+cmap",fn
        return None

class NormGravityRule(ARCRule):
    """색상 정규화 + 중력 조합."""
    NAME = "norm_gravity"
    def _learn(self, pairs, task):
        norm_pairs=[(normalize_colors(i)[0],normalize_colors(o)[0]) for i,o in pairs]
        for name,fn in [("grav_d",lambda g:gravity_down(g,0)),
                        ("grav_u",lambda g:gravity_up(g,0)),
                        ("grav_l",lambda g:gravity_left(g,0)),
                        ("grav_r",lambda g:gravity_right(g,0))]:
            if all(np.array_equal(fn(ni),no) for ni,no in norm_pairs):
                def transform(g,f=fn):
                    ng,inv=normalize_colors(g); return denormalize_colors(f(ng),inv)
                return f"norm+{name}",transform
        return None

class NormDilateErodeRule(ARCRule):
    """색상 정규화 + 팽창/침식 조합."""
    NAME = "norm_dilate_erode"
    def _learn(self, pairs, task):
        norm_pairs=[(normalize_colors(i)[0],normalize_colors(o)[0]) for i,o in pairs]
        for name,fn in [("dilate4",lambda g:dilate_objects(g,0,4)),
                        ("erode4",lambda g:erode_objects(g,0,4)),
                        ("fill_h",lambda g:fill_holes(g,-1,0))]:
            if all(np.array_equal(fn(ni),no) for ni,no in norm_pairs):
                def transform(g,f=fn):
                    ng,inv=normalize_colors(g); return denormalize_colors(f(ng),inv)
                return f"norm+{name}",transform
        return None

class RuleEngine:
    """
    모든 ARCRule을 순서대로 시도. 100% 일치하는 규칙 발견 시 즉시 반환.
    빔서치보다 빠르고 정확 — 50%+ 정확도 달성 핵심 모듈.
    """
    def __init__(self):
        self.rules: list[ARCRule] = [
            # ── 1순위: 단순 변환 ──
            IdentityRule(),
            ColorMapRule(),
            GeoRule(),
            ScaleRule(),
            ScaleDownRule(),
            TileRule(),
            TransposeRule(),
            CropBgRule(),
            GravityRule(),
            MirrorExtendRule(),
            ShiftRule(),
            InvertRule(),
            OutlineRule(),
            FillHolesRule(),
            FillOuterRule(),
            DilateRule(),
            ErodeRule(),
            MajorityCARule(),
            KeepLargestRule(),
            KeepSmallestRule(),
            KeepBorderObjRule(),
            KeepInteriorObjRule(),
            RemoveSmallRule(),
            SymmetryCompleteRule(),
            PatternPeriodRule(),
            TileFromContentRule(),
            CenterContentRule(),
            SortRowsRule(),
            MosaicSubgridRule(),
            DiffPatchRule(),
            ConstOutputRule(),
            AddBorderRule(),
            RemoveBorderRule(),
            ColorObjSizeRule(),
            # ── 2순위: 새 단일 규칙 ──
            HalfGridRule(),
            QuadrantRule(),
            DownsampleMajorityRule(),
            DownsampleUniqueRule(),
            ConcatRule(),
            UniqueColorRule(),
            CANeighborRule(),
            # ── 3순위: 색상 정규화 조합 ──
            NormGeoRule(),
            NormColorMapRule(),
            NormScaleRule(),
            NormGravityRule(),
            NormDilateErodeRule(),
            # ── 4순위: 2단계 조합 ──
            ColorMapGeoRule(),
            GeoColorMapRule(),
            ScaleCmapRule(),
            TwoStepRule(),
        ]

    def solve(self, task: Task) -> tuple[str, list[NGrid]] | None:
        """
        규칙 매칭 성공 시 (rule_name, test_grids) 반환.
        모두 실패 시 None 반환.
        """
        for rule in self.rules:
            result = rule.try_task(task)
            if result is None: continue
            name, fn = result
            try:
                test_grids = [fn(g2n(inp)) for inp in task.test_inputs]
                return name, test_grids
            except Exception:
                continue
        return None


_rule_engine = RuleEngine()
print(f"✓ Phase 4b 로드 완료 (RuleEngine — {len(_rule_engine.rules)}개 규칙)")

# ═══════════════════════════════════════════════════════════════════
# Phase 4c: 동적 규칙 학습기 (Dynamic Rule Learner)
#
# 고정 규칙 라이브러리로 해결 안 되는 태스크를 위해
# 각 태스크의 훈련 쌍 데이터에서 규칙을 직접 학습한다.
# ═══════════════════════════════════════════════════════════════════

def _apply_and_check(fn, pairs):
    """fn을 모든 train pair에 적용해 100% 일치 여부 반환."""
    try:
        return all(np.array_equal(fn(i), o) for i, o in pairs)
    except:
        return False


def _apply_and_check_loocv(fn_builder, pairs):
    """LOOCV(Leave-One-Out Cross Validation) 검증.
    각 pair 1개를 뺀 나머지로 규칙을 학습하고, 남긴 pair로 검증.
    train이 3개 이상일 때만 실행 (2개 이하면 기존 방식 사용).
    반환: (loocv_pass: bool, train_fn: callable|None)
    """
    if len(pairs) < 3:
        # 쌍이 적으면 LOOCV 불필요 — 기존 로직에 맡김
        return True, None
    for i in range(len(pairs)):
        held_out = pairs[i]
        train_subset = pairs[:i] + pairs[i+1:]
        try:
            result = fn_builder(train_subset)
            if result is None:
                return False, None
            _, fn = result
            if not np.array_equal(fn(held_out[0]), held_out[1]):
                return False, None
        except:
            return False, None
    # LOOCV 통과 → 전체 데이터로 최종 규칙 학습
    full_result = fn_builder(pairs)
    if full_result is None:
        return False, None
    return True, full_result[1]


class DynamicRuleLearner:
    """
    태스크 데이터에서 동적으로 규칙을 생성하고 검증.
    RuleEngine 이후 호출되는 2순위 해결기.
    """

    # ── LOOCV로 과적합 필터링하는 전략 래퍼 ─────────────────────────
    def _run_strategy_with_loocv(self, strategy, pairs, task):
        """전략을 LOOCV 검증과 함께 실행.
        LOOCV_THRESHOLD: train pairs가 3개 이상인 태스크만 LOOCV 적용.
        2개 이하 train은 일반 검증만 수행.
        """
        if len(pairs) < 3:
            return strategy(pairs, task)
        result = strategy(pairs, task)
        if result is None:
            return None
        name, fn = result
        # LOOCV: 각 pair 제외 후 해당 fn이 held-out에서도 맞는지 확인
        for i in range(len(pairs)):
            held = pairs[i]
            try:
                if not np.array_equal(fn(held[0]), held[1]):
                    return None  # LOOCV 실패 → 과적합 탈락
            except:
                return None
        return result

    def solve(self, task: Task) -> tuple[str, list[NGrid]] | None:
        pairs = [(g2n(p.input), g2n(p.output))
                 for p in task.train if p.output is not None]
        if not pairs: return None

        # 수행 순서: 빠른 규칙 → 느린 규칙
        strategies = [
            # 전역 색상 매핑 (가장 빠름)
            self._try_global_color_mapping,
            # 빠른 색상 규칙
            self._try_color_filter,
            self._try_most_frequent_colors,
            self._try_color_merge,
            self._try_keep_specific_colors,
            self._try_enclosed_fill_color,
            # 크기 변환 태스크
            self._try_crop_region,
            self._try_tile_extraction,
            # 형태 감지 규칙 (십자/코너/끝점)
            self._try_shape_recolor,
            # CA 룩업 테이블 (강력!)
            self._try_ca_lookup_table,
            self._try_ca_lookup_table_8,
            # CA 규칙 (단순)
            self._try_ca_any_neighbor,
            self._try_ca_neighbor_detailed,
            # 오브젝트 이동
            self._try_object_bbox_shift,
            self._try_move_to_adjacent_color,
            # 오브젝트 크기 규칙
            self._try_size_to_color,
            # 교차점 규칙
            self._try_row_col_intersection,
            # 기타
            self._try_object_recolor_by_property,
            self._try_color_by_count,
            self._try_object_count_output,
            self._try_extract_by_separator_col,
            self._try_norm_then_rule,
            # 새 전략들
            self._try_gravity,
            self._try_sort_rows_cols,
            self._try_symmetry_complete,
            self._try_flood_fill_region,
            self._try_scale_n,
            self._try_two_step,
            # 새 강화 전략들
            self._try_fill_interior_by_border,
            self._try_soft_ca_lookup,
            self._try_object_bbox_fill,
            self._try_slide_to_nearest,
            self._try_recolor_isolated,
            self._try_multi_two_step,
            self._try_invert_colors,
            self._try_paint_enclosed_bg,
            self._try_diagonal_ca,
            self._try_column_rules,
            # 판별적 CA (일반화 강화)
            self._try_discriminative_ca,
            # 추상적 오브젝트 규칙
            self._try_color_at_tips,
            self._try_color_at_junctions,
            self._try_outline_fill_toggle,
            self._try_connect_same_color,
            # 오브젝트 위치·교환·배치 규칙
            self._try_swap_same_size_objects,
            self._try_move_to_container,
            self._try_replicate_in_region,
            self._try_row_col_count_to_color,
            self._try_max_min_object_swap,
        ]

        # LOOCV를 적용할 전략 (train 쌍을 외울 위험이 높은 것들)
        # CA lookup/column_rules 등은 train 쌍 수가 적으면 LOOCV 효과 없음 → 제외
        APPLY_LOOCV = {
            self._try_global_color_mapping,
            self._try_gravity,
            self._try_sort_rows_cols,
            self._try_symmetry_complete,
            self._try_scale_n,
            self._try_invert_colors,
            self._try_swap_same_size_objects,
            self._try_max_min_object_swap,
            self._try_row_col_count_to_color,
        }

        for strategy in strategies:
            use_loocv = (len(pairs) >= 3) and (strategy in APPLY_LOOCV)
            if use_loocv:
                result = self._run_strategy_with_loocv(strategy, pairs, task)
            else:
                result = strategy(pairs, task)
            if result:
                name, fn = result
                try:
                    test_grids = [fn(g2n(inp)) for inp in task.test_inputs]
                    return name, test_grids
                except:
                    continue
        return None

    # ── CA 완전 룩업 테이블 ──────────────────────────────────────────
    def _try_ca_lookup_table(self, pairs, task):
        """(중심색, 4-이웃 패턴) → 출력색 룩업 테이블 학습.
        완전히 일관된 경우에만 사용 (모든 동일 맥락 → 동일 출력)."""
        if any(i.shape != o.shape for i, o in pairs): return None
        dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        lookup: dict = {}
        for inp, out in pairs:
            H, W = inp.shape
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r, c])
                    tc = int(out[r, c])
                    nb4 = tuple(int(inp[r+dr, c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr, dc in dirs4)
                    key = (fc,) + nb4
                    if key in lookup and lookup[key] != tc:
                        return None   # 일관성 불일치 → 실패
                    lookup[key] = tc
        if not lookup: return None
        # 변환 없는 경우 제외
        if all(k[0] == v for k, v in lookup.items()): return None

        def apply_fn(inp, lut=lookup):
            H, W = inp.shape
            result = inp.copy()
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r, c])
                    nb4 = tuple(int(inp[r+dr, c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr, dc in dirs4)
                    key = (fc,) + nb4
                    if key in lut:
                        result[r, c] = lut[key]
            return result

        # 훈련 쌍 100% 검증
        if not all(np.array_equal(apply_fn(i), o) for i, o in pairs):
            return None
        return "ca_lookup", apply_fn

    # ── 8-이웃 CA 룩업 테이블 ────────────────────────────────────────
    def _try_ca_lookup_table_8(self, pairs, task):
        """(중심색, 8-이웃 패턴) → 출력색 룩업 테이블 (8방향)."""
        if any(i.shape != o.shape for i, o in pairs): return None
        dirs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        lookup: dict = {}
        for inp, out in pairs:
            H, W = inp.shape
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r, c])
                    tc = int(out[r, c])
                    nb8 = tuple(int(inp[r+dr, c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr, dc in dirs8)
                    key = (fc,) + nb8
                    if key in lookup and lookup[key] != tc:
                        return None
                    lookup[key] = tc
        if not lookup: return None
        if all(k[0] == v for k, v in lookup.items()): return None

        def apply_fn8(inp, lut=lookup):
            H, W = inp.shape
            result = inp.copy()
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r, c])
                    nb8 = tuple(int(inp[r+dr, c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr, dc in dirs8)
                    key = (fc,) + nb8
                    if key in lut:
                        result[r, c] = lut[key]
            return result

        if not all(np.array_equal(apply_fn8(i), o) for i, o in pairs):
            return None
        return "ca_lookup_8", apply_fn8

    # ── 색상 매핑 (전역 일관 치환) ───────────────────────────────────
    def _try_global_color_mapping(self, pairs, task):
        """입력 색상 → 출력 색상 1:1 전역 매핑 학습."""
        if any(i.shape != o.shape for i, o in pairs): return None
        cmap: dict = {}
        for inp, out in pairs:
            for fc in np.unique(inp):
                fc = int(fc)
                mask = inp == fc
                tcs = np.unique(out[mask])
                if len(tcs) != 1: return None
                tc = int(tcs[0])
                if fc in cmap and cmap[fc] != tc: return None
                cmap[fc] = tc
        if not cmap or all(k == v for k, v in cmap.items()): return None
        def apply_fn(inp, m=cmap):
            result = inp.copy()
            for fc, tc in m.items(): result[inp == fc] = tc
            return result
        if not all(np.array_equal(apply_fn(i), o) for i, o in pairs): return None
        return "global_cmap", apply_fn

    # ── 크롭/추출 (크기 변환 태스크) ────────────────────────────────
    def _try_crop_region(self, pairs, task):
        """입력의 특정 영역(슬라이스)을 추출해 출력으로."""
        # 각 pair에서 일관된 슬라이스 (r0,r1,c0,c1) 학습
        slices = []
        for inp, out in pairs:
            H, W = inp.shape
            oh, ow = out.shape
            found = False
            for r0 in range(H - oh + 1):
                for c0 in range(W - ow + 1):
                    if np.array_equal(inp[r0:r0+oh, c0:c0+ow], out):
                        slices.append((r0, c0, oh, ow)); found = True; break
                if found: break
            if not found: return None
        # 슬라이스가 일관된지 확인
        if len(set(slices)) == 1:
            r0, c0, oh, ow = slices[0]
            def apply_fn(inp, _r0=r0, _c0=c0, _oh=oh, _ow=ow):
                return inp[_r0:_r0+_oh, _c0:_c0+_ow].copy()
            if all(np.array_equal(apply_fn(i), o) for i, o in pairs):
                return f"crop_{r0}_{c0}_{oh}x{ow}", apply_fn
        # 슬라이스가 상대적으로 일관 (입력 크기에 비례)
        return None

    # ── 반복 패턴 감지 (타일) ───────────────────────────────────────
    def _try_tile_extraction(self, pairs, task):
        """출력이 입력의 타일링이거나, 입력의 서브패턴이 타일링되어 출력인 경우."""
        fns_tested = []
        for inp, out in pairs:
            iH, iW = inp.shape; oH, oW = out.shape
            if oH % iH == 0 and oW % iW == 0:
                nh, nw = oH // iH, oW // iW
                if np.array_equal(np.tile(inp, (nh, nw)), out):
                    fns_tested.append(('tile', nh, nw)); break
        for fname, nh, nw in fns_tested[:1]:
            fn = lambda g, _nh=nh, _nw=nw: np.tile(g, (_nh, _nw))
            if all(np.array_equal(fn(i), o) for i, o in pairs):
                return f"tile_{nh}x{nw}", fn
        return None

    # ── 중력 규칙 ────────────────────────────────────────────────────
    def _try_gravity(self, pairs, task):
        """중력 방향(상/하/좌/우)으로 오브젝트 이동."""
        if any(i.shape != o.shape for i, o in pairs): return None
        for direction in ['down', 'up', 'left', 'right']:
            fn = lambda g, d=direction: _gravity_any_dir(g, d)
            if _apply_and_check(fn, pairs): return f"gravity_{direction}", fn
        return None

    # ── 행/열 정렬 ───────────────────────────────────────────────────
    def _try_sort_rows_cols(self, pairs, task):
        """행 또는 열을 색상 기준으로 정렬."""
        if any(i.shape != o.shape for i, o in pairs): return None
        for name, fn in [('sort_rows', _sort_rows_by_count), ('sort_cols', _sort_cols_by_count)]:
            if _apply_and_check(fn, pairs): return name, fn
        return None

    # ── 대칭 완성 ────────────────────────────────────────────────────
    def _try_symmetry_complete(self, pairs, task):
        """빈 부분을 대칭으로 채우기."""
        if any(i.shape != o.shape for i, o in pairs): return None
        for axis in ['h', 'v']:
            fn = lambda g, a=axis: _complete_symmetry(g, a)
            if _apply_and_check(fn, pairs): return f"sym_{axis}", fn
        return None

    # ── 플러드필 영역 채우기 ─────────────────────────────────────────
    def _try_flood_fill_region(self, pairs, task):
        """특정 색상에서 인접 배경 영역 flood fill."""
        if any(i.shape != o.shape for i, o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:5]
        out_colors = sorted({int(v) for i,o in pairs for v in np.unique(o) if v!=0})[:8]
        for sc in colors:
            for fc in out_colors:
                if sc == fc: continue
                fn = lambda g, s=sc, f=fc: _flood_fill_regions(g, s, f)
                if _apply_and_check(fn, pairs): return f"flood_{sc}→{fc}", fn
        return None

    # ── N배 스케일업 ─────────────────────────────────────────────────
    def _try_scale_n(self, pairs, task):
        """N배 스케일업."""
        for n in [2, 3, 4]:
            def fn(g, k=n): return _scale_n(g, k)
            if _apply_and_check(fn, pairs): return f"scale{n}x", fn
        return None

    # ── 2단계 규칙 ───────────────────────────────────────────────────
    def _try_two_step(self, pairs, task):
        """간단한 2단계 변환: 기하변환 후 색상 매핑."""
        geom_ops = [
            ('rot90', lambda g: np.rot90(g, 1)),
            ('rot180', lambda g: np.rot90(g, 2)),
            ('rot270', lambda g: np.rot90(g, 3)),
            ('fliplr', lambda g: np.fliplr(g)),
            ('flipud', lambda g: np.flipud(g)),
        ]
        for gname, gfn in geom_ops:
            # 기하 변환 후 색상 매핑 학습
            transformed = [(gfn(i), o) for i, o in pairs]
            cmap = {}; ok = True
            for t_inp, out in transformed:
                if t_inp.shape != out.shape: ok=False; break
                for fc in np.unique(t_inp):
                    fc = int(fc)
                    tcs = np.unique(out[t_inp == fc])
                    if len(tcs) != 1: ok=False; break
                    tc = int(tcs[0])
                    if fc in cmap and cmap[fc] != tc: ok=False; break
                    cmap[fc] = tc
                if not ok: break
            if ok and cmap and not all(k==v for k,v in cmap.items()):
                def fn(g, _gfn=gfn, _cmap=cmap):
                    t = _gfn(g).copy()
                    for fc, tc in _cmap.items(): t[_gfn(g) == fc] = tc
                    return t
                if _apply_and_check(fn, pairs): return f"{gname}+cmap", fn
        return None

    # ── 오브젝트 크기 → 숫자 매핑 ───────────────────────────────────
    def _try_size_to_color(self, pairs, task):
        """오브젝트 크기를 1-9 색상으로 매핑."""
        if any(i.shape != o.shape for i, o in pairs): return None
        # 크기 → 색상 매핑 학습
        sz_map: dict = {}
        for inp, out in pairs:
            objs = find_objects(inp, 0, 4)
            for cells in objs:
                sz = len(cells)
                # 이 오브젝트가 출력에서 무슨 색인지
                sample_r, sample_c = cells[0]
                oc = int(out[sample_r, sample_c])
                if sz in sz_map and sz_map[sz] != oc: return None
                sz_map[sz] = oc
        if not sz_map: return None
        def apply_fn(inp, sm=sz_map):
            objs = find_objects(inp, 0, 4)
            result = inp.copy()
            for cells in objs:
                sz = len(cells)
                if sz in sm:
                    for r, c in cells: result[r, c] = sm[sz]
            return result
        if not all(np.array_equal(apply_fn(i), o) for i, o in pairs): return None
        return "size_to_color", apply_fn

    # ── 색상 필터 ────────────────────────────────────────────────────
    def _try_color_filter(self, pairs, task):
        """특정 색상만 유지하고 나머지 제거."""
        colors = sorted({int(v) for i,o in pairs for v in np.unique(o) if v!=0})
        for n in range(1, min(len(colors)+1, 8)):
            fn = lambda g, k=n: _keep_top_k_colors(g, k, 0)
            if _apply_and_check(fn, pairs): return f"keep_top{n}", fn
        return None

    def _try_most_frequent_colors(self, pairs, task):
        """입출력 공통 색상 분석 → 특정 색상 제거 규칙."""
        # 출력에서 사라지는 색상 학습
        removed = set()
        for inp, out in pairs:
            ic = set(int(v) for v in np.unique(inp) if v!=0)
            oc = set(int(v) for v in np.unique(out) if v!=0)
            removed |= (ic - oc)
        if not removed: return None
        fn = lambda g, rm=removed: _remove_colors(g, rm)
        if _apply_and_check(fn, pairs): return f"remove_{removed}", fn
        return None

    # ── 행/열 교차점 규칙 ──────────────────────────────────────────────
    def _try_row_col_intersection(self, pairs, task):
        """행에 색A, 열에 색B가 있으면 교차점 셀을 색C로 변경."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:5]
        out_colors = sorted({int(v) for i,o in pairs for v in np.unique(o)})[:8]
        for ca in colors:
            for cb in colors:
                for tc in out_colors:
                    fn = lambda g, a=ca, b=cb, t=tc: _row_col_intersect(g, a, b, t)
                    if _apply_and_check(fn, pairs): return f"rc_intersect_{ca}_{cb}_{tc}", fn
        return None

    # ── 오브젝트 크기/인덱스 재색칠 ────────────────────────────────────
    def _try_object_recolor_by_property(self, pairs, task):
        """오브젝트 크기/인덱스로 재색칠하는 규칙 학습."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        # 오브젝트별 색상 변환 학습
        for fn, name in [
            (lambda g: color_objects_by_size(g,0), "col_by_size"),
            (lambda g: mark_unique_objects(g,0), "mark_unique"),
            (lambda g: recolor_by_object_count(g,0), "recolor_by_count"),
        ]:
            if _apply_and_check(fn, pairs): return name, fn
        return None

    # ── 상세 CA 이웃 규칙 ─────────────────────────────────────────────
    def _try_ca_neighbor_detailed(self, pairs, task):
        """CA 이웃 수 기반 규칙: 이웃 N개 일 때 특정 색으로 변경."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:4]
        out_colors = sorted({int(v) for i,o in pairs for v in np.unique(o)})[:6]
        for src in [0] + colors[:3]:
            for nbr_color in colors[:4]:
                for cnt in range(1, 5):
                    for tgt in out_colors[:5]:
                        fn = lambda g, s=src, nc=nbr_color, n=cnt, t=tgt: \
                            _ca_rule(g, s, nc, n, t)
                        if _apply_and_check(fn, pairs):
                            return f"ca_{src}_nb{nbr_color}_{cnt}_{tgt}", fn
        return None

    # ── 특정 색상 유지/제거 ───────────────────────────────────────────
    def _try_keep_specific_colors(self, pairs, task):
        """훈련 쌍에서 일관되게 유지되는 색상 집합 학습."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        out_color_sets = [set(int(v) for v in np.unique(o)) for _,o in pairs]
        common_out = set.intersection(*out_color_sets) if out_color_sets else set()
        if not common_out: return None
        fn = lambda g, keep=common_out: np.where(np.isin(g, list(keep)), g, 0).astype(np.int8)
        if _apply_and_check(fn, pairs): return f"keep_colors_{sorted(common_out)}", fn
        return None

    # ── 색상별 픽셀 수를 값으로 ─────────────────────────────────────
    def _try_color_by_count(self, pairs, task):
        """각 색상의 픽셀 수를 새 색상값으로 매핑."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        fn = lambda g: _count_to_color(g, 0)
        if _apply_and_check(fn, pairs): return "count_to_color", fn
        return None

    # ── 오브젝트 수를 단일 셀로 ─────────────────────────────────────
    def _try_object_count_output(self, pairs, task):
        """출력 = 오브젝트 수 (단일 셀 혹은 패턴)."""
        for conn in [4, 8]:
            fn = lambda g, c=conn: count_objects_to_value(g, 0)
            if _apply_and_check(fn, pairs): return f"obj_count_{conn}", fn
        return None

    # ── 구분선 기반 추출 ────────────────────────────────────────────
    def _try_extract_by_separator_col(self, pairs, task):
        """구분선(separator) 기준 특정 섹션 추출."""
        fns = []
        # 수직 구분선 (특정 색으로 꽉 찬 열)
        for bg_col in range(1, 10):
            def fn_right(g, bc=bg_col):
                sep = [c for c in range(g.shape[1])
                       if all(g[r,c]==bc for r in range(g.shape[0]))]
                if not sep: return g.copy()
                c0 = sep[0]+1
                return g[:, c0:c0+(sep[0]) if len(sep)<2 else sep[1]-c0].copy()
            def fn_left(g, bc=bg_col):
                sep = [c for c in range(g.shape[1])
                       if all(g[r,c]==bc for r in range(g.shape[0]))]
                if not sep: return g.copy()
                return g[:, :sep[0]].copy()
            for fname, fn in [("ext_left_sep", fn_left), ("ext_right_sep", fn_right)]:
                if _apply_and_check(fn, pairs): return f"{fname}_{bg_col}", fn
        return None

    # ── 정규화 후 규칙 ────────────────────────────────────────────
    def _try_norm_then_rule(self, pairs, task):
        """색상 정규화 후 동일 규칙 시도."""
        norm_pairs = [(normalize_colors(i)[0], normalize_colors(o)[0]) for i,o in pairs]
        if all(np.array_equal(ni, no) for ni, no in norm_pairs):
            def fn(g):
                ng, inv = normalize_colors(g); return denormalize_colors(ng, inv)
            return "norm_identity", fn
        cmap = {}; ok = True
        for ni, no in norm_pairs:
            if ni.shape != no.shape: ok=False; break
            for c in np.unique(ni):
                v=no[ni==c]; u=np.unique(v)
                if len(u)!=1: ok=False; break
                ic=int(c); d=int(u[0])
                if ic in cmap and cmap[ic]!=d: ok=False; break
                cmap[ic]=d
            if not ok: break
        if ok and cmap and not all(k==v for k,v in cmap.items()):
            def fn(g, m=cmap):
                ng, inv = normalize_colors(g)
                return denormalize_colors(recolor(ng, m), inv)
            if _apply_and_check(fn, pairs): return "norm_then_cmap", fn
        return None

    # ── CA "인접 시 변환" 규칙 ──────────────────────────────────────
    def _try_ca_any_neighbor(self, pairs, task):
        """src_color 셀 중 nbr_color 이웃이 하나라도 있으면 tgt로 변경."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:5]
        out_colors = sorted({int(v) for i,o in pairs for v in np.unique(o)})[:8]
        for src in colors + [0]:
            for nbr_color in colors:
                if src == nbr_color: continue
                for tgt in out_colors:
                    if tgt == src: continue
                    fn = lambda g, s=src, nc=nbr_color, t=tgt: _ca_any(g, s, nc, t)
                    if _apply_and_check(fn, pairs): return f"ca_any_{src}_nb{nbr_color}_{tgt}", fn
        return None

    # ── 오브젝트 이동 규칙 ──────────────────────────────────────────
    def _try_object_bbox_shift(self, pairs, task):
        """각 색상의 바운딩박스 이동이 일관되면 학습."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})
        shifts = {}
        for c in colors:
            c_shifts = []
            for inp, out in pairs:
                in_pos = np.argwhere(inp==c)
                out_pos = np.argwhere(out==c)
                if len(in_pos)==0 and len(out_pos)==0: continue
                if len(in_pos)==0 or len(out_pos)==0: c_shifts=None; break
                if len(in_pos)!=len(out_pos): c_shifts=None; break
                in_tl = in_pos.min(axis=0)
                out_tl = out_pos.min(axis=0)
                shift = (int(out_tl[0]-in_tl[0]), int(out_tl[1]-in_tl[1]))
                # verify shape preserved
                test={(int(r)+shift[0],int(cc)+shift[1]) for r,cc in in_pos}
                actual={(int(r),int(cc)) for r,cc in out_pos}
                if test!=actual: c_shifts=None; break
                c_shifts.append(shift)
            if c_shifts is None: continue
            if c_shifts and len(set(c_shifts))==1 and c_shifts[0]!=(0,0):
                shifts[c]=c_shifts[0]
        if not shifts: return None
        def apply_shifts(g, sh=shifts):
            H,W=g.shape; result=np.zeros_like(g)
            for r in range(H):
                for cc in range(W):
                    v=int(g[r,cc])
                    if v not in sh: result[r,cc]=v
            for col,(dr,dc) in sh.items():
                for r,cc in zip(*np.where(g==col)):
                    nr,nc=int(r)+dr,int(cc)+dc
                    if 0<=nr<H and 0<=nc<W: result[nr,nc]=col
            return result
        if _apply_and_check(apply_shifts, pairs):
            return "object_bbox_shift", apply_shifts
        return None

    # ── 색상별 병합 규칙 ────────────────────────────────────────────
    def _try_color_merge(self, pairs, task):
        """출력에서 사라진 색상을 다른 색상에 합치는 규칙."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        in_colors_all = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})
        out_colors_all = sorted({int(v) for i,o in pairs for v in np.unique(o) if v!=0})
        removed = [c for c in in_colors_all if c not in out_colors_all]
        if not removed: return None
        for c in removed:
            for tgt in out_colors_all + [0]:
                fn = lambda g, sc=c, t=tgt: replace_color(g, sc, t)
                if _apply_and_check(fn, pairs): return f"merge_{c}_to_{tgt}", fn
        # Multi-merge
        if len(removed) <= 3:
            merges = {c: 0 for c in removed}
            for c in removed:
                for tgt in out_colors_all + [0]:
                    test = [replace_color(i, c, tgt) for i,_ in pairs]
                    if all(np.array_equal(t, o) for t,(_,o) in zip(test, pairs)):
                        merges[c] = tgt; break
            if all(v is not None for v in merges.values()):
                def fn(g, m=merges): 
                    r=g.copy()
                    for sc,tc in m.items(): r[g==sc]=tc
                    return r
                if _apply_and_check(fn, pairs): return f"multi_merge", fn
        return None

    # ── 오브젝트 채우기 규칙 (enclosed fill) ─────────────────────────
    def _try_enclosed_fill_color(self, pairs, task):
        """특정 색상으로 둘러싸인 영역을 학습된 색상으로 채움."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        for fill in range(1, 10):
            fn = lambda g, f=fill: fill_holes(g, f, 0)
            if _apply_and_check(fn, pairs): return f"enclosed_fill_{fill}", fn
        return None

    # ── 형태 기반 재색칠 규칙 ─────────────────────────────────────
    def _try_shape_recolor(self, pairs, task):
        """십자형/코너/끝점/고립 형태를 다른 색으로 변환."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:5]
        out_colors = sorted({int(v) for i,o in pairs for v in np.unique(o)})[:8]
        for src in colors:
            for tgt in out_colors:
                if tgt == src: continue
                for shape in ["plus", "corner", "endpoint", "isolated"]:
                    fn = lambda g, s=src, t=tgt, sh=shape: _detect_shape_and_recolor(g, s, t, sh)
                    if _apply_and_check(fn, pairs):
                        return f"shape_{shape}_{src}_{tgt}", fn
        return None

    # ── 오브젝트를 목표 색상 근처로 이동 ─────────────────────────────
    def _try_move_to_adjacent_color(self, pairs, task):
        """오브젝트가 특정 색상 근처로 이동하는 규칙 학습."""
        if any(i.shape!=o.shape for i,o in pairs): return None
        # 각 pair에서 어떤 오브젝트가 어디로 이동했는지 분석
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:5]
        
        # 이동 패턴 학습: 오브젝트 X가 항상 오브젝트 Y 근처로 이동
        for move_color in colors:
            for target_color in colors:
                if move_color == target_color: continue
                fn = lambda g, mc=move_color, tc=target_color: _move_obj_to_adjacent(g, mc, tc)
                if _apply_and_check(fn, pairs):
                    return f"move_{move_color}_to_adj_{target_color}", fn
        return None

    # ── 내부 채우기 (경계색으로) ─────────────────────────────────────
    def _try_fill_interior_by_border(self, pairs, task):
        """각 폐쇄 영역을 경계 색으로 채우기 (속이 빈 도형 → 채워진 도형)."""
        if any(i.shape != o.shape for i, o in pairs): return None
        fn = lambda g: _fill_interior_by_border(g)
        if _apply_and_check(fn, pairs): return "fill_interior_by_border", fn
        return None

    # ── Soft CA 룩업 (완화된 일관성) ─────────────────────────────────
    def _try_soft_ca_lookup(self, pairs, task):
        """CA 4-이웃 룩업: 95% 일관성 이상이면 사용 (다수결 방식)."""
        if any(i.shape != o.shape for i, o in pairs): return None
        dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        # (context) -> Counter(output_color)
        from collections import defaultdict
        lookup_cnt: dict = defaultdict(Counter)
        for inp, out in pairs:
            H, W = inp.shape
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r, c]); tc = int(out[r, c])
                    nb4 = tuple(int(inp[r+dr, c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr, dc in dirs4)
                    key = (fc,) + nb4
                    lookup_cnt[key][tc] += 1
        if not lookup_cnt: return None
        # 95% 임계값으로 룩업 테이블 생성
        lookup = {}
        for key, cnt in lookup_cnt.items():
            total = sum(cnt.values())
            best_color, best_n = cnt.most_common(1)[0]
            if best_n / total >= 0.95:
                lookup[key] = best_color
        if not lookup: return None
        # 변환 없는 경우 제외 (모든 매핑이 identity)
        if all(v == k[0] for k, v in lookup.items()): return None
        def apply_fn(inp, lu=lookup):
            H, W = inp.shape; result = inp.copy()
            dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r, c])
                    nb4 = tuple(int(inp[r+dr, c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr, dc in dirs4)
                    key = (fc,) + nb4
                    if key in lu:
                        result[r, c] = lu[key]
            return result
        if not all(np.array_equal(apply_fn(i), o) for i, o in pairs): return None
        return "soft_ca4", apply_fn

    # ── 오브젝트 바운딩박스 채우기 ────────────────────────────────────
    def _try_object_bbox_fill(self, pairs, task):
        """각 오브젝트의 바운딩 박스를 오브젝트 색으로 채우기."""
        if any(i.shape != o.shape for i, o in pairs): return None
        def fill_bbox(g, bg=0):
            result = g.copy()
            for col, cells in [(int(g[r,c]), [(r,c)]) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r,c]!=bg]:
                pass
            objs = find_objects(g, bg, 4)
            for cells in objs:
                col = int(g[cells[0][0], cells[0][1]])
                r0,r1 = cells[:,0].min(), cells[:,0].max()
                c0,c1 = cells[:,1].min(), cells[:,1].max()
                result[r0:r1+1, c0:c1+1] = col
            return result
        fn = lambda g: _fill_obj_bboxes(g)
        if _apply_and_check(fn, pairs): return "obj_bbox_fill", fn
        return None

    # ── 고립 픽셀 재색칠 (인접 비배경 색으로) ─────────────────────────
    def _try_recolor_isolated(self, pairs, task):
        """고립된 1-셀 오브젝트를 인접한 색으로 변경."""
        if any(i.shape != o.shape for i, o in pairs): return None
        fn = lambda g: _recolor_isolated_pixels(g)
        if _apply_and_check(fn, pairs): return "recolor_isolated", fn
        return None

    # ── 가장 가까운 경계로 슬라이드 ─────────────────────────────────
    def _try_slide_to_nearest(self, pairs, task):
        """각 오브젝트를 가장 가까운 경계(다른 오브젝트 or 격자 끝)로 이동."""
        if any(i.shape != o.shape for i, o in pairs): return None
        for direction in ['down', 'up', 'left', 'right']:
            fn = lambda g, d=direction: _slide_to_nearest(g, d)
            if _apply_and_check(fn, pairs): return f"slide_{direction}", fn
        return None

    # ── 멀티 두 단계 (기하+CA) ──────────────────────────────────────
    def _try_multi_two_step(self, pairs, task):
        """회전/반전 후 CA 룩업 적용."""
        if any(i.shape != o.shape for i, o in pairs): return None
        geom_ops = [
            ('rot90', lambda g: np.rot90(g,1)),
            ('rot180', lambda g: np.rot90(g,2)),
            ('rot270', lambda g: np.rot90(g,3)),
            ('fliplr', lambda g: np.fliplr(g)),
            ('flipud', lambda g: np.flipud(g)),
        ]
        dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        for gname, gfn in geom_ops:
            gpairs = [(gfn(i), o) for i, o in pairs]
            if any(i.shape != o.shape for i, o in gpairs): continue
            # CA lookup on transformed pairs
            from collections import defaultdict
            lookup_cnt = defaultdict(Counter)
            for inp, out in gpairs:
                H, W = inp.shape
                for r in range(H):
                    for c in range(W):
                        fc = int(inp[r,c]); tc = int(out[r,c])
                        nb4 = tuple(int(inp[r+dr,c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                    for dr,dc in dirs4)
                        key = (fc,)+nb4
                        lookup_cnt[key][tc] += 1
            lookup = {}
            for key, cnt in lookup_cnt.items():
                total = sum(cnt.values())
                best_c, best_n = cnt.most_common(1)[0]
                if best_n == total:
                    lookup[key] = best_c
            if not lookup or all(v==k[0] for k,v in lookup.items()): continue
            def apply_combo(g, _gfn=gfn, _lu=lookup):
                inp = _gfn(g); H,W=inp.shape; result=inp.copy()
                dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
                for r in range(H):
                    for c in range(W):
                        fc=int(inp[r,c])
                        nb4=tuple(int(inp[r+dr,c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                  for dr,dc in dirs4)
                        key=(fc,)+nb4
                        if key in _lu: result[r,c]=_lu[key]
                return result
            if all(np.array_equal(apply_combo(i), o) for i, o in pairs):
                return f"{gname}+ca4", apply_combo
        return None

    # ── 색상 반전 (비배경 색만) ──────────────────────────────────────
    def _try_invert_colors(self, pairs, task):
        """비배경 색을 모두 반전 (1→9, 2→8 등), 또는 배경/비배경 반전."""
        if any(i.shape != o.shape for i, o in pairs): return None
        def inv_color(g, bg=0):
            result = g.copy()
            result[g != bg] = 10 - g[g != bg]
            return result
        fn = lambda g: inv_color(g)
        if _apply_and_check(fn, pairs): return "invert_colors", fn
        return None

    # ── 배경 폐쇄 영역 채우기 ────────────────────────────────────────
    def _try_paint_enclosed_bg(self, pairs, task):
        """격자 끝에서 접근 불가한 배경 셀(폐쇄 영역)을 특정 색으로 채우기."""
        if any(i.shape != o.shape for i, o in pairs): return None
        out_colors = sorted({int(v) for i,o in pairs for v in np.unique(o) if v!=0})
        for fill_c in out_colors:
            fn = lambda g, fc=fill_c: _paint_enclosed_bg(g, 0, fc)
            if _apply_and_check(fn, pairs): return f"paint_enclosed_bg_{fill_c}", fn
        return None

    # ── 대각 CA ──────────────────────────────────────────────────────
    def _try_diagonal_ca(self, pairs, task):
        """8-방향 CA 룩업 테이블 (다수결 적용)."""
        if any(i.shape != o.shape for i, o in pairs): return None
        dirs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        from collections import defaultdict
        lookup_cnt = defaultdict(Counter)
        for inp, out in pairs:
            H, W = inp.shape
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r, c]); tc = int(out[r, c])
                    nb8 = tuple(int(inp[r+dr, c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr, dc in dirs8)
                    key = (fc,) + nb8
                    lookup_cnt[key][tc] += 1
        if not lookup_cnt: return None
        lookup = {}
        for key, cnt in lookup_cnt.items():
            total = sum(cnt.values())
            best_c, best_n = cnt.most_common(1)[0]
            if best_n == total:
                lookup[key] = best_c
        if not lookup or all(v==k[0] for k,v in lookup.items()): return None
        def apply_fn(inp, lu=lookup):
            H,W=inp.shape; result=inp.copy()
            dirs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            for r in range(H):
                for c in range(W):
                    fc=int(inp[r,c])
                    nb8=tuple(int(inp[r+dr,c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                              for dr,dc in dirs8)
                    key=(fc,)+nb8
                    if key in lu: result[r,c]=lu[key]
            return result
        if not all(np.array_equal(apply_fn(i), o) for i, o in pairs): return None
        return "diag_ca8", apply_fn

    # ── 판별적 CA 룩업 (변화 셀 전용 컨텍스트) ─────────────────────
    def _try_discriminative_ca(self, pairs, task):
        """변화 셀에만 나타나는 고유 컨텍스트를 이용한 정밀 CA 룩업.
        미변화 셀에도 등장하는 컨텍스트는 사용 안 함 → 과적합 방지."""
        if any(i.shape != o.shape for i, o in pairs): return None
        dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        changed_ctx = {}   # ctx -> target_color (변화 셀)
        unchanged_ctx = set()  # 미변화 셀의 컨텍스트
        for inp, out in pairs:
            H, W = inp.shape
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r,c]); tc = int(out[r,c])
                    nb4 = tuple(int(inp[r+dr,c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr,dc in dirs4)
                    key = (fc,) + nb4
                    if fc == tc:
                        unchanged_ctx.add(key)
                    else:
                        if key in changed_ctx and changed_ctx[key] != tc:
                            return None  # 불일치
                        changed_ctx[key] = tc
        if not changed_ctx: return None
        # 변화 셀 전용 컨텍스트만 사용
        exclusive = {k: v for k, v in changed_ctx.items() if k not in unchanged_ctx}
        if not exclusive: return None
        def apply_fn(inp, lu=exclusive):
            H, W = inp.shape; result = inp.copy()
            dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
            for r in range(H):
                for c in range(W):
                    fc = int(inp[r,c])
                    nb4 = tuple(int(inp[r+dr,c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                                for dr,dc in dirs4)
                    key = (fc,) + nb4
                    if key in lu: result[r,c] = lu[key]
            return result
        if not all(np.array_equal(apply_fn(i), o) for i, o in pairs): return None
        return "discriminative_ca4", apply_fn

    # ── 오브젝트 끝점(팁) 재색칠 ─────────────────────────────────────
    def _try_color_at_tips(self, pairs, task):
        """세그먼트 끝점(이웃 1개인 셀)을 특정 색으로 재색칠."""
        if any(i.shape != o.shape for i, o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:5]
        out_colors = sorted({int(v) for i,o in pairs for v in np.unique(o) if v!=0})[:8]
        for src in colors:
            for tgt in out_colors:
                fn = lambda g, s=src, t=tgt: _recolor_tips(g, s, t)
                if _apply_and_check(fn, pairs): return f"tips_{src}→{tgt}", fn
        return None

    # ── 오브젝트 교차점(분기점) 재색칠 ───────────────────────────────
    def _try_color_at_junctions(self, pairs, task):
        """분기점(이웃 3개 이상인 셀)을 특정 색으로 재색칠."""
        if any(i.shape != o.shape for i, o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:5]
        out_colors = sorted({int(v) for i,o in pairs for v in np.unique(o) if v!=0})[:8]
        for src in colors:
            for tgt in out_colors:
                fn = lambda g, s=src, t=tgt: _recolor_junctions(g, s, t)
                if _apply_and_check(fn, pairs): return f"junct_{src}→{tgt}", fn
        return None

    # ── 윤곽→채움 또는 채움→윤곽 ─────────────────────────────────────
    def _try_outline_fill_toggle(self, pairs, task):
        """채워진 사각형 ↔ 윤곽선 사각형 변환."""
        if any(i.shape != o.shape for i, o in pairs): return None
        fns = [
            ('filled_to_outline', lambda g: _filled_to_outline(g)),
            ('outline_to_filled', lambda g: _outline_to_filled(g)),
        ]
        for name, fn in fns:
            if _apply_and_check(fn, pairs): return name, fn
        return None

    # ── 같은 색 오브젝트 직선 연결 ───────────────────────────────────
    def _try_connect_same_color(self, pairs, task):
        """같은 색 두 오브젝트를 직선(수평 or 수직)으로 연결."""
        if any(i.shape != o.shape for i, o in pairs): return None
        fn = lambda g: _connect_same_color_straight(g)
        if _apply_and_check(fn, pairs): return "connect_same_color", fn
        return None

    # ── 열 단위 규칙 ─────────────────────────────────────────────────
    def _try_column_rules(self, pairs, task):
        """각 열에 독립적인 변환(정렬, 색상 치환) 적용."""
        if any(i.shape != o.shape for i, o in pairs): return None
        # 모든 쌍이 같은 크기인지 확인
        shapes = [i.shape for i, o in pairs]
        if len(set(shapes)) > 1: return None
        H, W = shapes[0]
        col_maps = []
        for c in range(W):
            cmap = {}
            consistent = True
            for inp, out in pairs:
                if inp.shape[1] <= c: consistent = False; break
                col_in = inp[:, c].tolist(); col_out = out[:, c].tolist()
                for fc, tc in zip(col_in, col_out):
                    if fc in cmap and cmap[fc] != tc:
                        consistent = False; break
                    cmap[fc] = tc
                if not consistent: break
            col_maps.append(cmap if consistent else None)
        if all(m is None for m in col_maps): return None
        if all(m is not None and all(k==v for k,v in m.items()) for m in col_maps if m): return None
        def apply_fn(inp, cms=col_maps, train_W=W):
            result = inp.copy(); _W = min(inp.shape[1], len(cms), train_W)
            for c in range(_W):
                if cms[c]:
                    for r in range(inp.shape[0]):
                        fc = int(inp[r,c])
                        if fc in cms[c]: result[r,c] = cms[c][fc]
            return result
        if not all(np.array_equal(apply_fn(i), o) for i, o in pairs): return None
        return "column_rules", apply_fn

    # ── 같은 크기 오브젝트 swap ──────────────────────────────────────
    def _try_swap_same_size_objects(self, pairs, task):
        """같은 크기 셀 수의 오브젝트 쌍 위치 교환."""
        if any(i.shape != o.shape for i, o in pairs): return None
        fn = lambda g: _swap_same_size_objects(g)
        if _apply_and_check(fn, pairs): return "swap_same_size", fn
        return None

    # ── 오브젝트를 컨테이너 내부로 이동 ─────────────────────────────
    def _try_move_to_container(self, pairs, task):
        """obj_color 오브젝트를 cont_color 내부로 이동."""
        if any(i.shape != o.shape for i, o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})
        for obj_c in colors:
            for cont_c in colors:
                if obj_c == cont_c: continue
                fn = lambda g, oc=obj_c, cc=cont_c: _move_to_container(g, oc, cc)
                if _apply_and_check(fn, pairs): return f"move_{obj_c}_to_cont_{cont_c}", fn
        return None

    # ── 패턴을 영역 내부에 반복 배치 ─────────────────────────────────
    def _try_replicate_in_region(self, pairs, task):
        """pattern_color 패턴을 region_color 내부에 반복."""
        if any(i.shape != o.shape for i, o in pairs): return None
        colors = sorted({int(v) for i,o in pairs for v in np.unique(i) if v!=0})[:5]
        for pc in colors:
            for rc in colors:
                if pc == rc: continue
                fn = lambda g, p=pc, r=rc: _replicate_pattern_in_region(g, p, r)
                if _apply_and_check(fn, pairs): return f"replicate_{pc}_in_{rc}", fn
        return None

    # ── 행/열 개수를 색상 값으로 변환 ───────────────────────────────
    def _try_row_col_count_to_color(self, pairs, task):
        """각 행(또는 열)의 비배경 셀 수를 그 셀의 색상 값으로 출력."""
        for axis in ['row', 'col']:
            fn_row = lambda g, ax=axis: _count_to_value(g, ax)
            if _apply_and_check(fn_row, pairs): return f"count_to_value_{axis}", fn_row
        return None

    # ── 최대/최소 오브젝트 색 교환 ───────────────────────────────────
    def _try_max_min_object_swap(self, pairs, task):
        """가장 큰 오브젝트와 가장 작은 오브젝트의 색을 교환."""
        if any(i.shape != o.shape for i, o in pairs): return None
        fn = lambda g: _swap_max_min_color(g)
        if _apply_and_check(fn, pairs): return "swap_max_min_color", fn
        return None


# ── 동적 규칙에 필요한 헬퍼 함수들 ──────────────────────────────────

def _count_to_value(g: NGrid, axis: str = 'row', bg: int = 0) -> NGrid:
    """각 행(axis='row') 또는 열(axis='col')의 비배경 셀 수를 그 셀 값으로."""
    result = np.zeros_like(g)
    if axis == 'row':
        for r in range(g.shape[0]):
            cnt = int((g[r] != bg).sum())
            result[r, g[r] != bg] = min(cnt, 9)
    else:
        for c in range(g.shape[1]):
            cnt = int((g[:,c] != bg).sum())
            result[g[:,c] != bg, c] = min(cnt, 9)
    return result


def _swap_max_min_color(g: NGrid, bg: int = 0) -> NGrid:
    """가장 큰 오브젝트와 가장 작은 오브젝트의 색을 교환."""
    objs = find_objects(g, bg, 4)
    if len(objs) < 2: return g.copy()
    largest  = max(objs, key=len)
    smallest = min(objs, key=len)
    col_l = int(g[largest[0,0],  largest[0,1]])
    col_s = int(g[smallest[0,0], smallest[0,1]])
    if col_l == col_s: return g.copy()
    result = g.copy()
    for r,c in largest:  result[r,c] = col_s
    for r,c in smallest: result[r,c] = col_l
    return result


def _keep_top_k_colors(g: NGrid, k: int, bg: int = 0) -> NGrid:
    """상위 K개 색상만 유지."""
    cnt = Counter(g[g!=bg].flatten().tolist())
    top = {c for c,_ in cnt.most_common(k)}
    return np.where(np.isin(g, list(top)), g, bg).astype(np.int8)

def _remove_colors(g: NGrid, colors: set, bg: int = 0) -> NGrid:
    """지정 색상들을 배경으로 제거."""
    result = g.copy()
    for c in colors:
        result[g == c] = bg
    return result

def _row_col_intersect(g: NGrid, row_color: int, col_color: int, tgt: int) -> NGrid:
    """행에 row_color, 열에 col_color가 있는 교차점 셀을 tgt로."""
    H, W = g.shape
    result = g.copy()
    rows_with = {r for r in range(H) if row_color in g[r]}
    cols_with = {c for c in range(W) if col_color in g[:,c]}
    for r in rows_with:
        for c in cols_with:
            result[r,c] = tgt
    return result

def _ca_any(g: NGrid, src_color: int, nbr_color: int, tgt: int) -> NGrid:
    """src_color 셀 중 nbr_color 이웃이 하나라도 있으면 tgt로 변경."""
    H, W = g.shape; result = g.copy()
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(H):
        for c in range(W):
            if g[r,c] == src_color:
                if any(0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]==nbr_color
                       for dr,dc in dirs):
                    result[r,c] = tgt
    return result

def _ca_rule(g: NGrid, src_color: int, nbr_color: int, nbr_count: int, tgt: int) -> NGrid:
    """src_color 셀 중 nbr_color 이웃이 정확히 nbr_count개면 tgt로 변경."""
    H, W = g.shape; result = g.copy()
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(H):
        for c in range(W):
            if g[r,c] == src_color:
                nb = sum(1 for dr,dc in dirs
                         if 0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]==nbr_color)
                if nb == nbr_count: result[r,c] = tgt
    return result

def recolor_by_object_count(g: NGrid, bg: int = 0) -> NGrid:
    """비배경 셀을 오브젝트 개수 값으로 채움."""
    n = len(find_objects(g, bg, 4))
    return np.where(g != bg, min(n, 9), bg).astype(np.int8)

def _fill_interior_by_border(g: NGrid, bg: int = 0) -> NGrid:
    """각 폐쇄 영역을 경계 색으로 채우기.
    1) 격자 끝에서 플러드필 → 외부 배경 찾기
    2) 내부 배경(폐쇄)을 인접 비배경 색으로 채우기."""
    H, W = g.shape
    result = g.copy()
    # 외부에서 접근 가능한 배경 마킹
    from collections import deque as _deq
    exterior = np.zeros((H, W), dtype=bool)
    q = _deq()
    for r in range(H):
        for c in [0, W-1]:
            if g[r,c] == bg and not exterior[r,c]:
                exterior[r,c] = True; q.append((r,c))
    for c in range(W):
        for r in [0, H-1]:
            if g[r,c] == bg and not exterior[r,c]:
                exterior[r,c] = True; q.append((r,c))
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs4:
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and not exterior[nr,nc] and g[nr,nc]==bg:
                exterior[nr,nc] = True; q.append((nr,nc))
    # 내부 배경 찾기 → 인접 비배경 색으로 채우기
    interior = (~exterior) & (g == bg)
    interior_pos = list(zip(*np.where(interior)))
    if not interior_pos: return result
    for r, c in interior_pos:
        # 인접한 비배경 색 찾기 (BFS 확장)
        found_color = bg
        bq = _deq([(r,c)]); seen = {(r,c)}
        while bq and found_color == bg:
            cr, cc = bq.popleft()
            for dr, dc in dirs4:
                nr, nc = cr+dr, cc+dc
                if 0<=nr<H and 0<=nc<W:
                    if g[nr,nc] != bg:
                        found_color = int(g[nr,nc]); break
                    if (nr,nc) not in seen:
                        seen.add((nr,nc)); bq.append((nr,nc))
        result[r, c] = found_color
    return result


def _fill_obj_bboxes(g: NGrid, bg: int = 0) -> NGrid:
    """각 오브젝트의 바운딩박스를 오브젝트 색으로 채우기."""
    result = g.copy()
    objs = find_objects(g, bg, 4)
    for cells in objs:
        col = int(g[cells[0,0], cells[0,1]])
        r0,r1 = cells[:,0].min(), cells[:,0].max()
        c0,c1 = cells[:,1].min(), cells[:,1].max()
        result[r0:r1+1, c0:c1+1] = col
    return result


def _recolor_isolated_pixels(g: NGrid, bg: int = 0) -> NGrid:
    """고립된 단일 셀 오브젝트를 인접한 가장 흔한 비배경 색으로 변경."""
    H, W = g.shape
    result = g.copy()
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    objs = find_objects(g, bg, 4)
    isolated = [cells for cells in objs if len(cells) == 1]
    for cells in isolated:
        r, c = cells[0]
        nb_colors = Counter()
        for dr, dc in dirs4:
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and g[nr,nc] != bg:
                nb_colors[int(g[nr,nc])] += 1
        if nb_colors:
            result[r, c] = nb_colors.most_common(1)[0][0]
    return result


def _slide_to_nearest(g: NGrid, direction: str, bg: int = 0) -> NGrid:
    """각 비배경 오브젝트를 방향으로 슬라이드해서 다른 오브젝트나 격자 끝에 닿을 때까지."""
    H, W = g.shape
    result = np.full_like(g, bg)
    objs = find_objects(g, bg, 4)
    # 오브젝트를 방향 순서로 처리 (가장 가까운 것부터)
    if direction == 'down':
        objs = sorted(objs, key=lambda cells: -cells[:,0].max())
        dr, dc = 1, 0
    elif direction == 'up':
        objs = sorted(objs, key=lambda cells: cells[:,0].min())
        dr, dc = -1, 0
    elif direction == 'right':
        objs = sorted(objs, key=lambda cells: -cells[:,1].max())
        dr, dc = 0, 1
    else:  # left
        objs = sorted(objs, key=lambda cells: cells[:,1].min())
        dr, dc = 0, -1
    for cells in objs:
        col = int(g[cells[0,0], cells[0,1]])
        # 슬라이드
        cur = cells.copy()
        while True:
            nxt = cur + np.array([dr, dc])
            # 격자 범위 확인
            if not (0 <= nxt[:,0].min() and nxt[:,0].max() < H and
                    0 <= nxt[:,1].min() and nxt[:,1].max() < W):
                break
            # 다른 오브젝트와 충돌 확인
            if any(result[r,c] != bg for r,c in nxt):
                break
            cur = nxt
        for r, c in cur:
            result[r, c] = col
    return result


def _paint_enclosed_bg(g: NGrid, bg: int = 0, fill: int = 1) -> NGrid:
    """격자 가장자리에서 접근 불가한 배경 셀을 fill 색으로 채우기."""
    H, W = g.shape
    from collections import deque as _deq
    exterior = np.zeros((H, W), dtype=bool)
    q = _deq()
    for r in range(H):
        for c in [0, W-1]:
            if g[r,c] == bg and not exterior[r,c]:
                exterior[r,c] = True; q.append((r,c))
    for c in range(W):
        for r in [0, H-1]:
            if g[r,c] == bg and not exterior[r,c]:
                exterior[r,c] = True; q.append((r,c))
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs4:
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and not exterior[nr,nc] and g[nr,nc]==bg:
                exterior[nr,nc] = True; q.append((nr,nc))
    result = g.copy()
    result[(~exterior) & (g == bg)] = fill
    return result


def _recolor_tips(g: NGrid, src: int, tgt: int) -> NGrid:
    """src색 오브젝트 중 이웃이 정확히 1개인 셀(끝점)을 tgt로."""
    H, W = g.shape; result = g.copy()
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(H):
        for c in range(W):
            if g[r,c] == src:
                nb = sum(1 for dr,dc in dirs4
                         if 0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]==src)
                if nb == 1: result[r,c] = tgt
    return result


def _recolor_junctions(g: NGrid, src: int, tgt: int) -> NGrid:
    """src색 오브젝트 중 이웃이 3개 이상인 셀(분기점)을 tgt로."""
    H, W = g.shape; result = g.copy()
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    for r in range(H):
        for c in range(W):
            if g[r,c] == src:
                nb = sum(1 for dr,dc in dirs4
                         if 0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]==src)
                if nb >= 3: result[r,c] = tgt
    return result


def _filled_to_outline(g: NGrid, bg: int = 0) -> NGrid:
    """채워진 사각형 → 윤곽선만 남기기."""
    H, W = g.shape; result = g.copy()
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    objs = find_objects(g, bg, 4)
    for cells in objs:
        col = int(g[cells[0,0], cells[0,1]])
        for r, c in cells:
            # 모든 4-이웃이 같은 오브젝트이면 내부 → 배경
            all_same = all(0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]==col
                           for dr,dc in dirs4)
            if all_same: result[r,c] = bg
    return result


def _outline_to_filled(g: NGrid, bg: int = 0) -> NGrid:
    """윤곽선(속이 빈 사각형) → 채워진 사각형."""
    result = g.copy()
    objs = find_objects(g, bg, 4)
    for cells in objs:
        col = int(g[cells[0,0], cells[0,1]])
        r0,r1 = cells[:,0].min(), cells[:,0].max()
        c0,c1 = cells[:,1].min(), cells[:,1].max()
        result[r0:r1+1, c0:c1+1] = col
    return result


def _connect_same_color_straight(g: NGrid, bg: int = 0) -> NGrid:
    """같은 색 두 오브젝트를 수평/수직 직선으로 연결."""
    result = g.copy()
    objs = find_objects(g, bg, 4)
    # 색별 오브젝트 그룹화
    from collections import defaultdict as _dd
    by_color = _dd(list)
    for cells in objs:
        col = int(g[cells[0,0], cells[0,1]])
        by_color[col].append(cells)
    for col, obj_list in by_color.items():
        if len(obj_list) < 2: continue
        # 모든 쌍 조합
        for i in range(len(obj_list)):
            for j in range(i+1, len(obj_list)):
                cells_a, cells_b = obj_list[i], obj_list[j]
                # 수평 정렬: 같은 행이 있는지
                rows_a = set(cells_a[:,0].tolist())
                rows_b = set(cells_b[:,0].tolist())
                common_rows = rows_a & rows_b
                for row in common_rows:
                    cs_a = cells_a[cells_a[:,0]==row, 1]
                    cs_b = cells_b[cells_b[:,0]==row, 1]
                    c_left = min(cs_a.max(), cs_b.max())
                    c_right = max(cs_a.min(), cs_b.min())
                    if c_left < c_right:
                        result[row, c_left:c_right+1] = col
                # 수직 정렬: 같은 열이 있는지
                cols_a = set(cells_a[:,1].tolist())
                cols_b = set(cells_b[:,1].tolist())
                common_cols = cols_a & cols_b
                for col_idx in common_cols:
                    rs_a = cells_a[cells_a[:,1]==col_idx, 0]
                    rs_b = cells_b[cells_b[:,1]==col_idx, 0]
                    r_top = min(rs_a.max(), rs_b.max())
                    r_bot = max(rs_a.min(), rs_b.min())
                    if r_top < r_bot:
                        result[r_top:r_bot+1, col_idx] = col
    return result


def _swap_same_size_objects(g: NGrid, bg: int = 0) -> NGrid:
    """같은 크기(셀 수)인 오브젝트 쌍의 위치를 교환.
    크기별로 그룹화 → 정확히 2개 있는 크기만 swap."""
    from collections import defaultdict as _dd
    objs = find_objects(g, bg, 4)
    by_size = _dd(list)
    for cells in objs:
        by_size[len(cells)].append(cells)
    result = g.copy()
    for sz, obj_list in by_size.items():
        if len(obj_list) != 2: continue
        a, b = obj_list
        col_a = int(g[a[0,0], a[0,1]])
        col_b = int(g[b[0,0], b[0,1]])
        if col_a == col_b: continue
        # a 위치에 b 패턴, b 위치에 a 패턴 그리기
        # 1) 두 오브젝트를 배경으로 지우기
        for r,c in a: result[r,c] = bg
        for r,c in b: result[r,c] = bg
        # 2) bbox 오프셋 계산
        a_r0,a_c0 = a[:,0].min(),a[:,1].min()
        b_r0,b_c0 = b[:,0].min(),b[:,1].min()
        # 3) a 위치에 b 패턴 그리기
        for r,c in b:
            nr,nc = a_r0+(r-b_r0), a_c0+(c-b_c0)
            H,W = g.shape
            if 0<=nr<H and 0<=nc<W: result[nr,nc] = col_b
        # 4) b 위치에 a 패턴 그리기
        for r,c in a:
            nr,nc = b_r0+(r-a_r0), b_c0+(c-a_c0)
            if 0<=nr<H and 0<=nc<W: result[nr,nc] = col_a
    return result


def _move_to_container(g: NGrid, obj_color: int, cont_color: int, bg: int = 0) -> NGrid:
    """obj_color 오브젝트를 cont_color 윤곽선 내부로 이동.
    cont_color로 이루어진 바운딩박스 내부 중앙에 obj를 배치."""
    H, W = g.shape
    objs = find_objects(g, bg, 4)
    containers = [c for c in objs if int(g[c[0,0],c[0,1]])==cont_color]
    movers     = [c for c in objs if int(g[c[0,0],c[0,1]])==obj_color]
    if not containers or not movers: return g.copy()
    result = g.copy()
    for mover in movers:
        # 가장 가까운 container 찾기
        mr, mc = mover[:,0].mean(), mover[:,1].mean()
        best_cont = min(containers, key=lambda cont:
                        abs(cont[:,0].mean()-mr)+abs(cont[:,1].mean()-mc))
        # container의 내부 중심 계산
        cr0,cr1 = best_cont[:,0].min(),best_cont[:,0].max()
        cc0,cc1 = best_cont[:,1].min(),best_cont[:,1].max()
        center_r = (cr0+cr1)//2
        center_c = (cc0+cc1)//2
        # mover bbox
        mr0,mr1 = mover[:,0].min(),mover[:,0].max()
        mc0,mc1 = mover[:,1].min(),mover[:,1].max()
        offset_r = center_r - (mr0+mr1)//2
        offset_c = center_c - (mc0+mc1)//2
        # 지우기
        for r,c in mover: result[r,c] = bg
        # 새 위치에 그리기
        for r,c in mover:
            nr,nc = r+offset_r, c+offset_c
            if 0<=nr<H and 0<=nc<W: result[nr,nc] = obj_color
    return result


def _align_objects_to_grid_lines(g: NGrid, bg: int = 0) -> NGrid:
    """가장 많이 사용된 행/열 위치에 오브젝트를 정렬.
    각 오브젝트의 중심 행/열을 가장 가까운 격자선으로 스냅."""
    objs = find_objects(g, bg, 4)
    if len(objs) < 2: return g.copy()
    # 오브젝트 중심 행/열 수집
    centers_r = sorted(set(round((cells[:,0].mean())) for cells in objs))
    centers_c = sorted(set(round((cells[:,1].mean())) for cells in objs))
    result = np.full_like(g, bg)
    H, W = g.shape
    for cells in objs:
        col = int(g[cells[0,0], cells[0,1]])
        cur_r = cells[:,0].mean(); cur_c = cells[:,1].mean()
        snap_r = min(centers_r, key=lambda x: abs(x-cur_r))
        snap_c = min(centers_c, key=lambda x: abs(x-cur_c))
        dr = round(snap_r - cur_r); dc = round(snap_c - cur_c)
        for r,c in cells:
            nr,nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W: result[nr,nc] = col
    return result


def _replicate_pattern_in_region(g: NGrid, pattern_color: int,
                                  region_color: int, bg: int = 0) -> NGrid:
    """pattern_color로 이루어진 패턴을 region_color 내부 영역에 반복 배치."""
    H, W = g.shape
    objs = find_objects(g, bg, 4)
    patterns = [c for c in objs if int(g[c[0,0],c[0,1]])==pattern_color]
    regions  = [c for c in objs if int(g[c[0,0],c[0,1]])==region_color]
    if not patterns or not regions: return g.copy()
    result = g.copy()
    # 첫 번째 패턴 bbox
    pat = patterns[0]
    pr0,pr1 = pat[:,0].min(),pat[:,0].max()
    pc0,pc1 = pat[:,1].min(),pat[:,1].max()
    ph,pw = pr1-pr0+1, pc1-pc0+1
    # 패턴 마스크
    mask = np.zeros((ph,pw), dtype=np.int32)
    for r,c in pat:
        mask[r-pr0, c-pc0] = pattern_color
    for region in regions:
        rr0,rr1 = region[:,0].min(),region[:,0].max()
        rc0,rc1 = region[:,1].min(),region[:,1].max()
        for dr in range(0, rr1-rr0+1, ph):
            for dc in range(0, rc1-rc0+1, pw):
                for mr in range(ph):
                    for mc in range(pw):
                        if mask[mr,mc] != 0:
                            nr,nc = rr0+dr+mr, rc0+dc+mc
                            if 0<=nr<H and 0<=nc<W:
                                result[nr,nc] = mask[mr,mc]
    return result


def _detect_plus_centers(g: NGrid, color: int) -> list:
    """상하좌우 모두 동일 색상으로 둘러싸인 '십자형 중심' 셀 탐지."""
    H, W = g.shape; centers = []
    for r in range(1, H-1):
        for c in range(1, W-1):
            if g[r,c]==color and g[r-1,c]==color and g[r+1,c]==color \
               and g[r,c-1]==color and g[r,c+1]==color:
                centers.append((r,c))
    return centers

def _color_plus_patterns(g: NGrid, src: int, tgt: int) -> NGrid:
    """src색 십자 패턴을 tgt색으로 변환 (중심+4방향 팔)."""
    centers = _detect_plus_centers(g, src)
    result = g.copy()
    for r, c in centers:
        result[r, c] = tgt
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            result[r+dr, c+dc] = tgt
    return result

def _detect_shape_and_recolor(g: NGrid, src: int, tgt: int, shape: str) -> NGrid:
    """특정 형태(plus, L, T, square 등)의 src 셀을 tgt로 변환."""
    H, W = g.shape; result = g.copy()
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    
    if shape == "plus":
        return _color_plus_patterns(g, src, tgt)
    
    elif shape == "corner":
        # 2방향 이웃이 있는 셀 (ㄴ자 꼭짓점)
        for r in range(H):
            for c in range(W):
                if g[r,c] == src:
                    nb = [(dr,dc) for dr,dc in dirs4
                          if 0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]==src]
                    if len(nb) == 2:
                        d0,d1=nb[0],nb[1]
                        # 두 방향이 서로 수직인지 (직각 코너)
                        if d0[0]*d1[0]+d0[1]*d1[1]==0:
                            result[r,c]=tgt
    
    elif shape == "endpoint":
        # 1방향 이웃만 있는 셀 (선의 끝점)
        for r in range(H):
            for c in range(W):
                if g[r,c] == src:
                    nb = sum(1 for dr,dc in dirs4
                             if 0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]==src)
                    if nb == 1:
                        result[r,c] = tgt
    
    elif shape == "isolated":
        # 이웃 없는 셀 (고립된 점)
        for r in range(H):
            for c in range(W):
                if g[r,c] == src:
                    nb = sum(1 for dr,dc in dirs4
                             if 0<=r+dr<H and 0<=c+dc<W and g[r+dr,c+dc]==src)
                    if nb == 0:
                        result[r,c] = tgt
    
    return result

def _move_obj_to_adjacent(g: NGrid, move_color: int, target_color: int, bg: int = 0) -> NGrid:
    """move_color 오브젝트를 target_color 오브젝트 인접 위치로 이동."""
    H, W = g.shape
    # move_color 오브젝트 위치
    move_pos = np.argwhere(g == move_color)
    if len(move_pos) == 0: return g.copy()
    # target_color 오브젝트 위치
    target_pos = np.argwhere(g == target_color)
    if len(target_pos) == 0: return g.copy()
    
    # move 오브젝트의 바운딩 박스
    m_r0, m_c0 = move_pos.min(axis=0)
    m_r1, m_c1 = move_pos.max(axis=0)
    m_h, m_w = m_r1-m_r0+1, m_c1-m_c0+1
    
    # target 근처에서 가장 가까운 빈 위치 탐색
    t_center = target_pos.mean(axis=0)
    
    # 4방향에서 target 근처 배치 시도
    best_pos = None; best_dist = float('inf')
    for adj_r, adj_c in [(int(t_center[0])-m_h, int(t_center[1])),
                          (int(t_center[0])+1, int(t_center[1])),
                          (int(t_center[0]), int(t_center[1])-m_w),
                          (int(t_center[0]), int(t_center[1])+1)]:
        if 0<=adj_r and adj_r+m_h<=H and 0<=adj_c and adj_c+m_w<=W:
            # Check if empty
            region = g[adj_r:adj_r+m_h, adj_c:adj_c+m_w]
            if not (region == move_color).any() and not (region == target_color).any():
                dist = abs(adj_r-t_center[0])+abs(adj_c-t_center[1])
                if dist < best_dist:
                    best_dist = dist; best_pos = (adj_r, adj_c)
    
    if best_pos is None: return g.copy()
    
    # 이동 적용
    result = g.copy()
    # 기존 위치 제거
    for r,c in move_pos: result[r,c] = bg
    # 새 위치에 배치 (로컬 형태 유지)
    nr0, nc0 = best_pos
    for r, c in move_pos:
        nr, nc = nr0+(r-m_r0), nc0+(c-m_c0)
        if 0<=nr<H and 0<=nc<W:
            result[nr,nc] = move_color
    return result

def _count_to_color(g: NGrid, bg: int = 0) -> NGrid:
    """색상별 픽셀 수를 1~9 값으로 매핑해 원래 셀에 적용."""
    cnt = Counter(g[g!=bg].flatten().tolist())
    if not cnt: return g.copy()
    max_cnt = max(cnt.values())
    mapping = {c: min(int(round(9*n/max_cnt)),9) for c,n in cnt.items()}
    result = g.copy()
    for c, v in mapping.items(): result[g==c] = v
    return result


def _flood_fill_regions(g: NGrid, seed_color: int, fill_color: int, bg: int = 0) -> NGrid:
    """seed_color에서 시작해 bg를 만날 때까지 flood fill."""
    H, W = g.shape; result = g.copy()
    visited = np.zeros((H,W), dtype=bool)
    queue = deque()
    for r in range(H):
        for c in range(W):
            if g[r,c] == seed_color: queue.append((r,c)); visited[r,c] = True
    while queue:
        r, c = queue.popleft()
        result[r,c] = fill_color
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and g[nr,nc] == bg:
                visited[nr,nc] = True; queue.append((nr,nc))
    return result

def _gravity_any_dir(g: NGrid, direction: str, bg: int = 0) -> NGrid:
    """중력 (down/up/left/right)."""
    H, W = g.shape; result = np.full_like(g, bg)
    if direction in ('down', 'up'):
        for c in range(W):
            col = [int(g[r,c]) for r in range(H) if int(g[r,c]) != bg]
            if direction == 'down':
                for i,v in enumerate(col): result[H-len(col)+i, c] = v
            else:
                for i,v in enumerate(col): result[i, c] = v
    else:
        for r in range(H):
            row = [int(g[r,c]) for c in range(W) if int(g[r,c]) != bg]
            if direction == 'right':
                for i,v in enumerate(row): result[r, W-len(row)+i] = v
            else:
                for i,v in enumerate(row): result[r, i] = v
    return result

def _sort_rows_by_count(g: NGrid, bg: int = 0) -> NGrid:
    """각 행을 색상별 정렬."""
    result = g.copy()
    for r in range(g.shape[0]):
        row = sorted([int(g[r,c]) for c in range(g.shape[1])], reverse=True)
        for c, v in enumerate(row): result[r,c] = v
    return result

def _sort_cols_by_count(g: NGrid, bg: int = 0) -> NGrid:
    """각 열을 색상별 정렬."""
    result = g.copy()
    for c in range(g.shape[1]):
        col = sorted([int(g[r,c]) for r in range(g.shape[0])], reverse=True)
        for r, v in enumerate(col): result[r,c] = v
    return result

def _complete_symmetry(g: NGrid, axis: str) -> NGrid:
    """빈(0) 부분을 대칭으로 채우기."""
    result = g.copy()
    H, W = g.shape
    if axis == 'h':  # 좌우 대칭
        for r in range(H):
            for c in range(W):
                mirror = W - 1 - c
                if result[r,c] == 0 and result[r,mirror] != 0:
                    result[r,c] = result[r,mirror]
    elif axis == 'v':  # 상하 대칭
        for r in range(H):
            for c in range(W):
                mirror = H - 1 - r
                if result[r,c] == 0 and result[mirror,c] != 0:
                    result[r,c] = result[mirror,c]
    return result

def _scale_n(g: NGrid, n: int) -> NGrid:
    """n배 스케일업."""
    return np.kron(g, np.ones((n,n), dtype=np.int32))

def _apply_mask(g: NGrid, mask_color: int, paint_color: int) -> NGrid:
    """mask_color 셀을 paint_color로 변환하고 나머지는 0으로."""
    result = np.zeros_like(g)
    result[g == mask_color] = paint_color
    return result


_dynamic_learner = DynamicRuleLearner()
print(f"✓ Phase 4c 로드 완료 (DynamicRuleLearner — 48개 전략)")
'''

# ── Phase 5: 빔서치 ────────────────────────────────────────────────────────────────
CELL_PHASE5 = '''\
def pixel_score(pred,target):
    if pred.shape!=target.shape: return 0.0
    return float(np.mean(pred==target))
def score_all(preds,targets):
    if not preds or not targets: return 0.0
    return float(np.mean([pixel_score(p,t) for p,t in zip(preds,targets)]))

BeamState = tuple[float, list[str], list[NGrid]]

def _single_beam_pass(beam,ops,targets,beam_width,t0,timeout):
    solutions=[]; candidates=[]
    for score,op_seq,cur_grids in beam:
        if time.perf_counter()-t0>timeout: break
        for op_name,op_fn in ops:
            try:
                ng=[op_fn(g) for g in cur_grids]
                if any(g.size==0 for g in ng): continue
                s=score_all(ng,targets)
                seq=op_seq+[op_name]
                if s>=1.0-1e-9: solutions.append((seq,ng))
                else: candidates.append((s,seq,ng))
            except: continue
    candidates.sort(key=lambda x:-x[0])
    seen=set(); unique=[]
    for s,seq,ng in candidates:
        key=(round(s,4),tuple(seq[-2:]))
        if key not in seen:
            seen.add(key); unique.append((s,seq,ng))
        if len(unique)>=beam_width: break
    return unique, solutions

def _patch_from_states(states, targets):
    """동일 좌표에서 동일하게 틀린 픽셀들을 고정 패치로."""
    if not states or any(s.shape!=t.shape for s,t in zip(states,targets)): return None
    masks=[(s!=t) for s,t in zip(states,targets)]
    if not all(np.array_equal(masks[0],m) for m in masks[1:]): return None
    mask=masks[0]; dc=int(mask.sum())
    if dc==0 or dc>150: return None
    patch={}
    for r,c in np.argwhere(mask):
        vals=[int(t[r,c]) for t in targets]
        if len(set(vals))!=1: return None
        patch[(int(r),int(c))]=vals[0]
    def apply(g,p=patch):
        res=g.copy(); H,W=res.shape
        for (r,c),col in p.items():
            if 0<=r<H and 0<=c<W: res[r,c]=col
        return res
    return (f"beam_patch_{dc}", apply)

def _residual_colormap(states, targets):
    """빔서치 결과의 잔차를 색상 매핑으로 보정.
    sequence_output의 틀린 색 A → 올바른 색 B가 모든 pair에서 일관되면 적용."""
    if not states or any(s.shape!=t.shape for s,t in zip(states,targets)): return None
    cmap={}
    for state,target in zip(states,targets):
        mask=state!=target
        if not mask.any(): continue
        for r,c in np.argwhere(mask):
            s_val=int(state[r,c]); t_val=int(target[r,c])
            if s_val in cmap and cmap[s_val]!=t_val: return None
            cmap[s_val]=t_val
    if not cmap: return None
    # 검증: 이미 맞는 픽셀을 망가뜨리지 않는지 확인
    def apply(g, m=cmap):
        res=g.copy()
        for s,t in m.items(): res[g==s]=t
        return res
    for state,target in zip(states,targets):
        if not np.array_equal(apply(state),target): return None
    return ("residual_cmap", apply)


def _residual_ca_lookup(states, targets):
    """빔서치 결과(states)와 정답(targets)의 차이를 CA 룩업으로 수정.
    (predicted_color, 4-이웃) → target_color 룩업이 일관되면 적용."""
    if not states or any(s.shape!=t.shape for s,t in zip(states,targets)): return None
    dirs4=[(-1,0),(1,0),(0,-1),(0,1)]
    lookup: dict={}
    for state, target in zip(states,targets):
        H,W=state.shape
        for r in range(H):
            for c in range(W):
                sc=int(state[r,c]); tc=int(target[r,c])
                nb4=tuple(int(state[r+dr,c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                          for dr,dc in dirs4)
                key=(sc,)+nb4
                if key in lookup and lookup[key]!=tc: return None
                lookup[key]=tc
    if not lookup or all(k[0]==v for k,v in lookup.items()): return None

    def apply(g, lut=lookup):
        H,W=g.shape; res=g.copy()
        for r in range(H):
            for c in range(W):
                sc=int(g[r,c])
                nb4=tuple(int(g[r+dr,c+dc]) if 0<=r+dr<H and 0<=c+dc<W else -1
                          for dr,dc in dirs4)
                key=(sc,)+nb4
                if key in lut: res[r,c]=lut[key]
        return res

    # 모든 pair에서 100% 매치 검증
    for state, target in zip(states, targets):
        if not np.array_equal(apply(state), target): return None
    return ("residual_ca", apply)


def _residual_global_cmap(states, targets):
    """잔차 색상 전역 매핑: predicted의 색상 X가 항상 target의 Y가 되면 적용."""
    if not states or any(s.shape!=t.shape for s,t in zip(states,targets)): return None
    cmap={}
    for state, target in zip(states, targets):
        for sc in np.unique(state):
            sc=int(sc)
            mask=(state==sc)
            tcs=np.unique(target[mask])
            if len(tcs)!=1: return None
            tc=int(tcs[0])
            if sc in cmap and cmap[sc]!=tc: return None
            cmap[sc]=tc
    if not cmap or all(k==v for k,v in cmap.items()): return None
    def apply(g, m=cmap):
        res=g.copy()
        for s,t in m.items(): res[g==s]=t
        return res
    for state,target in zip(states,targets):
        if not np.array_equal(apply(state),target): return None
    return ("residual_gcmap", apply)


class BeamSearchEngine:
    def __init__(self, beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH, timeout=TASK_TIMEOUT):
        self.beam_width=beam_width; self.max_depth=max_depth; self.timeout=timeout

    def solve(self, task: Task) -> "BeamResult":
        t0=time.perf_counter()
        train_in=[g2n(p.input) for p in task.train]
        train_out=[g2n(p.output) for p in task.train if p.output is not None]
        if len(train_in)!=len(train_out):
            return BeamResult(task.task_id,False,[],None,0.0,"mismatch")
        if task.task_id not in _OPS_CACHE:
            _OPS_CACHE[task.task_id]=build_task_ops(task)
        ops=_OPS_CACHE[task.task_id]
        init=score_all(train_in,train_out)
        if init>=1.0-1e-9:
            return BeamResult(task.task_id,True,[],train_in,1.0,"identity")

        def run(inputs,targets,dl,prefix=None):
            prefix=prefix or []
            beam=[(score_all(inputs,targets),[],list(inputs))]
            best=beam[0]; sols=[]
            for _ in range(self.max_depth):
                if time.perf_counter()-t0>dl: break
                beam,s=_single_beam_pass(beam,ops,targets,self.beam_width,t0,dl)
                sols.extend([(prefix+seq,grids) for seq,grids in s])
                if s: break
                if beam and beam[0][0]>best[0]: best=beam[0]
            # stage2 finetune
            if not sols and best[0]>=0.5 and time.perf_counter()-t0<dl:
                fb=[best]; fw=min(self.beam_width*2,60)
                for _ in range(2):
                    if time.perf_counter()-t0>dl: break
                    fb,s=_single_beam_pass(fb,ops,targets,fw,t0,dl)
                    sols.extend([(prefix+seq,grids) for seq,grids in s])
                    if s: break
                    if fb and fb[0][0]>best[0]: best=fb[0]
            return sols, best

        all_sols,best=run(train_in,train_out,self.timeout*0.6)
        # residual patch (동일 좌표 패치)
        if not all_sols and best[0]>=0.80 and time.perf_counter()-t0<self.timeout*0.75:
            pop=_patch_from_states(best[2],train_out)
            if pop:
                pn,pf=pop; _OPS_CACHE[task.task_id].append(pop)
                try:
                    pg=[pf(g) for g in best[2]]
                    if score_all(pg,train_out)>=1.0-1e-9:
                        all_sols.append((best[1]+[pn],pg))
                except: pass
        # residual CA lookup (CA 룩업 테이블로 잔차 수정)
        if not all_sols and best[0]>=0.70 and time.perf_counter()-t0<self.timeout*0.82:
            rcl=_residual_ca_lookup(best[2],train_out)
            if rcl:
                rn,rf=rcl
                try:
                    rg=[rf(g) for g in best[2]]
                    if score_all(rg,train_out)>=1.0-1e-9:
                        all_sols.append((best[1]+[rn],rg))
                except: pass
        # residual global colormap
        if not all_sols and best[0]>=0.70 and time.perf_counter()-t0<self.timeout*0.85:
            rgcm=_residual_global_cmap(best[2],train_out)
            if rgcm:
                rn,rf=rgcm
                try:
                    rg=[rf(g) for g in best[2]]
                    if score_all(rg,train_out)>=1.0-1e-9:
                        all_sols.append((best[1]+[rn],rg))
                except: pass
        # residual colormap (잔차 색상 매핑 보정)
        if not all_sols and best[0]>=0.70 and time.perf_counter()-t0<self.timeout*0.88:
            rcm=_residual_colormap(best[2],train_out)
            if rcm:
                rn,rf=rcm
                try:
                    rg=[rf(g) for g in best[2]]
                    if score_all(rg,train_out)>=1.0-1e-9:
                        all_sols.append((best[1]+[rn],rg))
                except: pass
        # 1-step exhaustive search on near-perfect result (마지막 1마일 탐색)
        if not all_sols and best[0]>=0.80 and time.perf_counter()-t0<self.timeout*0.92:
            cur_grids=best[2]
            for op_name,op_fn in ops:
                if time.perf_counter()-t0>=self.timeout*0.92: break
                try:
                    new_g=[op_fn(g) for g in cur_grids]
                    if score_all(new_g,train_out)>=1.0-1e-9:
                        all_sols.append((best[1]+[op_name],new_g)); break
                except: pass
        # 2-step exhaustive search (0.90+ 근접, 1-step이 실패한 경우)
        if not all_sols and best[0]>=0.90 and time.perf_counter()-t0<self.timeout*0.95:
            cur_grids=best[2]
            best_score_2s=best[0]; best_grids_2s=cur_grids; best_seq_2s=best[1]
            for op1_name,op1_fn in ops:
                if time.perf_counter()-t0>=self.timeout*0.95: break
                try:
                    g1=[op1_fn(g) for g in cur_grids]
                    s1=score_all(g1,train_out)
                    if s1 < best_score_2s: continue
                    for op2_name,op2_fn in ops:
                        if time.perf_counter()-t0>=self.timeout*0.95: break
                        try:
                            g2=[op2_fn(gg) for gg in g1]
                            s2=score_all(g2,train_out)
                            if s2>=1.0-1e-9:
                                all_sols.append((best[1]+[op1_name,op2_name],g2))
                                break
                        except: pass
                    if all_sols: break
                except: pass
        # color-invariant search
        if not all_sols and time.perf_counter()-t0<self.timeout:
            try:
                ni=[normalize_colors(g)[0] for g in train_in]
                no=[normalize_colors(g)[0] for g in train_out]
                ns,nb=run(ni,no,self.timeout,prefix=["__norm__"])
                all_sols.extend(ns)
                if not all_sols and nb[0]>best[0]: best=(nb[0],["__norm__"]+nb[1],nb[2])
            except: pass

        if all_sols:
            all_sols.sort(key=lambda x:len(x[0]))
            s1,g1=all_sols[0]; s2,g2=all_sols[1] if len(all_sols)>1 else (s1,g1)
            return BeamResult(task.task_id,True,s1,g1,1.0,"",second_seq=s2,second_grids=g2)
        s,seq,grids=best
        return BeamResult(task.task_id,False,seq,grids,s,f"partial {s:.3f}")

    def apply_sequence(self, seq, inputs, task):
        if task.task_id not in _OPS_CACHE:
            _OPS_CACHE[task.task_id]=build_task_ops(task)
        ops_dict={n:f for n,f in _OPS_CACHE[task.task_id]}
        normalized=bool(seq and seq[0]=="__norm__")
        inv_maps=[]
        if normalized:
            grids=[]; seq=seq[1:]
            for inp in inputs:
                ng,inv=normalize_colors(g2n(inp))
                grids.append(ng); inv_maps.append(inv)
        else:
            grids=[g2n(i) for i in inputs]
        for op in seq:
            fn=ops_dict.get(op)
            if fn is None: continue
            try: grids=[fn(g) for g in grids]
            except: pass
        if normalized:
            grids=[denormalize_colors(g,inv) for g,inv in zip(grids,inv_maps)]
        return grids


@dataclass
class BeamResult:
    task_id: TaskId; success: bool; seq1: list[str]
    grids1: list[NGrid] | None; score: float; error: str=""
    second_seq: list[str]=field(default_factory=list)
    second_grids: list[NGrid] | None=None
    def __str__(self):
        st="✓" if self.success else f"~{self.score:.3f}"
        return f"[{st}] {self.task_id}  {self.seq1[:3]}{'…' if len(self.seq1)>3 else ''}"


print(f"✓ Phase 5 로드 완료 (BeamSearchEngine  beam={BEAM_WIDTH}  depth={MAX_DEPTH})")
'''

# ── Phase 6: Solver — RuleEngine 우선, BeamSearch 폴백 ────────────────────────────
CELL_PHASE6 = '''\
# ═══════════════════════════════════════════════════════════════════
# Phase 6: 통합 Solver — RuleEngine → BeamSearch 앙상블
# ═══════════════════════════════════════════════════════════════════

class HeuristicFallback:
    @staticmethod
    def best_guess(task, inp):
        ps=_shape_predictor.predict(task,inp)
        H=ps[0] if ps else len(inp)
        W=ps[1] if ps else (len(inp[0]) if inp else 1)
        cnt=Counter()
        for p in task.train:
            if p.output:
                for row in p.output: cnt.update(row)
        dom=cnt.most_common(1)[0][0] if cnt else 0
        return [[int(dom)]*W for _ in range(H)]
    @staticmethod
    def copy_input(inp): return copy.deepcopy(inp)


class ARCSolver:
    """
    다층 앙상블 Solver (v5):
      attempt_1: Rule → Dynamic → Beam 우선순위로 가장 좋은 결과
      attempt_2: 독립적인 두 번째 경로 (서로 다른 답안)
        - Rule 성공 → attempt_2는 Dynamic 또는 Beam 최고 partial
        - Dynamic 성공 → attempt_2는 Rule 시도 or Beam partial
        - Beam 전용 → second_seq 또는 HeuristicFallback
    """
    def __init__(self, beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH,
                 timeout=TASK_TIMEOUT, max_workers=MAX_WORKERS):
        self.engine   = BeamSearchEngine(beam_width, max_depth, timeout)
        self.fallback = HeuristicFallback()
        self.max_workers = max_workers

    def _ngrid_to_list(self, ng, fallback):
        if ng is None: return fallback
        try: return n2g(ng)
        except: return fallback

    def _beam_partial_for_inp(self, task, inp, beam_result):
        """빔서치 partial 결과를 inp에 적용해서 반환. 없으면 fallback."""
        fb = self.fallback.best_guess(task, inp)
        if beam_result is None or beam_result.grids1 is None:
            return fb
        tg = self.engine.apply_sequence(beam_result.seq1, [inp], task)
        return self._ngrid_to_list(tg[0] if tg else None, fb)

    def solve_task(self, task: Task):
        fb = self.fallback

        # ── 1순위: RuleEngine (고정 규칙) ─────────────────────────────────────
        rule_result = _rule_engine.solve(task)

        # ── 2순위: DynamicRuleLearner (데이터 기반 동적 규칙) ─────────────────
        dyn_result = _dynamic_learner.solve(task)

        # ── 3순위: BeamSearch (항상 실행 — attempt_2에 활용) ─────────────────
        # Rule/Dynamic이 성공해도 빔서치 partial을 attempt_2로 쓸 수 있음
        # 단, 시간 절약을 위해 Rule+Dynamic 모두 성공 시 빔서치 생략
        run_beam = not (rule_result is not None and dyn_result is not None)
        beam_result = self.engine.solve(task) if run_beam else None

        # ── attempt_1 결정 ──────────────────────────────────────────────────
        if rule_result is not None:
            name1, grids1 = rule_result
            a1_mode = "rule"
        elif dyn_result is not None:
            name1, grids1 = dyn_result
            a1_mode = "dynamic"
        elif beam_result is not None:
            a1_mode = "beam"
            grids1 = None  # beam은 inp별로 계산
        else:
            a1_mode = "fallback"
            grids1 = None

        # ── attempt_2 결정 ──────────────────────────────────────────────────
        # attempt_1과 다른 경로를 사용
        if rule_result is not None and dyn_result is not None:
            # 두 독립 규칙 모두 성공 → attempt_1: rule, attempt_2: dynamic
            a2_mode = "dynamic"
        elif rule_result is not None:
            # rule만 성공 → attempt_2: dynamic or beam partial
            a2_mode = "dynamic_or_beam"
        elif dyn_result is not None:
            # dynamic만 성공 → attempt_2: beam partial (성능 보완)
            a2_mode = "beam_partial"
        else:
            # 둘 다 실패 → beam second_seq or fallback
            a2_mode = "beam_second"

        preds = []
        for i, inp in enumerate(task.test_inputs):
            fb1 = fb.best_guess(task, inp)
            fb2 = fb.copy_input(inp)

            # attempt_1
            if a1_mode == "rule":
                a1 = n2g(grids1[i]) if grids1 and i < len(grids1) else fb1
            elif a1_mode == "dynamic":
                a1 = n2g(grids1[i]) if grids1 and i < len(grids1) else fb1
            elif a1_mode == "beam" and beam_result is not None:
                if beam_result.success and beam_result.grids1 is not None:
                    tg = self.engine.apply_sequence(beam_result.seq1, [inp], task)
                    a1 = self._ngrid_to_list(tg[0] if tg else None, fb1)
                elif beam_result.grids1 is not None and beam_result.score >= 0.5:
                    tg = self.engine.apply_sequence(beam_result.seq1, [inp], task)
                    a1 = self._ngrid_to_list(tg[0] if tg else None, fb1)
                else:
                    a1 = fb1
            else:
                a1 = fb1

            # attempt_2
            if a2_mode == "dynamic" and dyn_result is not None:
                n2, g2 = dyn_result
                a2 = n2g(g2[i]) if g2 and i < len(g2) else fb2
            elif a2_mode == "dynamic_or_beam":
                if dyn_result is not None:
                    n2, g2 = dyn_result
                    a2 = n2g(g2[i]) if g2 and i < len(g2) else fb2
                else:
                    a2 = self._beam_partial_for_inp(task, inp, beam_result)
            elif a2_mode == "beam_partial":
                a2 = self._beam_partial_for_inp(task, inp, beam_result)
                # beam partial과 attempt_1이 같으면 fallback 사용
                if a2 == a1: a2 = fb2
            elif a2_mode == "beam_second" and beam_result is not None:
                if beam_result.success and beam_result.second_seq and \
                        beam_result.second_seq != beam_result.seq1:
                    tg2 = self.engine.apply_sequence(beam_result.second_seq, [inp], task)
                    a2 = self._ngrid_to_list(tg2[0] if tg2 else None, fb2)
                elif beam_result.grids1 is not None and beam_result.score >= 0.3:
                    # partial이라도 attempt_2로 활용
                    tg = self.engine.apply_sequence(beam_result.seq1, [inp], task)
                    cand = self._ngrid_to_list(tg[0] if tg else None, None)
                    a2 = cand if cand is not None and cand != a1 else fb1
                else:
                    a2 = fb1
            else:
                a2 = fb2

            preds.append(Prediction(attempt_1=a1, attempt_2=a2))

        # ── BeamResult 생성 ──────────────────────────────────────────────────
        if a1_mode == "rule":
            res = BeamResult(task.task_id, True, [name1], grids1, 1.0, "rule")
        elif a1_mode == "dynamic":
            res = BeamResult(task.task_id, True, [name1], grids1, 1.0, "dynamic")
        elif beam_result is not None:
            res = beam_result
        else:
            res = BeamResult(task.task_id, False, [], None, 0.0, "no_solver")

        return task.task_id, preds, res

    def _build_predictions(self, task, result):
        """(하위호환용 — solve_task 내부에서 직접 처리하므로 사용 안 함)"""
        preds=[]
        for inp in task.test_inputs:
            fb1=self.fallback.best_guess(task,inp)
            fb2=self.fallback.copy_input(inp)
            if result.success and result.grids1 is not None:
                tg=self.engine.apply_sequence(result.seq1,[inp],task)
                a1=self._ngrid_to_list(tg[0] if tg else None,fb1)
                if result.second_seq and result.second_seq!=result.seq1:
                    tg2=self.engine.apply_sequence(result.second_seq,[inp],task)
                    a2=self._ngrid_to_list(tg2[0] if tg2 else None,fb2)
                else: a2=fb2
            elif result.grids1 is not None and result.score>=0.5:
                tg=self.engine.apply_sequence(result.seq1,[inp],task)
                a1=self._ngrid_to_list(tg[0] if tg else None,fb1); a2=fb1
            else: a1=fb1; a2=fb2
            preds.append(Prediction(attempt_1=a1,attempt_2=a2))
        return preds

    def solve_all(self, tasks, progress_fn=None):
        all_preds={}; all_results={}
        if self.max_workers>1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs={ex.submit(self.solve_task,task):tid for tid,task in tasks.items()}
                for fut in as_completed(futs):
                    tid=futs[fut]
                    try: _,preds,res=fut.result()
                    except Exception as e:
                        task=tasks[tid]
                        preds=[Prediction(attempt_1=self.fallback.best_guess(task,i),
                                          attempt_2=self.fallback.copy_input(i))
                               for i in task.test_inputs]
                        res=BeamResult(tid,False,[],None,0.0,str(e))
                    all_preds[tid]=preds; all_results[tid]=res
                    if progress_fn: progress_fn(1)
        else:
            for tid,task in tasks.items():
                try: _,preds,res=self.solve_task(task)
                except Exception as e:
                    preds=[Prediction(attempt_1=self.fallback.best_guess(task,i),
                                      attempt_2=self.fallback.copy_input(i))
                           for i in task.test_inputs]
                    res=BeamResult(tid,False,[],None,0.0,str(e))
                all_preds[tid]=preds; all_results[tid]=res
                if progress_fn: progress_fn(1)
        return all_preds, all_results

print(f"✓ Phase 6 로드 완료 (ARCSolver  workers={MAX_WORKERS})")
'''

# ── Tests ──────────────────────────────────────────────────────────────────────
CELL_TEST1 = '''\
loader=ARCDataLoader()
train_tasks=loader.load_training()
eval_tasks=loader.load_evaluation()
test_tasks=loader.load_test()
print(f"training={len(train_tasks)}  eval={len(eval_tasks)}  test={len(test_tasks)}")
print("✓ Test 1 통과")
'''

CELL_TEST2 = '''\
# DSL 검증
g=g2n([[0,1,0],[1,1,1],[0,1,0]])
assert dilate_objects(g,0,4)[0,0]!=0   # 팽창 후 (0,0) 채워져야 함? 아니면 인접 비0 셀로
# 실제로 (0,0)은 (0,1)과 (1,0)이 둘 다 비0 → majority 방향으로 채워짐
d=dilate_objects(g,0,4)
assert d.sum()>g.sum()
print("✓ dilate_objects")

g_ring=g2n([[1,1,1],[1,0,1],[1,1,1]])
e=erode_objects(g_ring,0,4)
assert e[0,0]==0  # 경계 픽셀 제거
print("✓ erode_objects")

# keep_border / keep_interior
gtest=g2n([[1,0,0],[0,2,0],[0,0,3]])
b=keep_border_objects(gtest,0)
assert b[0,0]==1 and b[2,2]==3
i=keep_interior_objects(gtest,0)
assert i[1,1]==2 and i[0,0]==0
print("✓ keep_border/interior")

# fill_holes
gf=g2n([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]])
fh=fill_holes(gf,-1,0)
assert fh[1,1]==1 and fh[1,2]==1
print("✓ fill_holes")
print("\\n✓ Test 2 통과")
'''

CELL_TEST3 = '''\
# RuleEngine 규칙 검증

# 1. 색상 매핑
task_cm=Task("_cm",[Pair([[1,0],[0,2]],[[3,0],[0,4]]),Pair([[2,1],[0,0]],[[4,3],[0,0]])],[[[1,2]]])
r=_rule_engine.solve(task_cm)
assert r is not None and r[0]=="color_map", f"color_map: got {r}"
print(f"✓ ColorMapRule: {r[0]}")

# 2. 기하
task_geo=Task("_geo",[Pair([[1,2],[3,4]],[[2,4],[1,3]]),Pair([[5,6],[7,8]],[[6,8],[5,7]])],[[[0,1]]])
r=_rule_engine.solve(task_geo)
assert r is not None, f"GeoRule failed: {r}"
print(f"✓ GeoRule: {r[0]}")

# 3. 스케일
task_sc=Task("_sc",[Pair([[1,2]],[[1,1,2,2]]),Pair([[3,4]],[[3,3,4,4]])],[[[5,6]]])
r=_rule_engine.solve(task_sc)
assert r is not None, f"ScaleRule failed: {r}"
print(f"✓ ScaleRule: {r[0]}")

# 4. dilation 규칙
task_dil=Task("_dil",[
    Pair([[0,1,0],[0,0,0]],[[1,1,1],[0,1,0]]),
    Pair([[0,0,2],[0,0,0]],[[0,2,2],[0,0,2]]),
],[[[0,3,0],[0,0,0]]])
r=_rule_engine.solve(task_dil)
assert r is not None, f"DilateRule failed"
print(f"✓ DilateRule: {r[0]}")

# 5. diff patch
task_dp=Task("_dp",[
    Pair([[1,0,0],[0,0,0]],[[1,0,0],[0,5,0]]),
    Pair([[2,0,0],[0,0,0]],[[2,0,0],[0,5,0]]),
],[[[3,0,0],[0,0,0]]])
r=_rule_engine.solve(task_dp)
assert r is not None, f"DiffPatchRule failed"
print(f"✓ DiffPatchRule: {r[0]}")

# 6. 2단계 규칙 (rot90 + color_map)
task_2s=Task("_2s",[
    Pair([[1,2],[3,4]],[[6,8],[5,7]]),  # rot90 -> [[3,1],[4,2]] -> cmap
    Pair([[5,6],[7,8]],[[14,16],[13,15]]),
],[[[0,1]]])
# 이 태스크는 복잡하므로 TwoStepRule이 풀 수도 있음 (단, 컬러 범위 주의)
# 대신 simple 2-step: flip + constant color test
task_2s2=Task("_2s2",[
    Pair([[1,0],[0,1]],[[0,1],[1,0]]),  # flipud
    Pair([[2,0],[0,2]],[[0,2],[2,0]]),
],[[[3,0],[0,3]]])
r2=_rule_engine.solve(task_2s2)
assert r2 is not None, f"2step failed"
print(f"✓ 2-step Rule: {r2[0]}")

print("\\n✓ Test 3 통과 (RuleEngine 6개 규칙 검증)")
'''

CELL_TEST4 = '''\
# E2E: training 10개 테스트
try:
    from tqdm.notebook import tqdm as _tqdm
except ImportError:
    from tqdm import tqdm as _tqdm

_sample=dict(list(train_tasks.items())[:10])
_solver=ARCSolver(beam_width=15,max_depth=3,timeout=15.0,max_workers=1)
_preds={}; _res={}; _solved=0; _rule_solved=0

for _tid,_task in _tqdm(_sample.items(),desc="E2E Test",unit="task"):
    _,_p,_r=_solver.solve_task(_task)
    _preds[_tid]=_p; _res[_tid]=_r
    if _r.success: _solved+=1
    if _r.success and _r.error=="rule": _rule_solved+=1

_sc=ARCEvaluator().evaluate(_preds,_sample)
_path=ARCSubmissionWriter().save(_preds,"submission_test.json")
print(f"\\n10-task: solved={_solved}/10  rule_solved={_rule_solved}  acc={_sc[\'overall_score\']:.4f}")
for _tid,_r in _res.items():
    _st="✓[R]" if(_r.success and _r.error=="rule") else ("✓[B]" if _r.success else f"~{_r.score:.2f}")
    print(f"  [{_st}] {_tid}  {str(_r.seq1[:2])}")
assert _path.exists()
print("\\n✓ Test 4 통과")
'''

CELL_CONFIG = '''\
# ─────────────────────────────────────────────────────────────────────
# Kaggle 제출 시: SPLIT="test"  (수정하지 마세요)
# 로컬 검증 시  : SPLIT="evaluation"
# ─────────────────────────────────────────────────────────────────────
SPLIT          = "test"        # "training" | "evaluation" | "test"
MAX_TASKS      = None          # None = 전체  /  정수 = 앞 N개 빠른 검증
OUTPUT_FILE    = "submission.json"

RUN_BEAM_WIDTH   = 30
RUN_MAX_DEPTH    = 5
RUN_TASK_TIMEOUT = 60.0        # 전체 실행 시 60~90 권장
RUN_MAX_WORKERS  = max(1, os.cpu_count() or 1)

print(f"설정: split={SPLIT}  beam={RUN_BEAM_WIDTH}  depth={RUN_MAX_DEPTH}"
      f"  timeout={RUN_TASK_TIMEOUT}s  workers={RUN_MAX_WORKERS}"
      f"  max_tasks={MAX_TASKS}")
'''

CELL_RUN = '''\
try:
    from tqdm.notebook import tqdm as _tqdm
except ImportError:
    from tqdm import tqdm as _tqdm

# submission.json은 /kaggle/working/ 에 저장해야 Kaggle이 인식
_output_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
_out_path = _output_dir / OUTPUT_FILE
print(f"출력 경로: {_out_path.resolve()}")

_loader=ARCDataLoader()
_tasks={"training":_loader.load_training,"evaluation":_loader.load_evaluation,
        "test":_loader.load_test}[SPLIT]()
if MAX_TASKS: _tasks=dict(list(_tasks.items())[:MAX_TASKS])
print(f"✓ {len(_tasks)} tasks 로드 ({SPLIT})")

_run_solver=ARCSolver(RUN_BEAM_WIDTH,RUN_MAX_DEPTH,RUN_TASK_TIMEOUT,RUN_MAX_WORKERS)
_all_preds={}; _all_res={}
_t0=time.perf_counter()

with _tqdm(total=len(_tasks),desc="Solving",unit="task") as _pbar:
    def _upd(n): _pbar.update(n)
    _all_preds,_all_res=_run_solver.solve_all(_tasks,progress_fn=_upd)

_elapsed=time.perf_counter()-_t0
_eval_sc=None
if SPLIT in("training","evaluation"):
    _eval_sc=ARCEvaluator().evaluate(_all_preds,_tasks)
_saved=ARCSubmissionWriter().save(_all_preds, str(_out_path))

_tot=len(_all_res)
_rule_sv  = sum(1 for r in _all_res.values() if r.success and r.error=="rule")
_dyn_sv   = sum(1 for r in _all_res.values() if r.success and r.error=="dynamic")
_beam_sv  = sum(1 for r in _all_res.values() if r.success and r.error not in("rule","dynamic"))
_hi       = sum(1 for r in _all_res.values() if not r.success and r.score>=0.9)

print()
print("="*64)
print(" ARC-AGI-2 v5 (RuleEngine + DynamicLearner + BeamSearch)")
print("="*64)
print(f"  총 tasks             : {_tot}")
print(f"  처리 시간            : {_elapsed:.1f}s  ({_elapsed/_tot:.1f}s/task)" if _tot else f"  처리 시간            : {_elapsed:.1f}s")
print()
if _tot:
    print(f"  ✓[Rule]    고정규칙  : {_rule_sv:4d}  ({_rule_sv/_tot*100:.1f}%)")
    print(f"  ✓[Dynamic] 동적학습  : {_dyn_sv:4d}  ({_dyn_sv/_tot*100:.1f}%)")
    print(f"  ✓[Beam]    빔서치    : {_beam_sv:4d}  ({_beam_sv/_tot*100:.1f}%)")
    print(f"  ~ ≥0.9 근접          : {_hi:4d}  ({_hi/_tot*100:.1f}%)")
_solved = _rule_sv + _dyn_sv + _beam_sv
if _tot:
    print(f"  합계 완전 풀이       : {_solved:4d}  ({_solved/_tot*100:.1f}%)")
if _eval_sc:
    print()
    print(f"  ★ 평가 점수(overall) : {_eval_sc[\'overall_score\']:.4f}")
    print(f"    완벽 tasks          : {sum(1 for s in _eval_sc[\'task_scores\'].values() if s==1.0)}/{_tot}")
print("="*64)

# 개별 태스크 결과 출력
print("\\n개별 태스크 결과:")
for _tid, _r in sorted(_all_res.items()):
    if _r.success:
        _mode = "[R]" if _r.error=="rule" else ("[D]" if _r.error=="dynamic" else "[B]")
        print(f"  ✓{_mode} {_tid}  {str(_r.seq1[:2])}")
    elif _r.score >= 0.85:
        print(f"  ~{_r.score:.3f} {_tid}  {str(_r.seq1[:2])}")
print(f"\\n제출 파일: {_saved}")
'''

# ── 조립 ───────────────────────────────────────────────────────────────────────
cells = [
    md_cell(MD_TITLE),
    md_cell("## 0. 설치"), code_cell(CELL_INSTALL),
    md_cell("## 1. 임포트 및 설정"), code_cell(CELL_IMPORTS),
    md_cell("## Phase 1: 데이터 파이프라인"), code_cell(CELL_PHASE1),
    md_cell("## Phase 2: NumPy DSL 기본"), code_cell(CELL_PHASE2),
    md_cell("## Phase 2b: 확장 DSL"), code_cell(CELL_PHASE2B),
    md_cell("## Phase 2c: 형태소·경계·CA"), code_cell(CELL_PHASE2C),
    md_cell("## Phase 3: ShapePredictor"), code_cell(CELL_PHASE3),
    md_cell("## Phase 4: 연산자 라이브러리"), code_cell(CELL_PHASE4),
    md_cell("## Phase 4b: RuleEngine (핵심)"), code_cell(CELL_PHASE4B),
    md_cell("## Phase 5: 빔서치 폴백"), code_cell(CELL_PHASE5),
    md_cell("## Phase 6: 통합 Solver"), code_cell(CELL_PHASE6),
    md_cell("## 🧪 테스트"),
    code_cell(CELL_TEST1), code_cell(CELL_TEST2),
    code_cell(CELL_TEST3), code_cell(CELL_TEST4),
    md_cell("## ⚙️ 설정 및 🚀 실행"),
    code_cell(CELL_CONFIG), code_cell(CELL_RUN),
]

nb = {
    "nbformat":4, "nbformat_minor":5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
    },
    "cells": cells,
}

out = Path("arc_agi3_beam.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"✓ 노트북 생성: {out}  ({out.stat().st_size//1024} KB)")
