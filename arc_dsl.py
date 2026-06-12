"""
ARC-AGI-2 Grid DSL (Domain-Specific Library)
=============================================
격자(Grid) 분석 및 조작을 위한 헬퍼 함수 라이브러리.
"""

from __future__ import annotations

import copy
from collections import Counter, deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator, Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Grid = list[list[int]]
Point = tuple[int, int]           # (row, col)
Connectivity = Literal[4, 8]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _copy(grid: Grid) -> Grid:
    return copy.deepcopy(grid)


def _validate(grid: Grid) -> None:
    if not grid or not grid[0]:
        raise ValueError("빈 Grid 는 허용되지 않습니다.")
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("모든 행의 길이가 같아야 합니다.")


def _rows(grid: Grid) -> int:
    return len(grid)


def _cols(grid: Grid) -> int:
    return len(grid[0]) if grid else 0


# ---------------------------------------------------------------------------
# 1. GeometricOps
# ---------------------------------------------------------------------------

class GeometricOps:
    """회전 / 대칭 / 크기 조절 관련 연산."""

    @staticmethod
    def rotate_90(grid: Grid) -> Grid:
        """시계 방향 90도 회전."""
        _validate(grid)
        g = _copy(grid)
        return [list(row) for row in zip(*g[::-1])]

    @staticmethod
    def rotate_180(grid: Grid) -> Grid:
        """180도 회전."""
        _validate(grid)
        g = _copy(grid)
        return [row[::-1] for row in g[::-1]]

    @staticmethod
    def rotate_270(grid: Grid) -> Grid:
        """시계 방향 270도 회전."""
        _validate(grid)
        g = _copy(grid)
        return [list(row) for row in zip(*g)][::-1]

    @staticmethod
    def rotate(grid: Grid, degrees: int) -> Grid:
        """degrees ∈ {0, 90, 180, 270} 만큼 시계 방향 회전."""
        degrees = degrees % 360
        ops = {
            0:   lambda g: _copy(g),
            90:  GeometricOps.rotate_90,
            180: GeometricOps.rotate_180,
            270: GeometricOps.rotate_270,
        }
        if degrees not in ops:
            raise ValueError(f"degrees 는 0/90/180/270 중 하나여야 합니다: {degrees}")
        return ops[degrees](grid)

    @staticmethod
    def flip_horizontal(grid: Grid) -> Grid:
        """좌우 반전."""
        _validate(grid)
        return [row[::-1] for row in _copy(grid)]

    @staticmethod
    def flip_vertical(grid: Grid) -> Grid:
        """상하 반전."""
        _validate(grid)
        return _copy(grid)[::-1]

    @staticmethod
    def flip_diagonal_main(grid: Grid) -> Grid:
        """주대각선 반전 (전치)."""
        _validate(grid)
        g = _copy(grid)
        return [list(row) for row in zip(*g)]

    @staticmethod
    def flip_diagonal_anti(grid: Grid) -> Grid:
        """부대각선 반전."""
        _validate(grid)
        g = _copy(grid)
        transposed = [list(row) for row in zip(*g)]
        return [row[::-1] for row in transposed[::-1]]

    @staticmethod
    def scale_up(grid: Grid, factor: int) -> Grid:
        """각 셀을 factor × factor 블록으로 확대."""
        _validate(grid)
        if factor < 1:
            raise ValueError(f"factor 는 1 이상이어야 합니다: {factor}")
        result: Grid = []
        for row in grid:
            scaled_row = []
            for cell in row:
                scaled_row.extend([cell] * factor)
            for _ in range(factor):
                result.append(list(scaled_row))
        return result

    @staticmethod
    def scale_down(grid: Grid, factor: int) -> Grid:
        """factor 픽셀마다 하나를 추출하여 축소."""
        _validate(grid)
        if factor < 1:
            raise ValueError(f"factor 는 1 이상이어야 합니다: {factor}")
        return [
            [grid[r][c] for c in range(0, _cols(grid), factor)]
            for r in range(0, _rows(grid), factor)
        ]

    @staticmethod
    def pad(
        grid: Grid,
        top: int = 0, bottom: int = 0,
        left: int = 0, right: int = 0,
        fill: int = 0,
    ) -> Grid:
        """Grid 주변에 fill 값으로 패딩을 추가한다."""
        _validate(grid)
        if any(v < 0 for v in (top, bottom, left, right)):
            raise ValueError("패딩 값은 0 이상이어야 합니다.")
        g = _copy(grid)
        cols_new = _cols(g) + left + right
        top_rows    = [[fill] * cols_new for _ in range(top)]
        bottom_rows = [[fill] * cols_new for _ in range(bottom)]
        middle = [[fill] * left + row + [fill] * right for row in g]
        return top_rows + middle + bottom_rows

    @staticmethod
    def crop(grid: Grid, r1: int, c1: int, r2: int, c2: int) -> Grid:
        """[r1:r2, c1:c2] 영역을 잘라낸다."""
        _validate(grid)
        r2 = min(r2, _rows(grid))
        c2 = min(c2, _cols(grid))
        if r1 < 0 or c1 < 0 or r1 >= r2 or c1 >= c2:
            raise ValueError(f"유효하지 않은 crop 범위: ({r1},{c1})-({r2},{c2})")
        return [row[c1:c2] for row in grid[r1:r2]]

    @staticmethod
    def trim(grid: Grid, background: int = 0) -> Grid:
        """배경색 테두리 행/열을 제거한다."""
        _validate(grid)
        g = _copy(grid)
        R, C = _rows(g), _cols(g)

        def is_bg_row(r: int) -> bool:
            return all(g[r][c] == background for c in range(C))

        def is_bg_col(c: int) -> bool:
            return all(g[r][c] == background for r in range(R))

        top = next((r for r in range(R) if not is_bg_row(r)), R)
        bot = next((r for r in range(R - 1, -1, -1) if not is_bg_row(r)), -1) + 1
        lft = next((c for c in range(C) if not is_bg_col(c)), C)
        rgt = next((c for c in range(C - 1, -1, -1) if not is_bg_col(c)), -1) + 1

        if top >= bot or lft >= rgt:
            raise ValueError("trim 후 빈 Grid 가 됩니다.")
        return [row[lft:rgt] for row in g[top:bot]]

    @staticmethod
    def overlay(base: Grid, patch: Grid, r: int, c: int, transparent: int | None = None) -> Grid:
        """base 위의 (r, c) 위치에 patch 를 덮어씌운다."""
        _validate(base)
        _validate(patch)
        result = _copy(base)
        for dr in range(_rows(patch)):
            for dc in range(_cols(patch)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < _rows(result) and 0 <= nc < _cols(result):
                    val = patch[dr][dc]
                    if transparent is None or val != transparent:
                        result[nr][nc] = val
        return result


# ---------------------------------------------------------------------------
# 2. ObjectOps
# ---------------------------------------------------------------------------

@dataclass
class ArcObject:
    """Grid 내의 단일 연결 컴포넌트(오브젝트)."""
    color: int
    cells: list[Point]
    bounding_box: tuple[int, int, int, int]

    @property
    def height(self) -> int:
        r_min, _, r_max, _ = self.bounding_box
        return r_max - r_min + 1

    @property
    def width(self) -> int:
        _, c_min, _, c_max = self.bounding_box
        return c_max - c_min + 1

    @property
    def size(self) -> int:
        return len(self.cells)

    def to_grid(self, background: int = 0) -> Grid:
        r_min, c_min, r_max, c_max = self.bounding_box
        h, w = r_max - r_min + 1, c_max - c_min + 1
        g: Grid = [[background] * w for _ in range(h)]
        for (r, c) in self.cells:
            g[r - r_min][c - c_min] = self.color
        return g


class ObjectOps:
    """연결 컴포넌트(오브젝트) 감지 및 추출 관련 연산."""

    _DIRS_4: list[Point] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _DIRS_8: list[Point] = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           (0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    @classmethod
    def _directions(cls, connectivity: Connectivity) -> list[Point]:
        return cls._DIRS_4 if connectivity == 4 else cls._DIRS_8

    @classmethod
    def find_objects(
        cls,
        grid: Grid,
        background: int = 0,
        connectivity: Connectivity = 4,
    ) -> list[ArcObject]:
        _validate(grid)
        R, C = _rows(grid), _cols(grid)
        visited = [[False] * C for _ in range(R)]
        dirs = cls._directions(connectivity)
        objects: list[ArcObject] = []

        for r in range(R):
            for c in range(C):
                if visited[r][c] or grid[r][c] == background:
                    continue
                color = grid[r][c]
                cells: list[Point] = []
                queue: deque[Point] = deque([(r, c)])
                visited[r][c] = True

                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                bbox = (min(rs), min(cs), max(rs), max(cs))
                objects.append(ArcObject(color=color, cells=cells, bounding_box=bbox))

        return objects

    @classmethod
    def label_grid(
        cls,
        grid: Grid,
        background: int = 0,
        connectivity: Connectivity = 4,
    ) -> list[list[int]]:
        _validate(grid)
        R, C = _rows(grid), _cols(grid)
        labels = [[0] * C for _ in range(R)]
        visited = [[False] * C for _ in range(R)]
        dirs = cls._directions(connectivity)
        label = 0

        for r in range(R):
            for c in range(C):
                if visited[r][c] or grid[r][c] == background:
                    continue
                label += 1
                color = grid[r][c]
                queue: deque[Point] = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    labels[cr][cc] = label
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < R and 0 <= nc < C
                                and not visited[nr][nc]
                                and grid[nr][nc] == color):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

        return labels

    @classmethod
    def isolate_shapes(
        cls,
        grid: Grid,
        background: int = 0,
        connectivity: Connectivity = 4,
    ) -> list[Grid]:
        objects = cls.find_objects(grid, background=background, connectivity=connectivity)
        return [obj.to_grid(background=background) for obj in objects]

    @classmethod
    def get_objects_by_color(
        cls,
        grid: Grid,
        color: int,
        connectivity: Connectivity = 4,
    ) -> list[ArcObject]:
        return [
            obj for obj in cls.find_objects(grid, background=0, connectivity=connectivity)
            if obj.color == color
        ]

    @classmethod
    def count_objects(
        cls,
        grid: Grid,
        background: int = 0,
        connectivity: Connectivity = 4,
    ) -> int:
        return len(cls.find_objects(grid, background=background, connectivity=connectivity))

    @classmethod
    def bounding_box_grid(cls, grid: Grid, background: int = 0) -> tuple[int, int, int, int]:
        _validate(grid)
        R, C = _rows(grid), _cols(grid)
        rows = [r for r in range(R) for c in range(C) if grid[r][c] != background]
        cols = [c for r in range(R) for c in range(C) if grid[r][c] != background]
        if not rows:
            raise ValueError("배경이 아닌 셀이 없습니다.")
        return (min(rows), min(cols), max(rows), max(cols))


# ---------------------------------------------------------------------------
# 3. ColorOps
# ---------------------------------------------------------------------------

class ColorOps:
    """색상 분석 및 변환 관련 연산."""

    @staticmethod
    def color_counts(grid: Grid) -> Counter:
        _validate(grid)
        return Counter(cell for row in grid for cell in row)

    @staticmethod
    def most_common_color(grid: Grid, exclude: set[int] | None = None) -> int:
        counts = ColorOps.color_counts(grid)
        if exclude:
            for c in exclude:
                counts.pop(c, None)
        if not counts:
            raise ValueError("제외 후 색상이 없습니다.")
        return counts.most_common(1)[0][0]

    @staticmethod
    def least_common_color(grid: Grid, exclude: set[int] | None = None) -> int:
        counts = ColorOps.color_counts(grid)
        if exclude:
            for c in exclude:
                counts.pop(c, None)
        if not counts:
            raise ValueError("제외 후 색상이 없습니다.")
        return counts.most_common()[-1][0]

    @staticmethod
    def unique_colors(grid: Grid) -> list[int]:
        _validate(grid)
        return sorted({cell for row in grid for cell in row})

    @staticmethod
    def replace_color(grid: Grid, old_color: int, new_color: int) -> Grid:
        _validate(grid)
        return [
            [new_color if cell == old_color else cell for cell in row]
            for row in grid
        ]

    @staticmethod
    def flood_fill(
        grid: Grid,
        start: Point,
        new_color: int,
        connectivity: Connectivity = 4,
    ) -> Grid:
        _validate(grid)
        R, C = _rows(grid), _cols(grid)
        r0, c0 = start
        if not (0 <= r0 < R and 0 <= c0 < C):
            raise ValueError(f"start {start} 가 Grid 범위를 벗어납니다.")

        old_color = grid[r0][c0]
        if old_color == new_color:
            return _copy(grid)

        result = _copy(grid)
        dirs = ObjectOps._DIRS_4 if connectivity == 4 else ObjectOps._DIRS_8
        queue: deque[Point] = deque([(r0, c0)])
        result[r0][c0] = new_color

        while queue:
            r, c = queue.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and result[nr][nc] == old_color:
                    result[nr][nc] = new_color
                    queue.append((nr, nc))

        return result

    @staticmethod
    def color_mask(grid: Grid, color: int) -> list[list[bool]]:
        _validate(grid)
        return [[cell == color for cell in row] for row in grid]

    @staticmethod
    def apply_palette(grid: Grid, mapping: dict[int, int]) -> Grid:
        _validate(grid)
        return [
            [mapping.get(cell, cell) for cell in row]
            for row in grid
        ]


# ---------------------------------------------------------------------------
# 4. SizeAnalyzer
# ---------------------------------------------------------------------------

@dataclass
class SizeRelationship:
    """input → output 크기 관계를 요약한다."""
    input_sizes:  list[tuple[int, int]]
    output_sizes: list[tuple[int, int]]

    is_fixed_output:   bool = False
    fixed_output_size: tuple[int, int] | None = None

    is_constant_scale: bool = False
    row_scale: Fraction | None = None
    col_scale: Fraction | None = None

    is_square_preserved: bool = False
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["[SizeRelationship]"]
        if self.is_fixed_output:
            lines.append(f"  고정 출력 크기: {self.fixed_output_size}")
        if self.is_constant_scale:
            lines.append(f"  상수 배율: rows×{self.row_scale}, cols×{self.col_scale}")
        if self.is_square_preserved:
            lines.append("  정방 Grid 보존")
        for note in self.notes:
            lines.append(f"  {note}")
        return "\n".join(lines)


class SizeAnalyzer:
    """학습 예제의 input/output 크기 관계를 분석한다."""

    @staticmethod
    def analyze(
        train_inputs:  list[Grid],
        train_outputs: list[Grid],
    ) -> SizeRelationship:
        if len(train_inputs) != len(train_outputs):
            raise ValueError("inputs 와 outputs 의 수가 달라야 합니다.")
        if not train_inputs:
            raise ValueError("최소 하나의 학습 예제가 필요합니다.")

        in_sizes  = [(_rows(g), _cols(g)) for g in train_inputs]
        out_sizes = [(_rows(g), _cols(g)) for g in train_outputs]
        rel = SizeRelationship(input_sizes=in_sizes, output_sizes=out_sizes)

        if len(set(out_sizes)) == 1:
            rel.is_fixed_output = True
            rel.fixed_output_size = out_sizes[0]
            rel.notes.append(f"모든 출력이 동일 크기: {out_sizes[0]}")

        row_scales = [Fraction(o[0], i[0]) for i, o in zip(in_sizes, out_sizes)]
        col_scales = [Fraction(o[1], i[1]) for i, o in zip(in_sizes, out_sizes)]

        if len(set(row_scales)) == 1 and len(set(col_scales)) == 1:
            rel.is_constant_scale = True
            rel.row_scale = row_scales[0]
            rel.col_scale = col_scales[0]
            rel.notes.append(
                f"일정 배율: rows×{rel.row_scale}, cols×{rel.col_scale}"
            )

        if all(i == o for i, o in zip(in_sizes, out_sizes)):
            rel.notes.append("입출력 크기 동일 (identity size)")

        if all(i == (o[1], o[0]) for i, o in zip(in_sizes, out_sizes)):
            rel.notes.append("행↔열 교환 (전치 크기)")

        if all(r == c for r, c in in_sizes) and all(r == c for r, c in out_sizes):
            rel.is_square_preserved = True
            rel.notes.append("입출력 모두 정방 Grid")

        if all(o[0] <= i[0] and o[1] <= i[1] for i, o in zip(in_sizes, out_sizes)):
            if not all(i == o for i, o in zip(in_sizes, out_sizes)):
                rel.notes.append("출력이 입력보다 작거나 같음 (crop 가능성)")

        return rel

    @staticmethod
    def predict_output_size(
        test_input: Grid,
        rel: SizeRelationship,
    ) -> tuple[int, int] | None:
        if rel.is_fixed_output and rel.fixed_output_size:
            return rel.fixed_output_size

        if rel.is_constant_scale and rel.row_scale and rel.col_scale:
            r = _rows(test_input)
            c = _cols(test_input)
            pred_r = int(r * rel.row_scale)
            pred_c = int(c * rel.col_scale)
            if pred_r > 0 and pred_c > 0:
                return (pred_r, pred_c)

        return None


# ---------------------------------------------------------------------------
# 5. GridDSL
# ---------------------------------------------------------------------------

class GridDSL:
    """Grid DSL 의 통합 진입점."""

    geo   = GeometricOps()
    obj   = ObjectOps()
    color = ColorOps()

    def __init__(self, grid: Grid) -> None:
        _validate(grid)
        self.grid: Grid = _copy(grid)

    @property
    def shape(self) -> tuple[int, int]:
        return (_rows(self.grid), _cols(self.grid))

    def __repr__(self) -> str:
        R, C = self.shape
        return f"GridDSL(shape={R}×{C})"

    def display(self) -> str:
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)

    def rotate(self, degrees: int) -> "GridDSL":
        return GridDSL(self.geo.rotate(self.grid, degrees))

    def flip_h(self) -> "GridDSL":
        return GridDSL(self.geo.flip_horizontal(self.grid))

    def flip_v(self) -> "GridDSL":
        return GridDSL(self.geo.flip_vertical(self.grid))

    def scale_up(self, factor: int) -> "GridDSL":
        return GridDSL(self.geo.scale_up(self.grid, factor))

    def flood_fill(self, start: Point, new_color: int) -> "GridDSL":
        return GridDSL(self.color.flood_fill(self.grid, start, new_color))

    def objects(
        self,
        background: int = 0,
        connectivity: Connectivity = 4,
    ) -> list[ArcObject]:
        return self.obj.find_objects(self.grid, background=background, connectivity=connectivity)

    def analyze_sizes(self, train_outputs: list[Grid]) -> SizeRelationship:
        return SizeAnalyzer.analyze([self.grid], train_outputs)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    print("=== ARC DSL 자체 테스트 ===\n")

    g: Grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    r90  = GeometricOps.rotate_90(g)
    r180 = GeometricOps.rotate_180(g)
    r270 = GeometricOps.rotate_270(g)
    assert GeometricOps.rotate_90(r90)  == r180
    assert GeometricOps.rotate_90(r180) == r270
    assert GeometricOps.rotate_90(r270) == g
    print("✓ 회전 (90/180/270)")

    fh = GeometricOps.flip_horizontal(g)
    assert fh[0] == [3, 2, 1]
    fv = GeometricOps.flip_vertical(g)
    assert fv[0] == [7, 8, 9]
    fd = GeometricOps.flip_diagonal_main(g)
    assert fd[0] == [1, 4, 7]
    print("✓ 대칭 (좌우/상하/대각선)")

    padded = GeometricOps.pad(g, top=1, left=1)
    assert padded[0] == [0, 0, 0, 0]
    cropped = GeometricOps.crop(g, 0, 0, 2, 2)
    assert cropped == [[1, 2], [4, 5]]
    sparse: Grid = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    trimmed = GeometricOps.trim(sparse)
    assert trimmed == [[5]]
    print("✓ 패딩 / crop / trim")

    grid2: Grid = [[1, 1, 0, 2], [1, 0, 0, 2], [0, 0, 3, 0]]
    objs = ObjectOps.find_objects(grid2, background=0, connectivity=4)
    assert len(objs) == 3
    assert {o.color for o in objs} == {1, 2, 3}
    print("✓ 오브젝트 감지 (CCL)")

    filled = ColorOps.flood_fill(grid2, (0, 0), 9)
    assert filled[0][0] == 9
    assert filled[1][0] == 9
    assert filled[0][2] == 0
    print("✓ Flood fill")

    grid3: Grid = [[1, 1, 2], [2, 2, 3]]
    assert ColorOps.most_common_color(grid3) == 2
    assert ColorOps.least_common_color(grid3) == 3
    print("✓ 색상 카운팅")

    inputs  = [
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ]
    outputs = [
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
        [[1]*6, [1]*6, [1]*6, [1]*6, [1]*6, [1]*6],
    ]
    rel = SizeAnalyzer.analyze(inputs, outputs)
    assert rel.is_constant_scale, f"expected constant scale, got {rel}"
    assert rel.row_scale == Fraction(2)
    assert rel.col_scale == Fraction(2)
    pred = SizeAnalyzer.predict_output_size([[0, 0, 0], [0, 0, 0]], rel)
    assert pred == (4, 6), f"pred={pred}"
    rel_fixed = SizeAnalyzer.analyze([[[0, 0], [0, 0]]], [[[1, 1], [1, 1]]])
    assert rel_fixed.is_fixed_output
    print("✓ 크기 분석 및 예측")

    dsl = GridDSL(g)
    result = dsl.rotate(90).flip_h().grid
    assert result is not None
    print("✓ GridDSL 체이닝")

    print("\n모든 테스트 통과!")


if __name__ == "__main__":
    _run_tests()
