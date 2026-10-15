"""
Circle initialization: SIFT keypoints + Poisson disk gap-fill.

SIFT finds natural scale-space extrema — regions where there is meaningful
structure at a specific scale. These make ideal initial circle seeds because
they cluster in detail-rich areas with radii matching local feature scale.

Poisson disk gap-fill ensures the flat/smooth regions also get coverage,
using larger circles. Together these give a physically-covering initial set
that the optimizer then refines.
"""

import numpy as np
import cv2
import math


def sift_seeds(
    gray: np.ndarray,
    r_min: float,
    r_max: float,
    contrast_threshold: float = 0.02,
    edge_threshold: float = 10.0,
) -> list[tuple[float, float, float]]:
    """
    Detect SIFT keypoints and convert to (x, y, r) circles.
    gray: (H, W) float32 [0–1]
    Returns circles with r clamped to [r_min, r_max].
    """
    gray_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
    sift = cv2.SIFT_create(
        nfeatures=0,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=1.6,
    )
    kps = sift.detect(gray_u8, None)
    circles = []
    for kp in kps:
        x, y = kp.pt
        # SIFT size ≈ diameter of the feature region
        r = float(np.clip(kp.size / 2.5, r_min, r_max))
        circles.append((x, y, r))
    return circles


class SpatialGrid:
    """Uniform spatial hash for fast nearest-neighbour conflict checks."""

    def __init__(self, width: int, height: int, cell_size: float):
        self.cell_size = max(cell_size, 0.5)
        self.width = width
        self.height = height
        self.grid: dict[tuple[int, int], list[int]] = {}
        self.circles: list[tuple[float, float, float]] = []
        self.max_r: float = 0.0   # largest radius ever inserted

    def _cell(self, x: float, y: float):
        return (int(x / self.cell_size), int(y / self.cell_size))

    def insert(self, x: float, y: float, r: float):
        self.circles.append((x, y, r))
        if r > self.max_r:
            self.max_r = r
        c = self._cell(x, y)
        self.grid.setdefault(c, []).append(len(self.circles) - 1)

    def has_conflict(self, x: float, y: float, r: float, gap: float) -> bool:
        cx, cy = self._cell(x, y)
        # Search must reach any existing circle whose edge could touch ours.
        # Worst case: existing circle has radius max_r, so center distance
        # for conflict = r + max_r + gap.
        sr = math.ceil((r + self.max_r + gap) / self.cell_size) + 1
        for dx in range(-sr, sr + 1):
            for dy in range(-sr, sr + 1):
                for idx in self.grid.get((cx + dx, cy + dy), []):
                    ex, ey, er = self.circles[idx]
                    if (x - ex) ** 2 + (y - ey) ** 2 < (r + er + gap) ** 2:
                        return True
        return False

    def rebuild(self, circles: list[tuple[float, float, float]]):
        """Rebuild grid from a new circle list (used after Lloyd step)."""
        self.grid = {}
        self.circles = list(circles)
        self.max_r = 0.0
        for i, (x, y, r) in enumerate(self.circles):
            if r > self.max_r:
                self.max_r = r
            c = self._cell(x, y)
            self.grid.setdefault(c, []).append(i)


def poisson_gap_fill(
    existing: list[tuple[float, float, float]],
    radius_map: np.ndarray,
    gap: float,
    candidates_k: int = 30,
    progress_callback=None,
    preview_callback=None,
    preview_every: int = 1000,
) -> list[tuple[float, float, float]]:
    """
    Fill coverage gaps left by SIFT seeds using Poisson disk sampling.
    Multi-seed: restarts from uncovered areas so every region is reached.

    Returns only the NEW circles (caller appends to existing).
    """
    H, W = radius_map.shape
    r_min = float(radius_map.min())
    cell_size = max(r_min / 1.42, 0.5)

    # Negative gap means airbrush overlap — clamp to 0 for placement purposes.
    # Overlapping is handled by the optimizer and SVG renderer, not here.
    placement_gap = max(gap, 0.0)

    grid = SpatialGrid(W, H, cell_size)
    for (x, y, r) in existing:
        grid.insert(x, y, r)

    rng = np.random.default_rng(1)
    placed: list[tuple[float, float, float]] = []

    # Coarse coverage tracker for restarts
    cover_cell = max(1, int(r_min * 1.8))
    cover_rows = H // cover_cell + 1
    cover_cols = W // cover_cell + 1
    covered = np.zeros((cover_rows, cover_cols), dtype=bool)

    # Mark existing circles as covered
    for (x, y, r) in existing:
        cr, cc = int(y) // cover_cell, int(x) // cover_cell
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r2, c2 = cr + dr, cc + dc
                if 0 <= r2 < cover_rows and 0 <= c2 < cover_cols:
                    covered[r2, c2] = True

    def mark(x, y):
        cr, cc = int(y) // cover_cell, int(x) // cover_cell
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r2, c2 = cr + dr, cc + dc
                if 0 <= r2 < cover_rows and 0 <= c2 < cover_cols:
                    covered[r2, c2] = True

    # Pre-mark the edge band as covered so next_seed never picks those cells
    r_min_map = float(radius_map.min())
    edge_px = r_min_map + placement_gap   # minimum safe inset in pixels
    edge_cover_cells = int(math.ceil(edge_px / cover_cell)) + 1
    covered[:edge_cover_cells, :] = True
    covered[-edge_cover_cells:, :] = True
    covered[:, :edge_cover_cells] = True
    covered[:, -edge_cover_cells:] = True

    # Pre-shuffle all cover-grid positions so restarts are O(1) — no repeated
    # argwhere scans. Near the end of fill, run_from exhausts quickly and seeds
    # are consumed rapidly; scanning the full grid each time was the bottleneck.
    all_cells = np.argwhere(np.ones((cover_rows, cover_cols), dtype=bool))
    rng.shuffle(all_cells)
    seed_iter = iter(all_cells)

    def next_seed():
        for row, col in seed_iter:
            if covered[row, col]:
                continue
            x = float(np.clip((col + rng.uniform(0.1, 0.9)) * cover_cell, edge_px, W - 1 - edge_px))
            y = float(np.clip((row + rng.uniform(0.1, 0.9)) * cover_cell, edge_px, H - 1 - edge_px))
            return x, y
        return None

    def run_from(sx, sy):
        sr = float(radius_map[int(np.clip(sy, 0, H-1)), int(np.clip(sx, 0, W-1))])
        # Reject seed if its circle would cross the canvas edge
        if sx - sr < placement_gap or sx + sr > W - 1 - placement_gap:
            mark(sx, sy)
            return
        if sy - sr < placement_gap or sy + sr > H - 1 - placement_gap:
            mark(sx, sy)
            return
        if grid.has_conflict(sx, sy, sr, placement_gap):
            mark(sx, sy)
            return
        grid.insert(sx, sy, sr)
        placed.append((sx, sy, sr))
        mark(sx, sy)
        active = [(sx, sy, sr)]

        while active:
            i = rng.integers(len(active))
            px, py, pr = active[i]
            found = False
            for _ in range(candidates_k):
                angle = rng.uniform(0, 2 * math.pi)
                min_d = max(pr + placement_gap + 0.5, pr * 0.1 + 0.5)
                dist = rng.uniform(min_d, min_d * 2.2)
                cx = px + dist * math.cos(angle)
                cy = py + dist * math.sin(angle)
                ix, iy = int(cx), int(cy)
                if ix < 0 or ix >= W or iy < 0 or iy >= H:
                    continue
                cr = float(radius_map[iy, ix])
                # Reject if circle edge would cross canvas boundary
                if cx - cr < placement_gap or cx + cr > W - 1 - placement_gap:
                    continue
                if cy - cr < placement_gap or cy + cr > H - 1 - placement_gap:
                    continue
                if grid.has_conflict(cx, cy, cr, placement_gap):
                    continue
                grid.insert(cx, cy, cr)
                placed.append((cx, cy, cr))
                mark(cx, cy)
                active.append((cx, cy, cr))
                found = True
                n = len(placed)
                if progress_callback and n % 300 == 0:
                    progress_callback(n)
                if preview_callback and n % preview_every == 0:
                    preview_callback(list(existing) + placed)
                break
            if not found:
                active.pop(i)

    # Seed from center, then restart from uncovered areas
    run_from(W / 2.0, H / 2.0)
    while True:
        s = next_seed()
        if s is None:
            break
        run_from(s[0], s[1])

    return placed
