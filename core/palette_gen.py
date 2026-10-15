"""
Auto-generate a perceptually optimal paint palette from an image.

Uses k-means clustering in CIELAB space so distances are perceptually uniform.
Pixels are weighted by saturation so vivid colors aren't drowned out by large
neutral/background regions — important for portraits where skin tones dominate.
"""

import numpy as np
from .image_processor import load_image


def generate_palette(image_path: str, n_colors: int = 6, max_px: int = 800) -> list[str]:
    """Returns n_colors hex strings — a paint palette anchored on the image's extremes."""
    img = load_image(image_path, max_working_px=max_px)
    lab = img.lab.reshape(-1, 3).astype(np.float32)
    return [_lab_to_hex(c) for c in palette_from_lab(lab, n_colors)]


def palette_from_lab(lab, n_colors: int):
    """Build a palette that REPRESENTS the image's range of colours, not just its dominant
    one.  Plain k-means allocates clusters by area, so a big flat region (e.g. a pink wall)
    hogs most of the palette while rare-but-distinct hues (a lone green, a purple) get one
    slot or none.  Here we (1) anchor the brightest/darkest extremes, (2) flatten the colour
    density so minority hues compete fairly, then (3) de-duplicate near-identical colours and
    refill by farthest-point for maximum spread.  Returns an (n_colors, 3) LAB array."""
    lab = np.asarray(lab, dtype=np.float32)
    rng = np.random.default_rng(0)
    samp = lab[rng.choice(len(lab), 60000, replace=False)] if len(lab) > 60000 else lab
    L = samp[:, 0]
    chroma = np.sqrt(samp[:, 1] ** 2 + samp[:, 2] ** 2)

    # DENSITY-FLATTENING weight: bin LAB coarsely and down-weight over-represented bins, so
    # a huge uniform region can't monopolise the clusters.  ×mild chroma emphasis so vivid
    # colours still register, and a floor so single-pixel noise isn't over-weighted.
    q = np.round(samp / 12.0).astype(np.int64)
    keys = q[:, 0] * 1_000_003 + q[:, 1] * 1009 + q[:, 2]
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    bin_count = counts[inv].astype(np.float32)
    flat_w = 1.0 / np.power(np.maximum(bin_count, 2.0), 0.7)
    weights = flat_w * (1.0 + chroma / 40.0)

    anchors = []
    if n_colors >= 4:
        anchors.append(samp[L >= np.percentile(L, 98.5)].mean(axis=0))
        anchors.append(samp[L <= np.percentile(L, 1.5)].mean(axis=0))
    anchors = (np.array(anchors, dtype=np.float32)
               if anchors else np.zeros((0, 3), np.float32))

    n_rem = n_colors - len(anchors)
    if n_rem > 0:
        centers = _weighted_kmeans(samp, weights, n_rem, n_init=6, max_iter=100)
    else:
        centers = np.zeros((0, 3), np.float32)

    palette = np.vstack([anchors, centers]) if len(anchors) else centers
    palette = _dedup_and_refill(palette, samp, weights, n_colors, min_dE=10.0)
    return boost_chroma(palette, samp)


def _dedup_and_refill(colors, samp, weights, n_colors, min_dE=10.0):
    """Drop colours within min_dE ΔE of one already kept, then refill to n_colors by
    farthest-point selection (pick the well-represented colour furthest from the current
    palette) — maximising hue/tone spread instead of piling up near-duplicates."""
    kept = []
    for c in np.asarray(colors, dtype=np.float32):
        if all(float(np.linalg.norm(c - k)) >= min_dE for k in kept):
            kept.append(c)

    # Only refill from reasonably-present colours (avoid single-pixel noise).
    thresh = np.percentile(weights, 40)
    pool = samp[weights <= thresh] if (weights <= thresh).sum() > n_colors * 5 else samp
    poolw = np.ones(len(pool), np.float32)

    while len(kept) < n_colors and len(pool):
        K = np.array(kept, dtype=np.float32)
        dmin = np.min(((pool[:, None, :] - K[None, :, :]) ** 2).sum(2), axis=1)
        j = int(np.argmax(dmin * poolw))
        if dmin[j] < (min_dE * 0.6) ** 2:      # nothing meaningfully new left
            break
        kept.append(pool[j])
        poolw[j] = 0.0

    kept = np.array(kept, dtype=np.float32)
    # If dedup left us short and refill couldn't help, top up from the original set.
    if len(kept) < n_colors:
        extra = np.asarray(colors, dtype=np.float32)
        kept = np.vstack([kept, extra])[:n_colors] if len(kept) else extra[:n_colors]
    return kept[:n_colors]


def boost_chroma(centers, lab, percentile=72.0, cap=1.4):
    """Push each cluster centre's chroma out toward the vivid edge of its cluster.

    A mosaic can only be as saturated as its palette's most saturated colours (optical
    mixing only ever pulls colours *toward* neutral).  Plain k-means centroids are cluster
    AVERAGES, so they come out muted and the mosaic looks flat.  Here each centre keeps its
    hue + lightness but its chroma is raised to a high percentile of its own cluster — so
    vivid regions yield vivid "primaries" while genuinely muted clusters stay muted (the
    mid-tones needed for dithering)."""
    C = np.asarray(centers, dtype=np.float32).copy()
    n = len(lab)
    rng = np.random.default_rng(0)
    samp = lab[rng.choice(n, size=40000, replace=False)] if n > 40000 else lab
    assign = ((samp[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
    for k in range(len(C)):
        mem = samp[assign == k]
        if len(mem) < 3:
            continue
        ch = np.sqrt(mem[:, 1] ** 2 + mem[:, 2] ** 2)
        target = float(np.percentile(ch, percentile))
        c0 = float(np.hypot(C[k, 1], C[k, 2]))
        if c0 > 1e-3 and target > c0:
            s = min(target / c0, cap)
            C[k, 1] *= s
            C[k, 2] *= s
    return C


def _weighted_kmeans(
    X: np.ndarray,
    weights: np.ndarray,
    k: int,
    n_init: int = 5,
    max_iter: int = 100,
) -> np.ndarray:
    """Weighted k-means in LAB. Returns (k, 3) cluster centers."""
    rng = np.random.default_rng(42)
    best_inertia = float('inf')
    best_centers = None

    for _ in range(n_init):
        centers = _kmeans_plusplus(X, weights, k, rng)
        centers, inertia = _run_kmeans(X, weights, centers, max_iter)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers

    return best_centers


def _kmeans_plusplus(X, weights, k, rng):
    """Weighted k-means++ initialization."""
    n = len(X)
    idx = rng.choice(n, p=weights / weights.sum())
    centers = [X[idx]]

    for _ in range(k - 1):
        # Distance from each point to nearest center (in LAB — perceptually uniform)
        dists = np.array([
            np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        ])[0]
        prob = dists * weights
        prob_sum = prob.sum()
        if prob_sum == 0:
            prob = weights / weights.sum()
        else:
            prob /= prob_sum
        idx = rng.choice(n, p=prob)
        centers.append(X[idx])

    return np.array(centers, dtype=np.float32)


def _run_kmeans(X, weights, centers, max_iter):
    k = len(centers)
    centers = centers.copy()

    for _ in range(max_iter):
        # Assignment: each point → nearest center
        diffs = X[:, None, :] - centers[None, :, :]   # (N, k, 3)
        dists2 = (diffs ** 2).sum(axis=2)              # (N, k)
        labels = dists2.argmin(axis=1)                 # (N,)

        # Update: weighted centroid per cluster
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                w = weights[mask]
                new_centers[j] = (X[mask] * w[:, None]).sum(axis=0) / w.sum()
            else:
                new_centers[j] = centers[j]  # keep if empty

        if np.allclose(new_centers, centers, atol=0.1):
            break
        centers = new_centers

    inertia = float(((X - centers[labels]) ** 2).sum(axis=1).dot(weights))
    return centers, inertia


def _lab_to_hex(lab: np.ndarray) -> str:
    """CIELAB (3,) → #RRGGBB via OpenCV."""
    import cv2
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    lab_u8 = np.array([[[
        np.clip(L * 255.0 / 100.0, 0, 255),
        np.clip(a + 128.0, 0, 255),
        np.clip(b + 128.0, 0, 255),
    ]]], dtype=np.uint8)
    rgb = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)[0, 0]
    return '#{:02X}{:02X}{:02X}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
