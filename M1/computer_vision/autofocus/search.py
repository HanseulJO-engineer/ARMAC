"""Centered-difference gradient ascent and guarded local quadratic refinement."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    index: int
    visited: list[int]
    local_index: int
    corrected: bool
    message: str


@dataclass
class FineResult:
    distance_mm: float | None
    coefficients: list[float] | None
    center_mm: float
    scale_mm: float
    fit_indices: list[int]
    r_squared: float | None
    rmse: float | None
    status: str

    def evaluate(self, x):
        if self.coefficients is None:
            return np.full_like(np.asarray(x, dtype=float), np.nan)
        return np.polyval(self.coefficients, (np.asarray(x) - self.center_mm) / self.scale_mm)


def find_coarse_sharpest(distances: np.ndarray, scores: np.ndarray, strategy="gradient_verified", start_fraction=0.25) -> SearchResult:
    x, y = np.asarray(distances, dtype=float), np.asarray(scores, dtype=float)
    if len(x) != len(y) or not len(y) or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Focus scores and positions must be finite and have matching lengths.")
    if len(x) > 1 and not np.all(np.diff(x) > 0):
        raise ValueError("Positions must increase strictly.")
    if strategy not in {"gradient_verified", "global", "multistart"}:
        raise ValueError(f"Unknown search strategy: {strategy}")
    global_index = int(np.argmax(y))
    if strategy == "global" or len(y) < 3:
        return SearchResult(global_index, list(range(len(y))), global_index, False, "Selected the maximum over the full recorded stack.")
    yrange = float(np.ptp(y))
    if yrange <= 1e-12:
        return SearchResult(0, [0], 0, False, "The focus curve is flat. The first image is returned as a tie, not an identifiable optimum.")
    yn = (y - y.min()) / yrange

    def climb(start):
        i = int(start)
        visited = [i]
        alpha = max(1.0, (len(y) - 1) / 8)
        for _ in range(4 * len(y)):
            left, right = max(0, i - 1), min(len(y) - 1, i + 1)
            gradient = (yn[right] - yn[left]) / (x[right] - x[left])
            direction = int(np.sign(gradient))
            if direction == 0:
                break
            # Normalized centered gradient, decreasing step, improvement-only updates.
            candidate = int(np.clip(i + direction * max(1, round(alpha)), 0, len(y) - 1))
            if candidate != i and yn[candidate] > yn[i] + 1e-12:
                i = candidate
                visited.append(i)
            elif alpha > 1:
                alpha = max(1.0, alpha * 0.5)
            else:
                neighbors = [j for j in (left, right) if j != i]
                best = max(neighbors, key=lambda j: yn[j])
                if yn[best] <= yn[i] + 1e-12:
                    break
                i = best
                visited.append(i)
        return i, visited

    if not 0 <= start_fraction <= 1:
        raise ValueError("The initial position fraction must be between 0 and 1.")
    starts = [round(start_fraction * (len(y) - 1))]
    if strategy == "multistart":
        starts = sorted(set(starts + [round(t * (len(y) - 1)) for t in np.linspace(0, 1, 7)]))
    attempts = [climb(start) for start in starts]
    local_index, _ = max(attempts, key=lambda p: y[p[0]])
    visited = [j for _, path in attempts for j in path]
    corrected = bool(y[local_index] < y[global_index] - max(1e-12, yrange * 1e-10))
    message = "A local gradient maximum was corrected using the larger full-stack maximum." if corrected else "Centered-difference gradient ascent was verified against the full stack."
    return SearchResult(global_index, visited, local_index, corrected, message)


def estimate_fine_position(distances: np.ndarray, scores: np.ndarray, coarse: int, window=5) -> FineResult:
    x, y = np.asarray(distances, dtype=float), np.asarray(scores, dtype=float)
    if window < 3 or window % 2 != 1:
        raise ValueError("The quadratic fit window must be an odd number of at least 3.")
    radius = window // 2
    indices = list(range(max(0, coarse - radius), min(len(x), coarse + radius + 1)))
    center = float(x[coarse])
    scale = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    base = dict(coefficients=None, center_mm=center, scale_mm=scale, fit_indices=indices, r_squared=None, rmse=None)
    if len(indices) < 3:
        return FineResult(None, status="Fine position unavailable: fewer than three samples.", **base)
    if coarse in (0, len(x) - 1):
        return FineResult(None, status="Fine position withheld: the maximum is at a scan boundary with no samples on both sides.", **base)
    xx, yy = x[indices], y[indices]
    if np.ptp(yy) <= max(1e-12, abs(float(np.mean(yy))) * 1e-10):
        return FineResult(None, status="Fine position unavailable: the local curve is flat.", **base)
    coeff = np.polyfit((xx - center) / scale, yy, 2)
    residual = yy - np.polyval(coeff, (xx - center) / scale)
    sst = float(np.sum((yy - yy.mean()) ** 2))
    base.update(coefficients=coeff.tolist(), r_squared=1 - float(np.sum(residual ** 2)) / sst, rmse=float(np.sqrt(np.mean(residual ** 2))))
    if coeff[0] >= -max(1e-12, np.ptp(yy) * 1e-10):
        return FineResult(None, status="Fine position withheld: the fitted quadratic is not concave down.", **base)
    peak = float(center - scale * coeff[1] / (2 * coeff[0]))
    # Interpolation is valid only inside the immediate neighbors of the measured maximum.
    if not x[coarse - 1] <= peak <= x[coarse + 1]:
        return FineResult(None, status="Fine position withheld: the vertex lies outside the immediate neighbors. Try a smaller fit window.", **base)
    return FineResult(peak, status="Fine position estimated from the quadratic vertex. No image was acquired at this interpolated position.", **base)
