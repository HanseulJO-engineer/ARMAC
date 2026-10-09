"""UI-independent pipeline with cancellation and reproducible settings."""
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Callable
import cv2
import numpy as np
from .dataset import Dataset, ImageEntry, grayscale
from .metrics import calculate_focus, BY_KEY, DEFAULT_METHODS
from .search import find_coarse_sharpest, estimate_fine_position, SearchResult, FineResult

CONSENSUS = "rank_consensus"


def method_name(key: str) -> str:
    return "Consensus · mean rank" if key == CONSENSUS else BY_KEY[key].name


@dataclass(frozen=True)
class AnalysisOptions:
    methods: tuple[str, ...] = DEFAULT_METHODS
    search: str = "gradient_verified"
    start_fraction: float = 0.25
    fit_window: int = 5
    gaussian_sigma: float = 0.0
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


@dataclass
class MethodResult:
    search: SearchResult
    fine: FineResult
    statistics: dict[str, float]


@dataclass
class AnalysisResult:
    dataset: Dataset
    options: AnalysisOptions
    entries: list[ImageEntry]
    scores: dict[str, np.ndarray]
    methods: dict[str, MethodResult]
    warnings: list[str]
    elapsed_seconds: float
    image_shape: tuple[int, int]

    @property
    def distances(self):
        return np.asarray([e.distance_mm for e in self.entries])

    def summary(self):
        return {
            "dataset": self.dataset.name, "source": str(self.dataset.source),
            "position_source": self.dataset.position_source,
            "settings": asdict(self.options), "images_found": len(self.dataset.images),
            "images_analyzed": len(self.entries), "image_shape": self.image_shape,
            "elapsed_seconds": self.elapsed_seconds, "warnings": self.warnings,
            "methods": {key: {"name": method_name(key), "image": self.dataset.display_path(self.entries[r.search.index]),
                              "sequence_1_based": self.entries[r.search.index].sequence + 1,
                              "coarse_distance_mm": self.entries[r.search.index].distance_mm,
                              "coarse_focus": float(self.scores[key][r.search.index]),
                              "top_three": [{"rank": n+1, "filename": self.entries[i].name,
                                             "sequence_1_based": self.entries[i].sequence+1,
                                             "distance_mm": self.entries[i].distance_mm,
                                             "score": float(self.scores[key][i])} for n,i in enumerate(top_indices(self.scores[key]))],
                              "search": asdict(r.search), "fine": asdict(r.fine),
                              "statistics": r.statistics} for key, r in self.methods.items()}}


class Cancelled(Exception):
    pass


def top_indices(values, count=3) -> list[int]:
    """Stable descending rank: tied scores retain original scan order."""
    return np.argsort(-np.asarray(values), kind="stable")[:count].tolist()


def prepare_image(image: np.ndarray, options: AnalysisOptions) -> np.ndarray:
    a = grayscale(image)
    x, y, w, h = options.roi
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x + w <= 1.000001 and y + h <= 1.000001):
        raise ValueError("ROI must have a positive size and stay inside the image.")
    height, width = a.shape
    x0, y0 = round(x * width), round(y * height)
    x1, y1 = min(width, round((x + w) * width)), min(height, round((y + h) * height))
    a = a[y0:y1, x0:x1]
    if min(a.shape, default=0) < 3:
        raise ValueError("ROI is too small. Select at least 3×3 pixels.")
    if options.gaussian_sigma < 0 or not np.isfinite(options.gaussian_sigma):
        raise ValueError("Gaussian sigma must be finite and nonnegative.")
    if options.gaussian_sigma:
        a = cv2.GaussianBlur(a, (0, 0), options.gaussian_sigma)
    return a


def ranks(values: np.ndarray) -> np.ndarray:
    """Zero-based average ranks, including ties; ascending focus quality."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def rank_correlation(scores: dict[str, np.ndarray]) -> np.ndarray:
    vectors = np.asarray([ranks(s) for s in scores.values()])
    vectors -= vectors.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(vectors, axis=1)
    denom = norm[:, None] * norm[None, :]
    return np.divide(vectors @ vectors.T, denom, out=np.full_like(denom, np.nan), where=denom > 0)


def normalize(values):
    a = np.asarray(values, dtype=float)
    span = float(np.ptp(a))
    return (a - a.min()) / span if span > 1e-12 else np.zeros_like(a)


def analyze_dataset(dataset: Dataset, options: AnalysisOptions,
                    progress: Callable[[int, int, str], None] | None = None,
                    cancelled: Callable[[], bool] | None = None) -> AnalysisResult:
    if not options.methods or any(k not in BY_KEY for k in options.methods):
        raise ValueError("Select at least one focus measure.")
    if len(set(options.methods)) != len(options.methods):
        raise ValueError("Duplicate focus measures were selected.")
    # Validate fractions without making assumptions about actual image dimensions.
    x, y, w, h = options.roi
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x + w <= 1.000001 and y + h <= 1.000001):
        raise ValueError("ROI must have a positive size and stay inside the image.")
    if options.gaussian_sigma < 0 or not np.isfinite(options.gaussian_sigma):
        raise ValueError("Gaussian sigma must be finite and nonnegative.")
    if options.fit_window < 3 or options.fit_window % 2 == 0:
        raise ValueError("The fit window must be an odd number of at least 3.")
    start = perf_counter()
    warnings = list(dataset.warnings)
    columns = {k: [] for k in options.methods}
    timings = {}
    entries, shape, expected_dtype = [], None, None
    with dataset.reader() as read:
        for i, entry in enumerate(dataset.images):
            if cancelled and cancelled():
                raise Cancelled()
            try:
                original = read(entry)
                a = prepare_image(original, options)
                if shape is None:
                    shape = grayscale(original).shape
                    expected_dtype = original.dtype
                elif grayscale(original).shape != shape or original.dtype != expected_dtype:
                    raise ValueError("Image size or pixel type differs from the first valid image.")
                values = calculate_focus(a, options.methods, timings)
                entries.append(entry)
                for key in options.methods:
                    columns[key].append(values[key])
            except (ValueError, OSError, cv2.error, KeyError) as e:
                warnings.append(f"Skipped: {entry.name} · {e}")
            if progress:
                progress(i + 1, len(dataset.images), entry.name)
    if not entries:
        raise ValueError("No images could be analyzed. Check the input files and ROI.")
    scores = {k: np.asarray(v, dtype=float) for k, v in columns.items()}
    distances = np.asarray([e.distance_mm for e in entries])
    outcomes = {}
    for key, values in scores.items():
        search = find_coarse_sharpest(distances, values, options.search, options.start_fraction)
        fine = estimate_fine_position(distances, values, search.index, options.fit_window)
        if np.ptp(values) <= max(1e-12, abs(float(values.mean())) * 1e-10):
            warnings.append(f"{method_name(key)}: Flat focus curve; the optimal position cannot be identified.")
        sorted_scores = np.sort(values)
        gap = float((sorted_scores[-1] - sorted_scores[-2]) / max(abs(sorted_scores[-1]), 1e-12) * 100) if len(values) > 1 else 0.0
        stats = {"mean": float(values.mean()), "median": float(np.median(values)),
                 "std_population": float(values.std()), "min": float(values.min()),
                 "max": float(values.max()), "peak_gap_percent": gap}
        if key in timings:
            stats["median_scoring_ms"] = float(np.median(timings[key]))
        outcomes[key] = MethodResult(search, fine, stats)
    return AnalysisResult(dataset, options, entries, scores, outcomes, warnings, perf_counter() - start, shape)
