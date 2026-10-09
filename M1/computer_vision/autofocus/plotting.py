"""Shared plots for the desktop UI and portable reports (English axes)."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from .analysis import AnalysisResult, method_name, normalize, rank_correlation, CONSENSUS, top_indices

COLORS = ["#13a98c", "#487fe8", "#d088ef", "#ebad3d", "#e56d83", "#69a4a8", "#967350", "#a5b749", "#7e89bc", "#e28657", "#34465e"]


def style(ax, title="", xlabel="Working distance (mm)", ylabel="Focus score"):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=14, color="#22344b")
    ax.set_xlabel(xlabel, fontsize=9, color="#60718a")
    ax.set_ylabel(ylabel, fontsize=9, color="#60718a")
    ax.tick_params(labelsize=8, colors="#60718a")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#dce4ed")
    ax.grid(alpha=0.35, color="#dce4ed")
    ax.set_axisbelow(True)
    if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def curves_figure(result: AnalysisResult, selected: str | None = None) -> Figure:
    fig = Figure(figsize=(11, 6), layout="constrained", facecolor="white")
    ax = fig.add_subplot(111)
    for i, (key, values) in enumerate(result.scores.items()):
        lw = 2.5 if key == selected or key == CONSENSUS else 1.4
        alpha = 1 if selected is None or key == selected else 0.4
        ax.plot(result.distances, normalize(values), label=method_name(key), color=COLORS[i % len(COLORS)], lw=lw, alpha=alpha)
        peak = result.methods[key].search.index
        ax.scatter(result.distances[peak], normalize(values)[peak], s=30, color=COLORS[i % len(COLORS)], zorder=4)
    if selected and result.options.search != "global":
        visits = sorted(set(result.methods[selected].search.visited))
        ax.scatter(result.distances[visits], normalize(result.scores[selected])[visits], facecolors="none", edgecolors="#273a52", s=44, linewidths=1, label="Gradient search visits", zorder=5)
    style(ax, "Focus curves · scaled within each method", ylabel="Relative score (min–max 0–1)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=8)
    return fig


def fine_figure(result: AnalysisResult, key: str) -> Figure:
    fig = Figure(figsize=(10, 5), layout="constrained", facecolor="white")
    ax = fig.add_subplot(111)
    outcome = result.methods[key]
    x, y = result.distances, result.scores[key]
    coarse = outcome.search.index
    ax.plot(x, y, color="#b0bdce", lw=1, label="Measured focus")
    indices = outcome.fine.fit_indices
    ax.scatter(x[indices], y[indices], color="#13a98c", s=42, label="Fit samples", zorder=4)
    ax.scatter(x[coarse], y[coarse], marker="*", color="#e9a93b", s=140, label="Coarse maximum", zorder=5)
    if outcome.fine.coefficients is not None:
        xx = np.linspace(x[indices[0]], x[indices[-1]], 160)
        ax.plot(xx, outcome.fine.evaluate(xx), color="#487fe8", lw=2, label="Local quadratic fit")
    if outcome.fine.distance_mm is not None:
        ax.axvline(outcome.fine.distance_mm, ls="--", color="#487fe8", label=f"Fine estimate: {outcome.fine.distance_mm:.6f} mm")
    radius = max(6, result.options.fit_window)
    ax.set_xlim(x[max(0, coarse - radius)], x[min(len(x) - 1, coarse + radius)] if len(x) > 1 else x[0] + 0.01)
    local = y[max(0, coarse - radius): min(len(x), coarse + radius + 1)]
    span = max(float(np.ptp(local)), abs(float(local.mean())) * 0.01, 1e-3)
    ax.set_ylim(local.min() - span * 0.15, local.max() + span * 0.25)
    style(ax, f"Quadratic refinement · {method_name(key)}")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    return fig


def method_figure(result: AnalysisResult, key: str) -> Figure:
    """Selected algorithm only: full recorded stack above its local quadratic fit."""
    fig = Figure(figsize=(7, 7), layout="constrained", facecolor="white")
    ax = fig.add_subplot(211)
    x, y = result.distances, result.scores[key]
    ax.plot(x, y, color="#139c83", lw=1.7, label="Recorded focus")
    for rank, index in enumerate(top_indices(y)):
        color = ["#e5a63a", "#477fea", "#a270cf"][rank]
        ax.scatter(x[index], y[index], color=color, s=45, zorder=5,
                   label=f"Rank {rank+1}: image #{result.entries[index].sequence+1}")
    outcome = result.methods[key]
    if result.options.search != "global":
        visits = sorted(set(outcome.search.visited))
        ax.scatter(x[visits], y[visits], facecolors="none", edgecolors="#57687e", s=27, lw=.8, zorder=4, label="Gradient visits")
    style(ax, "Measured focus · selected algorithm", ylabel="Raw focus score")
    ax.legend(frameon=False, fontsize=7, loc="best", ncols=2)
    zoom = fig.add_subplot(212)
    i = outcome.search.index
    chosen = outcome.fine.fit_indices
    radius = max(6, result.options.fit_window)
    lo, hi = max(0,i-radius), min(len(x),i+radius+1)
    zoom.plot(x[lo:hi],y[lo:hi],color="#b7c3d1", lw=1, label="Measured")
    zoom.scatter(x[chosen],y[chosen],color="#139c83",s=36,zorder=4,label="Fit samples")
    zoom.scatter(x[i],y[i],marker="*",s=120,color="#e5a63a",zorder=5,label="Coarse maximum")
    if outcome.fine.coefficients is not None:
        xx = np.linspace(x[chosen[0]],x[chosen[-1]],150)
        zoom.plot(xx,outcome.fine.evaluate(xx),color="#477fea",lw=2,label="Quadratic fit")
    if outcome.fine.distance_mm is not None:
        zoom.axvline(outcome.fine.distance_mm,color="#477fea",ls="--",lw=1.3,label=f"Fine: {outcome.fine.distance_mm:.6f} mm")
    r2 = f"R² = {outcome.fine.r_squared:.4f}" if outcome.fine.r_squared is not None else "fit unavailable"
    style(zoom,f"Local quadratic refinement · {r2}",ylabel="Raw focus score")
    zoom.legend(frameon=False,fontsize=7,loc="best",ncols=2)
    return fig


def statistics_figure(result: AnalysisResult) -> Figure:
    fig = Figure(figsize=(12, 6), layout="constrained", facecolor="white")
    box = fig.add_subplot(121)
    keys = list(result.scores)
    values = [normalize(result.scores[k]) for k in keys]
    artists = box.boxplot(values, orientation="horizontal", patch_artist=True, tick_labels=[method_name(k) for k in keys], showfliers=False)
    for i, patch in enumerate(artists["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.65)
    style(box, "Score distributions", xlabel="Relative score (0–1)", ylabel="")
    corr = fig.add_subplot(122)
    matrix = rank_correlation(result.scores)
    im = corr.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1)
    labels = ["Consensus" if k == CONSENSUS else method_name(k) for k in keys]
    corr.set_xticks(range(len(keys)), labels, rotation=55, ha="right", fontsize=7)
    corr.set_yticks(range(len(keys)), labels, fontsize=7)
    corr.set_title("Spearman rank correlation", loc="left", fontsize=11, fontweight="bold", pad=14, color="#22344b")
    if len(keys) <= 12:
        for i in range(len(keys)):
            for j in range(len(keys)):
                value = matrix[i, j]
                caption = f"{value:.2f}".replace("0.", ".") if np.isfinite(value) else "—"
                corr.text(j, i, caption, ha="center", va="center", fontsize=7 if len(keys) <= 7 else 6, color="white" if abs(value) > 0.6 else "#22344b")
    fig.colorbar(im, ax=corr, fraction=0.05, pad=0.04)
    return fig


def batch_figure(results: list[AnalysisResult]) -> Figure:
    fig = Figure(figsize=(11, 6), layout="constrained", facecolor="white")
    ax = fig.add_subplot(111)
    keys = list(dict.fromkeys(key for result in results for key in result.scores))
    for j, key in enumerate(keys):
        points = [(i + (j - (len(keys) - 1) / 2) * 0.055, r.entries[r.methods[key].search.index].distance_mm) for i, r in enumerate(results) if key in r.methods]
        if points:
            xx, yy = zip(*points)
            ax.scatter(xx, yy, color=COLORS[j % len(COLORS)], s=45, label=method_name(key))
    ax.set_xticks(range(len(results)), [r.dataset.name for r in results])
    style(ax, "Coarse working distance across scanning speeds", xlabel="Imageset", ylabel="Working distance (mm)")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1, 1), loc="upper left")
    return fig
