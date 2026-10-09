# FocusLab Desktop — CV1A Lab 5.6

A native Python/OpenCV/PySide6 desktop application for the **provided pollen stacks**, following `2026-2027_CV1A-lab`, section 5.6, pages 29–30, and the lecture's Chapter 6.

## Run

Double-click **FocusLab.app** on this Mac. Keep it in this project: the launcher uses the installed `.venv` here. `Launch FocusLab.command` is an alternative launcher.

```sh
cd /Users/kobe/workspace/ARMAC/M1/computer_vision
.venv/bin/python run_app.py
```

To analyze the provided stacks immediately after opening:

```sh
.venv/bin/python run_app.py --auto-batch
```

On another computer, install Python 3.10 or newer and create an environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_app.py
```

## Follow the lab procedure

1. Load **Documents for autofocusing app**. The four supplied scanning-speed stacks are discovered individually. Full directory and original image paths are preserved. ZIP stacks can also be read directly.
2. Select focus measures; clicking a method previews its equation before analysis. Click **Analyze selected stack** or **Analyze all four stacks**. A visible animated analysis message, percentage progress bar, current file/count, and Cancel button appear immediately.
3. In **Selected algorithm**, choose a method. The **left side** shows its coarse sharpest image and **rank 1, 2 and 3** thumbnails. Each includes the original filename, original 1-based scan image number, WD and focus score. Click a ranked image to inspect it. Rank is distinct from original scan order.
4. The **right side** shows the method's rendered mathematical equation, implementation parameters, source, full focus curve and local quadratic fit. The header reports the coarse WD and estimated fine WD.
5. In **Compare algorithms**, compare every selected method separately from the single-method screen. Switch between overlaid focus curves and distributions/Spearman rank correlations. Order the table by peak separation, median scoring time or local quadratic fit R². Click a row to inspect that method's top 3.
6. In **Scan-speed comparison**, inspect results across the provided image sets. **Files** lists original filenames, sequence numbers, WD, scores and full paths. **Methods & sources** explains the course procedure and all operators.
7. **Export results** saves a self-contained English HTML report, CSV/JSON data, each algorithm's top-three original images and its own focus/refinement figure.

The required coarse-to-fine flow remains central: focus computation → centered-difference gradient ascent with full-stack verification → display the acquired coarse image → local quadratic fine-position estimation. Extra features visualize and compare this same procedure; no manual reference labels, separate synthetic benchmark or microscope control is included.

## Which algorithm is best?

Compare the **actual ranked images** and each curve's shape first. The comparison table measures three different trade-offs:

- **Peak gap %**: difference between the highest and second-highest measured scores, divided by the absolute highest score. Larger separation can help discrimination but may also reflect noise.
- **Median ms/image**: measured operator computation time on this machine, excluding loading and shared preprocessing. Intermediate results are not reused between algorithms when timing them.
- **Fit R²**: goodness of the local quadratic fit. It is not a probability or evidence of true physical focus accuracy. With three samples a quadratic can fit exactly, so compare methods with the same fit window.

Raw focus values have different scales and cannot select a winner across formulas. The recorded stacks provide no independent ground-truth winner. The app therefore shows the leading method **for the chosen descriptive criterion**, without inventing an overall accuracy ranking.

## Focus measures and research additions

| Operator | Mathematical technique | Source |
|---|---|---|
| Normalized variance | Brightness variance / mean | Chapter 6, Eq. 6.5; lab baseline |
| Squared gradient | Horizontal first finite difference squared | Eq. 6.2 |
| Brenner · course | Absolute two-pixel horizontal difference | Eq. 6.3 |
| Brenner · squared | Squared two-pixel difference | Additional variant |
| Laplacian variance | Variance of a second derivative | Eq. 6.4 adapted to avoid signed cancellation |
| Tenengrad · Sobel | Two-direction gradient energy | Pertuz et al., GRA6 |
| Shannon entropy | Histogram information in bits | Eq. 6.6; standard negative sign |
| Autocorrelation | Difference of correlations at lags 1 and 2 | Eq. 6.7 |
| Haar wavelet · L1 | Absolute detail coefficients | Eq. 6.8 |
| Haar wavelet · variance | Variance of absolute detail coefficients | Eq. 6.9 |
| **Modified Laplacian · SML** | Absolute directional second differences | Pertuz et al., LAP2, Eqs. A.24–A.25 |
| **Gaussian derivative energy** | Squared first Gaussian derivatives | Pertuz et al., GRA1, Eqs. A.15–A.16 |
| **Tenengrad variance** | Variance of Sobel gradient magnitude | Pertuz et al., GRA7, Eq. A.22 |

Research source: [Pertuz, Puig & Garcia (2013), *Analysis of focus measure operators for shape-from-focus*, Pattern Recognition 46, 1415–1432](https://doi.org/10.1016/j.patcog.2012.11.011). The three research additions are discrete global adaptations tested on the supplied SEM stacks, not a reproduction of the paper's depth-reconstruction experiments. Kernel sizes, normalization, borders and the Gaussian derivative's internal σ=1 are explicit in the app.

All scoring uses floating-point grayscale intensities with a fixed 0–255 range. Course sums are divided by valid pixel or coefficient counts. Full ROI and no added Gaussian denoising are defaults. Optional ROI/denoising apply equally to selected methods on the next analysis. Edge overlay affects display only.

## Recorded WD ambiguity

Each provided stack contains **199 images** and **200 WD values**. Recorded distances begin at **10.26 mm with 5 µm spacing**, while the lab text says 10 mm and 10 µm. The app uses the recorded metadata and reports this discrepancy.

Default alignment is the first naturally sorted image ↔ **WD[0]**. **WD[1]** is available because correspondence is not unambiguously stated in the data. It shifts absolute distances by 5 µm without changing image rankings. Manual start/step fields are only a fallback if metadata is disabled or missing.

Fine WD is an **interpolated position**, not another acquired image. Boundary peaks, flat curves, non-concave fits and vertices outside the immediate neighboring interval are withheld with an explanation. Tied image scores keep natural scan order. Failed/mismatched images are logged without shifting the remaining original distances.

## Saved results

Each export creates a new timestamped folder:

- `report.html`: English report with embedded figures and ranked previews.
- `comparison.csv`: per-stack/per-method coarse/fine results, timing and peak gaps.
- `top_three.csv`: rank, original filename, scan image number, path, WD and score.
- Per-stack `scores.csv` and `summary.json`: all raw scores, original paths, settings, top-three identities, search trace, fit coefficients/status and warnings.
- `best_METHOD.png`, `rank2_METHOD.png`, `rank3_METHOD.png`: original full-resolution images.
- `focus_METHOD.png`: that method's full focus curve and local quadratic fit.
- `focus_curves.png`, `statistics.png`, `scanning_speed_comparison.png`: overall comparison figures.

## Reproduce the supplied-stack analysis

```sh
.venv/bin/python run_app.py --batch --all-methods --output output/english
.venv/bin/python -m unittest discover -s tests -v
```

`REPORT.md` documents the lab requirements and measured results. Functional tests cover the focus operators, gradient local-maximum correction, quadratic estimation, original WD preservation, stable top-three ranking, exports, visible progress, English equations and desktop navigation.

Core files: `autofocus/metrics.py`, `dataset.py`, `search.py`, `analysis.py`, `plotting.py`, `gui.py`, `export.py`.
