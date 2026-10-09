# FocusLab — Lab 5.6 implementation report

## Scope and required outputs

This application follows `2026-2027_CV1A-lab`, section 5.6, pages 29–30. The input is the path of a recorded SEM image stack. Its outputs are the acquired coarse sharpest image, focus value, image number and working distance, followed by the fine working-distance estimate from a local quadratic fit. The lecture reference is `2026-2027_IMG_lectures`, Chapter 6.

| Lab requirement | Implementation |
|---|---|
| Focus function | `calculate_focus` in `metrics.py`; normalized variance baseline |
| Derivative-based coarse search | `find_coarse_sharpest` in `search.py`; centered differences, normalized direction and decreasing step |
| Avoid local maxima | Verify the result against the complete recorded stack and correct a lower local maximum |
| Display coarse image, focus and position | Left-hand image viewer and metadata; original path preserved |
| Quadratic fine position | `estimate_fine_position`; centered/scaled local least squares and a guarded concave vertex |
| Report and code | This report, source files, and exported HTML/CSV/JSON |

The application remains a tool for the supplied pollen stacks. Extra functionality is limited to English explanations/equations, top-three acquired images, individual graphs, a separate algorithm comparison, visible progress, and three research-derived operators. No user-labeled reference focus, synthetic benchmark or hardware control is included.

## Data and distance mapping

All four supplied scanning-speed sets contain 199 grayscale 1024×768 images. Each WD file contains 200 values beginning at 10.26 mm with 5 µm spacing. These differ from the lab text (10 mm, 10 µm). Metadata is used, and the first naturally sorted image is mapped to WD[0] in these results. WD[1] is selectable because the original correspondence is ambiguous; it shifts positions by 5 µm without changing image rankings.

## Course algorithm and numerical conventions

Normalized variance is Var(I)/mean(I), matching Eq. 6.5. All scores are calculated in floating point at a fixed grayscale scale. The default ROI is the full image and preprocessing sigma is zero. Course sums are divided by their valid support sizes; identical image sizes preserve their ranking. Entropy uses the conventional negative sign; Laplacian variance avoids signed cancellation.

Coarse search uses a centered-difference gradient on normalized measured focus scores. It follows an increasing direction, halves an unsuccessful step, and stops at a local maximum. All images are measured for method comparison, so this local result can be checked against the full stack. No acquisition-speed reduction is claimed. Fine position is a least-squares quadratic vertex from five local samples, with WD centered and scaled for numerical stability. Boundary, flat, non-concave or out-of-neighbor fits are withheld. Fine WD is an interpolation, not an acquired image.

## Actual supplied-image results

Settings: all 13 methods, verified gradient ascent, initial position 25%, five-point fit, full ROI, no added denoising, WD offset zero. All 796 images were successfully analyzed. The following table contrasts the course baseline with the three additional research methods.

| Stack | Method | Rank-1 file / scan image # | Coarse mm | Fine mm | Local R² | Median scoring ms/image |
|---|---|---|---:|---:|---:|---:|
| pollen.spd1 | Normalized variance | pollen.spd1.0100.png / #100 | 10.755000 | 10.752305 | 0.9481 | 0.747 |
| pollen.spd1 | Modified Laplacian · SML | pollen.spd1.0097.png / #97 | 10.740000 | Withheld | 0.7474 | 2.036 |
| pollen.spd1 | Gaussian derivative energy | pollen.spd1.0100.png / #100 | 10.755000 | 10.757064 | 0.9056 | 3.590 |
| pollen.spd1 | Tenengrad variance | pollen.spd1.0100.png / #100 | 10.755000 | 10.754397 | 0.9064 | 2.876 |
| pollen.spd2 | Normalized variance | pollen.spd2.0099.png / #99 | 10.750000 | 10.745985 | 0.9217 | 0.723 |
| pollen.spd2 | Modified Laplacian · SML | pollen.spd2.0097.png / #97 | 10.740000 | 10.742099 | 0.9585 | 1.974 |
| pollen.spd2 | Gaussian derivative energy | pollen.spd2.0099.png / #99 | 10.750000 | 10.752808 | 0.9925 | 3.494 |
| pollen.spd2 | Tenengrad variance | pollen.spd2.0099.png / #99 | 10.750000 | 10.750069 | 0.8383 | 2.781 |
| pollen.spd3 | Normalized variance | pollen.spd3.0098.png / #98 | 10.745000 | 10.744482 | 0.9437 | 0.743 |
| pollen.spd3 | Modified Laplacian · SML | pollen.spd3.0098.png / #98 | 10.745000 | 10.742119 | 0.9945 | 2.033 |
| pollen.spd3 | Gaussian derivative energy | pollen.spd3.0099.png / #99 | 10.750000 | 10.750923 | 0.9497 | 3.577 |
| pollen.spd3 | Tenengrad variance | pollen.spd3.0098.png / #98 | 10.745000 | 10.747292 | 0.9683 | 2.866 |
| pollen.spd5 | Normalized variance | pollen.spd5.0098.png / #98 | 10.745000 | 10.744543 | 0.9855 | 0.739 |
| pollen.spd5 | Modified Laplacian · SML | pollen.spd5.0097.png / #97 | 10.740000 | 10.742341 | 0.9983 | 2.024 |
| pollen.spd5 | Gaussian derivative energy | pollen.spd5.0099.png / #99 | 10.750000 | 10.750263 | 0.9990 | 3.583 |
| pollen.spd5 | Tenengrad variance | pollen.spd5.0098.png / #98 | 10.745000 | 10.746170 | 0.9964 | 2.864 |

The application and `top_three.csv` also list ranks 2 and 3 with the exact original filename, 1-based scan number, distance and score. Ties retain original scan order.

## Comparing methods without inventing a winner

The comparison tab orders methods by observed peak separation, median operator time, or local quadratic R². It also overlays normalized curves and displays score distributions and Spearman rank correlations. These indicators describe different trade-offs. They do not supply independent physical focus labels: high separation can come from noise, high R² indicates model fit, and fast execution alone does not establish accuracy. The analyst can inspect the actual top-three candidates and choose a method for the recorded data without an unsupported universal winner. Timing excludes file decoding and shared preprocessing; each operator computes its own intermediates.

## Research additions

[Pertuz, Puig & Garcia (2013), *Analysis of focus measure operators for shape-from-focus*, Pattern Recognition 46, 1415–1432](https://doi.org/10.1016/j.patcog.2012.11.011) provides the definitions used for Modified Laplacian (LAP2), Gaussian derivative energy (GRA1) and Tenengrad variance (GRA7). These implementations use explicit global mean/variance and discrete kernels for the SEM stacks; they do not replicate a depth-reconstruction experiment. The app shows the actual equation and kernel parameters for every selected method.

## Verification and reproducibility

Functional tests check known intensities, response to blur, constant images, local-maximum correction, an exact quadratic vertex, boundary/flat protection, ZIP equivalence, WD preservation after a failed image, stable top-three ranks and exported identities. Desktop tests cover a visible loading panel/progress, threaded analysis, every selected equation, rank-2 inspection, the separate comparison views, file selection and ROI. Actual-data validation analyzed every supplied image with all 13 methods.

Reproduce the analysis:

```sh
.venv/bin/python run_app.py --batch --all-methods --output output/english
.venv/bin/python -m unittest discover -s tests -v
```

English HTML report: `output/english/autofocus_20261009_091621/report.html`. Full per-method data and original top-three images are stored beside it.
