"""Course measures and explicitly parameterized operators from the literature."""
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np

PAPER_URL = "https://doi.org/10.1016/j.patcog.2012.11.011"


@dataclass(frozen=True)
class Method:
    key: str
    name: str
    family: str
    formula: str
    explanation: str
    limitation: str
    reference: str
    latex: str
    parameters: str
    source_url: str = ""


METHODS = (
    Method("normalized_variance", "Normalized variance", "Statistics", "Var(I) / mean(I)",
           "Brightness variance divided by mean intensity. The recommended baseline in lab 5.6.",
           "Responds to contrast, illumination and noise. An all-black image receives zero.", "Chapter 6, Eq. 6.5",
           r"F_{NV}=\frac{1}{P\mu}\sum_{(x,y)\in\Omega}(I(x,y)-\mu)^2",
           "P = number of ROI pixels; μ = mean intensity. Fixed intensity range: 0–255."),
    Method("squared_gradient", "Squared gradient", "Derivatives", "mean((I(x+1,y) − I(x,y))²)",
           "First-order horizontal finite differences emphasize abrupt intensity changes.",
           "Directional and noise sensitive. The course sum is divided by the valid pixel count.", "Chapter 6, Eq. 6.2",
           r"F_{SG}=\frac{1}{P_x}\sum_{\Omega_x}[I(x+1,y)-I(x,y)]^2",
           "Horizontal step = 1 pixel; Px counts valid pairs; no threshold."),
    Method("brenner", "Brenner · course", "Derivatives", "mean(|I(x+2,y) − I(x,y)|)",
           "Absolute horizontal differences over a two-pixel separation, following the course convention.",
           "Different from the squared Brenner variant; sensitive to edge orientation.", "Chapter 6, Eq. 6.3",
           r"F_{B}=\frac{1}{P_2}\sum_{\Omega_2}|I(x+2,y)-I(x,y)|",
           "Horizontal step = 2 pixels. Absolute difference, not squared difference."),
    Method("brenner_squared", "Brenner · squared", "Derivatives", "mean((I(x+2,y) − I(x,y))²)",
           "A squared two-pixel difference variant that gives stronger edges greater weight.",
           "Can amplify noise. This is not the absolute-value expression in course Eq. 6.3.", "Squared variant; Pertuz et al., MIS2",
           r"F_{B^2}=\frac{1}{P_2}\sum_{\Omega_2}[I(x+2,y)-I(x,y)]^2",
           "Horizontal step = 2 pixels; mean over valid pairs.", PAPER_URL),
    Method("laplacian_variance", "Laplacian variance", "Derivatives", "Var(∇²I)",
           "Variance of the second derivative highlights fine intensity transitions without signed-sum cancellation.",
           "Highly noise sensitive. The course Laplacian is adapted to a variance measure.", "Chapter 6, Eq. 6.4; variance adaptation",
           r"F_{LV}=\frac{1}{P}\sum_{\Omega}(\Delta I-\overline{\Delta I})^2",
           "OpenCV Laplacian kernel size = 3; reflect-101 border."),
    Method("tenengrad", "Tenengrad · Sobel", "Derivatives", "mean(Gx² + Gy²)",
           "Squared horizontal and vertical Sobel responses measure edge energy in both directions.",
           "Contrast and noise affect the score; thresholds are not applied.", "Pertuz et al., GRA6, Eq. A.21",
           r"F_T=\frac{1}{P}\sum_{\Omega}(G_x^2+G_y^2)",
           "Gx = I*Sx and Gy = I*Sy; Sobel 3×3; reflect-101 border.", PAPER_URL),
    Method("entropy", "Shannon entropy", "Statistics", "−Σ p(i) log₂ p(i)",
           "Histogram information content in bits, using the conventional Shannon sign.",
           "Noise can increase entropy. The course omits the minus sign; spatial structure is not measured.", "Chapter 6, Eq. 6.6; standard sign",
           r"F_H=-\sum_{i=0}^{255}p_i\log_2 p_i",
           "256 bins; round and clip to 0–255; terms with p=0 omitted."),
    Method("autocorrelation", "Autocorrelation", "Statistics", "mean(I(x,y)[I(x+1,y) − I(x+2,y)])",
           "Difference between one- and two-pixel-lag autocorrelations on the same valid support.",
           "May be negative; depends on contrast, brightness and noise.", "Chapter 6, Eq. 6.7",
           r"F_A=\frac{1}{P_2}\sum_{\Omega_2}I(x,y)[I(x+1,y)-I(x+2,y)]",
           "Horizontal lags 1 and 2; common valid support for both terms."),
    Method("wavelet1", "Haar wavelet · L1", "Wavelets", "mean(|HL| + |LH| + |HH|)",
           "Mean absolute detail coefficients from one orthonormal Haar level.",
           "High-frequency noise contributes. An unmatched last row or column is excluded.", "Chapter 6, Eq. 6.8",
           r"F_{W1}=\frac{1}{Q}\sum(|W_{HL}|+|W_{LH}|+|W_{HH}|)",
           "One-level 2×2 orthonormal Haar; each detail coefficient has factor 1/2."),
    Method("wavelet2", "Haar wavelet · variance", "Wavelets", "Var(|HL|) + Var(|LH|) + Var(|HH|)",
           "Variation of the absolute Haar detail coefficients measures high-frequency texture changes.",
           "Noise sensitive. The course sum is normalized by the number of coefficients.", "Chapter 6, Eq. 6.9",
           r"F_{W2}=\sum_{b\in\{HL,LH,HH\}}\mathrm{Var}(|W_b|)",
           "One-level orthonormal Haar; population variances over detail coefficients."),
    Method("modified_laplacian", "Modified Laplacian · SML", "Research / Laplacian", "mean(|I*Lx| + |I*Ly|)",
           "Adds absolute directional second differences before they can cancel. A global mean adaptation of LAP2.",
           "Noise can create second differences. Thresholding and a local SFF window are not used here.", "Pertuz et al. (2013), LAP2, Eqs. A.24–A.25",
           r"F_{ML}=\frac{1}{P}\sum_{\Omega}(|I*L_x|+|I*L_y|)",
           "Lx=[−1,2,−1]; Ly=Lxᵀ; step=1; reflect-101 border; no threshold.", PAPER_URL),
    Method("gaussian_derivative", "Gaussian derivative energy", "Research / Gradient", "mean((I*∂xGσ)² + (I*∂yGσ)²)",
           "First derivatives of a Gaussian select edge structure at a chosen scale. A discrete global version of GRA1.",
           "Scale changes which details contribute. The fixed internal smoothing is part of this operator.", "Pertuz et al. (2013), GRA1, Eqs. A.15–A.16",
           r"F_{GD}=\frac{1}{P}\sum_{\Omega}[(I*\partial_xG_\sigma)^2+(I*\partial_yG_\sigma)^2]",
           "Internal σ=1 px; separable 7-tap derivative kernels; Gaussian unit sum; reflect-101 border.", PAPER_URL),
    Method("tenengrad_variance", "Tenengrad variance", "Research / Gradient", "Var(√(Gx² + Gy²))",
           "Variance of Sobel gradient magnitude, rather than mean squared energy. Global mean adaptation of GRA7.",
           "Texture distribution and noise influence the score; thresholds are not used.", "Pertuz et al. (2013), GRA7, Eq. A.22",
           r"F_{TV}=\frac{1}{P}\sum_{\Omega}(\sqrt{G_x^2+G_y^2}-\overline{G})^2",
           "Sobel 3×3; population variance of gradient magnitude; reflect-101 border.", PAPER_URL),
)
BY_KEY = {m.key: m for m in METHODS}
DEFAULT_METHODS = ("normalized_variance", "tenengrad", "laplacian_variance", "modified_laplacian", "gaussian_derivative", "tenengrad_variance")


def calculate_focus(image: np.ndarray, methods: tuple[str, ...] | list[str], timings=None) -> dict[str, float]:
    """Score finite grayscale floats; timings capture individual operator work only."""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 3 or not np.isfinite(a).all():
        raise ValueError("Analysis requires a finite grayscale image of at least 3×3 pixels.")
    unknown = set(methods) - BY_KEY.keys()
    if unknown:
        raise ValueError(f"Unknown focus measures: {unknown}")
    out = {}
    for key in methods:
        started = perf_counter()
        if key == "normalized_variance":
            mu = float(a.mean())
            value = float(a.var() / mu) if mu > 1e-12 else 0.0
        elif key == "squared_gradient":
            value = np.mean(np.diff(a, axis=1) ** 2)
        elif key in ("brenner", "brenner_squared"):
            diff = a[:, 2:] - a[:, :-2]
            value = np.mean(np.abs(diff) if key == "brenner" else diff ** 2)
        elif key == "laplacian_variance":
            value = cv2.Laplacian(a, cv2.CV_64F, ksize=3, borderType=cv2.BORDER_REFLECT_101).var()
        elif key in ("tenengrad", "tenengrad_variance"):
            gx = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
            energy = gx * gx + gy * gy
            value = energy.mean() if key == "tenengrad" else np.sqrt(energy).var()
        elif key == "modified_laplacian":
            kernel = np.array([-1., 2., -1.])
            dx = cv2.sepFilter2D(a, cv2.CV_64F, kernel, np.ones(1))
            dy = cv2.sepFilter2D(a, cv2.CV_64F, np.ones(1), kernel)
            value = np.mean(np.abs(dx) + np.abs(dy))
        elif key == "gaussian_derivative":
            xx = np.arange(-3, 4, dtype=float)
            g = np.exp(-xx**2 / 2)
            g /= g.sum()
            derivative = -xx * g
            gx = cv2.sepFilter2D(a, cv2.CV_64F, derivative, g)
            gy = cv2.sepFilter2D(a, cv2.CV_64F, g, derivative)
            value = np.mean(gx*gx + gy*gy)
        elif key == "entropy":
            hist = np.bincount(np.clip(np.rint(a), 0, 255).astype(np.uint8).ravel(), minlength=256)
            p = hist[hist > 0] / a.size
            value = -np.sum(p * np.log2(p))
        elif key == "autocorrelation":
            value = np.mean(a[:, :-2] * (a[:, 1:-1] - a[:, 2:]))
        else:
            h, w = a.shape
            c = a[:h // 2 * 2, :w // 2 * 2]
            p, q, r, s = c[::2, ::2], c[::2, 1::2], c[1::2, ::2], c[1::2, 1::2]
            bands = (np.abs((p + q - r - s) / 2), np.abs((p - q + r - s) / 2), np.abs((p - q - r + s) / 2))
            value = sum(float(b.mean() if key == "wavelet1" else b.var()) for b in bands)
        out[key] = float(value)
        if timings is not None:
            timings.setdefault(key, []).append((perf_counter() - started) * 1000)
    return out
