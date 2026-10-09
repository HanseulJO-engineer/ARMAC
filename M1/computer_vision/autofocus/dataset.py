"""Read real directories or ZIP archives without modifying the original dataset."""
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from contextlib import contextmanager
import re
import cv2
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pgm"}


def natural_key(name: str):
    return [int(s) if s.isdigit() else s.casefold() for s in re.split(r"(\d+)", name)]


@dataclass(frozen=True)
class ImageEntry:
    name: str
    path: str
    sequence: int
    distance_mm: float


@dataclass
class Dataset:
    name: str
    source: Path
    images: list[ImageEntry]
    position_source: str
    warnings: list[str] = field(default_factory=list)
    archived: bool = False

    @contextmanager
    def reader(self):
        if self.archived:
            with ZipFile(self.source) as archive:
                yield lambda e: decode_image(archive.read(e.path))
        else:
            yield lambda e: decode_image(Path(e.path).read_bytes())

    def load(self, entry: ImageEntry) -> np.ndarray:
        with self.reader() as read:
            return read(entry)

    def display_path(self, entry: ImageEntry) -> str:
        return f"{self.source} :: {entry.path}" if self.archived else entry.path


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot decode the image. Check its format or file integrity.")
    return image


def grayscale(image: np.ndarray) -> np.ndarray:
    """Use a fixed dtype scale, never an independent contrast stretch per image."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported number of image channels.")
    if image.ndim != 2:
        raise ValueError("Unsupported image dimensions.")
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float64) * (255.0 / np.iinfo(image.dtype).max)
    result = image.astype(np.float64)
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 255:
        raise ValueError("Floating-point images must use the intensity range [0,255].")
    return result


def _wd_values(text: str) -> list[float]:
    # OpenCV's %YAML:1.0 header is not standard YAML. Only parse the numeric WD vector.
    match = re.search(r"\bWD\s*:\s*\[([^\]]+)\]", text, re.S)
    if not match:
        raise ValueError("The metadata has no numeric WD: [...] vector.")
    values = [float(s.strip()) for s in match.group(1).split(",") if s.strip()]
    if not values or not np.isfinite(values).all():
        raise ValueError("Invalid WD metadata values.")
    return values


def open_dataset(source: str | Path, start_mm: float = 10.0, step_um: float = 10.0,
                 use_metadata: bool = True, metadata_offset: int = 0) -> Dataset:
    path = Path(source).expanduser().resolve()
    if step_um <= 0 or not np.isfinite([start_mm, step_um]).all():
        raise ValueError("Enter a finite starting distance and a positive step size.")
    if metadata_offset not in (0, 1):
        raise ValueError("WD offset must be 0 or 1.")
    archived = path.is_file() and path.suffix.lower() == ".zip"
    if archived:
        try:
            with ZipFile(path) as z:
                names = [n for n in z.namelist() if not n.startswith("__MACOSX/") and Path(n).suffix.lower() in EXTENSIONS]
                names.sort(key=natural_key)
                # An archive must represent ONE imageset, not intermixed scanning speeds.
                if len({str(Path(n).parent) for n in names}) > 1:
                    raise ValueError("This ZIP contains multiple stacks. Select one stack folder or ZIP.")
                metas = [n for n in z.namelist() if Path(n).suffix.lower() in {".yml", ".yaml"} and "wd" in Path(n).stem.lower()]
                metadata = z.read(metas[0]).decode("utf-8-sig") if len(metas) == 1 else None
        except BadZipFile as e:
            raise ValueError("Invalid ZIP archive.") from e
    elif path.is_dir():
        names = sorted((str(p.resolve()) for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda n: natural_key(Path(n).name))
        metas = sorted(p for p in path.iterdir() if p.suffix.lower() in {".yml", ".yaml"} and "wd" in p.stem.lower())
        metadata = metas[0].read_text(encoding="utf-8-sig") if len(metas) == 1 else None
    else:
        raise ValueError(f"Image folder or ZIP not found: {path}")
    if not names:
        raise ValueError("No images in this folder. Choose the subfolder containing the images.")
    warnings = []
    distances = None
    if use_metadata and metadata is not None:
        values = _wd_values(metadata)
        if len(values) < len(names) + metadata_offset:
            raise ValueError(f"Cannot map {len(values)} WD values to {len(names)} images. Check the offset or use manual distances.")
        distances = values[metadata_offset:metadata_offset + len(names)]
        position_source = f"{Path(metas[0]).name} · WD[{metadata_offset}:] · mm"
        if len(values) != len(names):
            warnings.append(f"{len(values)} WD values / {len(names)} images. First naturally sorted image ↔ WD[{metadata_offset}]. The offset is selectable because the source alignment is ambiguous.")
        median_step = np.median(np.abs(np.diff(distances))) * 1000 if len(distances) > 1 else step_um
        if abs(median_step - 10) > 0.01 or abs(min(distances) - 10) > 0.001:
            warnings.append(f"Actual WD: minimum {min(distances):.5f} mm, median step {median_step:.3f} µm. Metadata takes precedence over the lab text (10 mm / 10 µm).")
    else:
        distances = [start_mm + i * step_um / 1000 for i in range(len(names))]
        position_source = f"Manual · {start_mm:g} mm + index × {step_um:g} µm"
        if use_metadata:
            warnings.append("No WD metadata; using the manual start and step.")
        if len(metas) > 1:
            warnings.append("Multiple WD files; automatic mapping is disabled.")
    if len(distances) > 1 and not np.all(np.diff(distances) > 0):
        raise ValueError("WD values must increase strictly in natural filename order.")
    images = [ImageEntry(Path(n).name, n, i, float(distances[i])) for i, n in enumerate(names)]
    return Dataset(path.stem if archived else path.name, path, images, position_source, warnings, archived)


def discover_sources(root: str | Path) -> list[Path]:
    path = Path(root).expanduser().resolve()
    if path.is_file():
        return [path] if path.suffix.lower() == ".zip" else []
    if not path.is_dir():
        return []
    sources = [path] if any(p.suffix.lower() in EXTENSIONS for p in path.iterdir() if p.is_file()) else []
    sources.extend(p for p in path.iterdir() if p.is_dir() and any(f.is_file() and f.suffix.lower() in EXTENSIONS for f in p.iterdir()))
    sources.extend(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".zip" and not (path / p.stem).is_dir())
    return sorted(sources, key=lambda p: natural_key(p.name))
