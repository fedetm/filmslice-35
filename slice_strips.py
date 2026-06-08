# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "Pillow",
# ]
# ///

"""
slice_strips.py — film strip frame slicer

Detects inter-frame gaps in scanned film strips and exports each frame as
an individual TIFF file.  Designed for use with film holders (sprocket holes
are never visible in the scan).

Usage:
    python slice_strips.py INPUT_FOLDER [options]
    python slice_strips.py INPUT_FOLDER --dry-run -v
    python slice_strips.py INPUT_FOLDER --threshold 36 --dry-run -v
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None   # scans legitimately exceed Pillow's decompression-bomb guard

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_GAP_PX = 10
DEFAULT_MIN_FRAME_PX = 1800
DEFAULT_EXTENSIONS = ["tif", "tiff", "jpg", "jpeg", "png"]
DEFAULT_GAP_THRESHOLD = 12       # max auto-threshold: real inter-frame borders are near-black (<10)
                                 # keeping this tight avoids misclassifying dark frame content as gaps
MIN_GAP_DARK_FRACTION = 0.80     # ≥80% of trimmed-row pixels must be "dark" to count as a gap;
                                 # rejects photo rows with a single dark corner that drag the mean down
DARK_PIXEL_MULTIPLIER = 2.5      # pixel threshold for the dark-fraction test = threshold × this value;
                                 # e.g., at default threshold=12 → test pixels < 30, so slightly-lifted
                                 # film-base (mean=15-20) still qualifies while bright photo rows don't
EDGE_TRIM_FRACTION = 0.155       # trim this fraction from each column edge before computing row means;
                                 # excludes bright holder/film-edge artefacts that skew gap detection

HOLDER_BRIGHTNESS = 240   # rows at or above this mean are considered empty holder material
HOLDER_STD = 10           # rows below this std are too uniform to be real film content
HOLDER_SCAN_WINDOW = 10   # sliding-window size for the inward holder scan

CONTACT_THUMB_HEIGHT = 300   # px — thumbnail height in the contact sheet
CONTACT_PADDING = 8          # px — gap between thumbnails and edges
CONTACT_LABEL_H = 18         # px — space below thumbnails for frame-number labels


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GapRegion:
    y_start: int
    y_end: int  # inclusive

    @property
    def height(self) -> int:
        return self.y_end - self.y_start + 1


@dataclass
class FrameRegion:
    y_start: int
    y_end: int  # inclusive
    index: int  # 1-based

    @property
    def height(self) -> int:
        return self.y_end - self.y_start + 1


@dataclass
class CropBox:
    x_left: int
    x_right: int  # exclusive (PIL convention)
    method: str   # "sprocket" | "geometric" | "full"

    @property
    def width(self) -> int:
        return self.x_right - self.x_left


@dataclass
class ProcessingResult:
    filepath: Path
    n_frames: int = 0
    frames: List[FrameRegion] = field(default_factory=list)
    crop_box: Optional[CropBox] = None
    threshold_used: float = 0.0
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice scanned film strips into individual frame TIFFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slice_strips.py ~/scans/
  python slice_strips.py ~/scans/ --dry-run -v
  python slice_strips.py ~/scans/ --threshold 36 --max-gap 750
  python slice_strips.py ~/scans/ --output ~/frames/
""",
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing scanned film strip images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <script folder>/output).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        metavar="INT",
        help="Gap darkness threshold 0-255. Rows with brightness below this are gaps. "
             "Default: auto-detected via 1-D Otsu, capped at 12. "
             "Increase to 36 if too few frames are detected (film base brighter than default cap).",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=DEFAULT_MIN_GAP_PX,
        metavar="INT",
        help=f"Minimum gap height in pixels to be considered a real gap. "
             f"Default: {DEFAULT_MIN_GAP_PX}.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=None,
        metavar="INT",
        help="Maximum gap height in pixels. Gaps taller than this are discarded. "
             "Useful with --threshold 36 to remove large false gaps caused by very "
             "dark frame content. Default: disabled (no upper limit). "
             "Suggested value for dark rolls: 750.",
    )
    parser.add_argument(
        "--min-frame",
        type=int,
        default=DEFAULT_MIN_FRAME_PX,
        metavar="INT",
        help=f"Minimum frame height in pixels. Smaller regions are discarded. "
             f"Default: {DEFAULT_MIN_FRAME_PX}.",
    )
    parser.add_argument(
        "--frame-trim",
        type=int,
        default=0,
        metavar="INT",
        help="Pixels to trim from the top and bottom of every exported frame "
             "to remove the film rebate. Default: 0.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(DEFAULT_EXTENSIONS),
        metavar="EXTS",
        help=f"Comma-separated file extensions to process. "
             f"Default: {','.join(DEFAULT_EXTENSIONS)}.",
    )
    parser.add_argument(
        "--no-holder-trim",
        action="store_true",
        help="Disable automatic trimming of bright film-holder areas at the "
             "leading and trailing edges of each scan. On by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detected frames and crop boxes without writing any files.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-gap and per-frame debug information.",
    )

    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"Input folder does not exist: {args.input_folder}")

    if args.threshold is not None and not (0 <= args.threshold <= 255):
        parser.error("--threshold must be between 0 and 255.")

    args.extensions = [e.lower().lstrip(".") for e in args.extensions.split(",") if e.strip()]

    if args.output is None:
        args.output = Path(__file__).parent / "output"

    return args


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_input_files(folder: Path, extensions: List[str], output_dir: Path) -> List[Path]:
    """Return sorted list of image files in folder, excluding the output subfolder."""
    files = []
    for path in sorted(folder.iterdir()):
        if path.is_dir():
            continue
        try:
            path.relative_to(output_dir)
            continue  # skip files inside the output directory
        except ValueError:
            pass
        if path.suffix.lower().lstrip(".") in extensions:
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_cv(filepath: Path) -> np.ndarray:
    """Load image as float32 grayscale in [0, 255] range for analysis."""
    pil_img = Image.open(filepath)

    # Normalise mode so np.array always yields a 2-D (H×W) array
    if pil_img.mode in ('I;16', 'I;16B'):
        pil_img = pil_img.convert('I')   # raw 16-bit (any endian) → 32-bit signed int
    elif pil_img.mode not in ('L', 'I', 'F'):
        pil_img = pil_img.convert('L')   # RGB/RGBA/etc. → 8-bit grayscale

    img = np.array(pil_img, dtype=np.float32)

    # Normalise 16-bit to [0, 255] so --threshold values are always in 0-255 range
    if img.max() > 255.0:
        img = img / 65535.0 * 255.0

    return img


def load_image_pil(filepath: Path) -> Image.Image:
    """Load image with PIL, preserving original bit depth and mode."""
    return Image.open(filepath)


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def compute_row_means(gray: np.ndarray, trim_fraction: float = 0.0) -> np.ndarray:
    """Mean brightness per row, shape (H,). Trims outer columns to exclude sprocket holes."""
    if trim_fraction > 0:
        W = gray.shape[1]
        x0 = int(W * trim_fraction)
        x1 = W - x0
        if x1 > x0:
            return gray[:, x0:x1].mean(axis=1)
    return gray.mean(axis=1)


def compute_row_stds(gray: np.ndarray, trim_fraction: float = 0.0) -> np.ndarray:
    """Std deviation per row, shape (H,). Same column trimming as compute_row_means."""
    if trim_fraction > 0:
        W = gray.shape[1]
        x0 = int(W * trim_fraction)
        x1 = W - x0
        if x1 > x0:
            return gray[:, x0:x1].std(axis=1)
    return gray.std(axis=1)


def compute_row_dark_fractions(
    gray: np.ndarray,
    threshold: float,
    trim_fraction: float = 0.0,
) -> np.ndarray:
    """Fraction of pixels per row that are below threshold, shape (H,). Applies same column trim."""
    if trim_fraction > 0:
        W = gray.shape[1]
        x0 = int(W * trim_fraction)
        x1 = W - x0
        if x1 > x0:
            return (gray[:, x0:x1] < threshold).mean(axis=1)
    return (gray < threshold).mean(axis=1)


def auto_threshold(row_means: np.ndarray) -> float:
    """
    1-D Otsu threshold on the row_means distribution.
    Returns the threshold in the same scale as row_means (normalised to [0,255]).
    Falls back to the 25th percentile for unimodal distributions.
    """
    vmin, vmax = float(row_means.min()), float(row_means.max())
    if vmax == vmin:
        return vmin

    scaled = ((row_means - vmin) / (vmax - vmin) * 255.0).astype(np.int32)
    scaled = np.clip(scaled, 0, 255)

    hist, _ = np.histogram(scaled, bins=256, range=(0, 256))
    prob = hist.astype(np.float64) / hist.sum()

    # Prefix sums for O(1) per-candidate evaluation
    bins = np.arange(256, dtype=np.float64)
    cumw = np.cumsum(prob)
    cumwp = np.cumsum(prob * bins)

    best_t, best_var = 1, -1.0
    for t in range(1, 255):
        w0 = cumw[t - 1]
        w1 = 1.0 - w0
        if w0 <= 0 or w1 <= 0:
            continue
        mu0 = cumwp[t - 1] / w0
        mu1 = (cumwp[255] - cumwp[t - 1]) / w1
        var_b = w0 * w1 * (mu0 - mu1) ** 2
        if var_b > best_var:
            best_var = var_b
            best_t = t

    # Unimodality: fall back to 25th percentile
    total_var = float(np.var(scaled))
    if total_var > 0 and best_var < 0.01 * total_var:
        return min(float(np.percentile(row_means, 25)), DEFAULT_GAP_THRESHOLD)

    # Convert bin index back to original value scale; cap to avoid misclassifying
    # dark frame content as inter-frame gap (sprocket-zone brightness inflates Otsu)
    return min(float(best_t) / 255.0 * (vmax - vmin) + vmin, DEFAULT_GAP_THRESHOLD)


def _find_gap_runs(
    row_means: np.ndarray,
    threshold: float,
    min_gap_px: int,
) -> List[GapRegion]:
    """Core detection: contiguous runs below threshold of at least min_gap_px rows."""
    is_gap = row_means < threshold

    padded = np.concatenate([[False], is_gap, [False]])
    edges = np.diff(padded.astype(np.int32))
    starts = np.where(edges == 1)[0]   # inclusive
    ends = np.where(edges == -1)[0]    # exclusive

    gaps = []
    for s, e in zip(starts, ends):
        g = GapRegion(y_start=int(s), y_end=int(e) - 1)
        if g.height >= min_gap_px:
            gaps.append(g)
    return gaps


def detect_gaps(
    row_means: np.ndarray,
    threshold: float,
    min_gap_px: int,
    max_gap_px: Optional[int] = None,
    gray: Optional[np.ndarray] = None,
) -> List[GapRegion]:
    """Find inter-frame gaps; refine wide gaps caused by dark photographic content."""
    # Pass 1: mean-based detection (original behavior).
    gaps = _find_gap_runs(row_means, threshold, min_gap_px)

    image_height = len(row_means)

    # Pass 2: dark-fraction supplement — catch narrow gaps whose row mean is slightly
    # above threshold because one photo corner is bright.  Only gaps that are (a) small
    # enough to be a real film-base strip, (b) do not overlap any mean-detected gap, and
    # (c) produce frames on both sides that are at least 75% of the typical frame height.
    # The size cap and the frame-height guard together prevent false cuts inside dark photos.
    if gray is not None:
        # Derive typical frame height from pass-1 inter-gap intervals (≥ min-frame only,
        # to exclude tiny leading/trailing edges).
        _prev = -1
        _p1_heights: List[int] = []
        for _g in gaps:
            _h = _g.y_start - _prev - 1
            if _h >= DEFAULT_MIN_FRAME_PX:
                _p1_heights.append(_h)
            _prev = _g.y_end
        _tail = image_height - _prev - 1
        if _tail >= DEFAULT_MIN_FRAME_PX:
            _p1_heights.append(_tail)
        _pass1_median = float(np.median(_p1_heights)) if _p1_heights else float(DEFAULT_MIN_FRAME_PX)
        _min_candidate_frame_h = _pass1_median * 0.75

        dark_fracs = compute_row_dark_fractions(
            gray, threshold * DARK_PIXEL_MULTIPLIER, trim_fraction=EDGE_TRIM_FRACTION,
        )
        is_dark_gap = dark_fracs >= MIN_GAP_DARK_FRACTION
        padded = np.concatenate([[False], is_dark_gap, [False]])
        df_edges = np.diff(padded.astype(np.int32))
        df_starts = np.where(df_edges == 1)[0]
        df_ends = np.where(df_edges == -1)[0]
        _MAX_SUPPLEMENT_GAP = min_gap_px * 15  # e.g., ≤150px at default min_gap=10
        for s, e in zip(df_starts, df_ends):
            cand = GapRegion(int(s), int(e) - 1)
            if cand.height < min_gap_px or cand.height > _MAX_SUPPLEMENT_GAP:
                continue
            # Skip candidates that overlap an already-detected gap.
            if any(cand.y_start <= g.y_end and cand.y_end >= g.y_start for g in gaps):
                continue
            # Reject if either resulting frame would be too small vs. typical frame height.
            _prev_end = max((g.y_end for g in gaps if g.y_end < cand.y_start), default=-1)
            _next_start = min((g.y_start for g in gaps if g.y_start > cand.y_end), default=image_height)
            if (cand.y_start - _prev_end - 1 < _min_candidate_frame_h
                    or _next_start - cand.y_end - 1 < _min_candidate_frame_h):
                continue
            gaps.append(cand)
        gaps.sort(key=lambda g: g.y_start)

    _LARGE_GAP_FACTOR = 4         # gaps > 4 × min_gap_px are candidates for refinement
    strict_threshold = threshold / 3.0  # ~12 at default threshold=36; catches film-base only

    refined = []
    for gap in gaps:
        is_interior = image_height - gap.y_end > DEFAULT_MIN_FRAME_PX
        if gap.height <= _LARGE_GAP_FACTOR * min_gap_px or not is_interior:
            refined.append(gap)
            continue
        sub = _find_gap_runs(row_means[gap.y_start:gap.y_end + 1], strict_threshold, min_gap_px)
        if sub:
            for sg in sub:
                refined.append(GapRegion(gap.y_start + sg.y_start, gap.y_start + sg.y_end))
        else:
            refined.append(gap)

    # Discard gaps that exceed the user-specified maximum height.
    if max_gap_px is not None:
        refined = [g for g in refined if g.height <= max_gap_px]

    # Remove a large trailing gap that would abnormally truncate the last frame.
    # "Trailing" = no frame-sized content follows. "Abnormal" = last frame < 80% of typical.
    if len(refined) >= 2:
        last = refined[-1]
        if (last.height > _LARGE_GAP_FACTOR * min_gap_px
                and image_height - last.y_end <= DEFAULT_MIN_FRAME_PX):
            interior = refined[:-1]
            last_frame_start = interior[-1].y_end + 1
            last_frame_height = last.y_start - last_frame_start
            mid_heights = [
                interior[i + 1].y_start - interior[i].y_end - 1
                for i in range(len(interior) - 1)
                if interior[i + 1].y_start - interior[i].y_end - 1 > 0
            ]
            if mid_heights:
                typical_h = float(np.median(mid_heights))
                if last_frame_height < 0.8 * typical_h:
                    refined = refined[:-1]

    return refined


def gaps_to_frames(
    gaps: List[GapRegion],
    image_height: int,
    min_frame_px: int,
) -> List[FrameRegion]:
    """
    Compute frame regions as the intervals between gaps.
    Includes leading and trailing partial frames if large enough.
    """
    if not gaps:
        if image_height >= min_frame_px:
            return [FrameRegion(y_start=0, y_end=image_height - 1, index=1)]
        return []

    starts, ends = [], []

    # Leading frame
    if gaps[0].y_start > 0:
        starts.append(0)
        ends.append(gaps[0].y_start - 1)

    # Frames between consecutive gaps
    for i in range(len(gaps) - 1):
        starts.append(gaps[i].y_end + 1)
        ends.append(gaps[i + 1].y_start - 1)

    # Trailing frame
    if gaps[-1].y_end < image_height - 1:
        starts.append(gaps[-1].y_end + 1)
        ends.append(image_height - 1)

    frames, idx = [], 1
    for s, e in zip(starts, ends):
        if e < s:
            continue
        f = FrameRegion(y_start=s, y_end=e, index=idx)
        if f.height >= min_frame_px:
            frames.append(f)
            idx += 1

    return frames


def _is_holder_window(
    row_means: np.ndarray,
    row_stds: np.ndarray,
    y: int,
    window: int,
    brightness: float,
    std_thresh: float,
) -> bool:
    """Return True if every row in [y, y+window) looks like empty holder material."""
    end = min(y + window, len(row_means))
    return bool(
        np.all(row_means[y:end] >= brightness) and np.all(row_stds[y:end] < std_thresh)
    )


def trim_holder_edges(
    frames: List[FrameRegion],
    row_means: np.ndarray,
    row_stds: np.ndarray,
    min_frame_px: int,
    brightness: float = HOLDER_BRIGHTNESS,
    std_thresh: float = HOLDER_STD,
    window: int = HOLDER_SCAN_WINDOW,
) -> List[FrameRegion]:
    """
    Trim uniformly bright film-holder material from the outer edges of the
    leading and trailing frames.  A row is considered holder material when its
    mean brightness ≥ `brightness` AND its std deviation < `std_thresh`.
    Scans inward using a sliding window so a single noisy row doesn't
    prematurely stop the trim.  Frames that become smaller than `min_frame_px`
    after trimming are dropped entirely.
    """
    if not frames:
        return frames

    result = list(frames)

    # --- Trim the leading edge of the first frame ---
    first = result[0]
    new_start = first.y_start
    y = first.y_start
    while y < first.y_end and _is_holder_window(row_means, row_stds, y, window, brightness, std_thresh):
        new_start = y + window
        y += window
    if new_start != first.y_start:
        trimmed = FrameRegion(y_start=new_start, y_end=first.y_end, index=first.index)
        if trimmed.height >= min_frame_px:
            result[0] = trimmed
        else:
            result = result[1:]

    if not result:
        return result

    # --- Trim the trailing edge of the last frame ---
    last = result[-1]
    new_end = last.y_end
    y = last.y_end - window + 1
    while y > last.y_start and _is_holder_window(row_means, row_stds, y, window, brightness, std_thresh):
        new_end = y - 1
        y -= window
    if new_end != last.y_end:
        trimmed = FrameRegion(y_start=last.y_start, y_end=new_end, index=last.index)
        if trimmed.height >= min_frame_px:
            result[-1] = trimmed
        else:
            result = result[:-1]

    return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_frames(
    pil_image: Image.Image,
    frames: List[FrameRegion],
    crop_box: CropBox,
    output_dir: Path,
    stem: str,
    verbose: bool,
    frame_trim: int = 0,
) -> None:
    """Crop and save each frame as a LZW-compressed TIFF."""
    for frame in frames:
        filename = f"{stem}_{frame.index:02d}.tiff"
        out_path = output_dir / filename

        y0 = min(frame.y_start + frame_trim, frame.y_end)
        y1 = max(frame.y_end + 1 - frame_trim, y0 + 1)
        crop_tuple = (crop_box.x_left, y0, crop_box.x_right, y1)
        cropped = pil_image.crop(crop_tuple)
        cropped.save(str(out_path), format="TIFF", compression="tiff_lzw")

        if verbose:
            print(f"  [DEBUG] Wrote {filename}: "
                  f"{crop_box.width}x{frame.height}px, mode={cropped.mode}")


def generate_contact_sheet(
    pil_image: Image.Image,
    frames: List[FrameRegion],
    crop_box: CropBox,
    output_dir: Path,
    stem: str,
    frame_trim: int = 0,
) -> None:
    """Save a side-by-side JPEG overview of all extracted frames."""
    thumbs = []
    for frame in frames:
        y0 = min(frame.y_start + frame_trim, frame.y_end)
        y1 = max(frame.y_end + 1 - frame_trim, y0 + 1)
        crop_tuple = (crop_box.x_left, y0, crop_box.x_right, y1)
        cropped = pil_image.crop(crop_tuple)

        # Normalise to 8-bit grayscale for JPEG output
        if cropped.mode not in ('L', 'RGB'):
            arr = np.array(cropped, dtype=np.float32)
            if arr.max() > 255.0:
                arr = arr / 65535.0 * 255.0
            cropped = Image.fromarray(arr.astype(np.uint8), mode='L')

        ratio = CONTACT_THUMB_HEIGHT / cropped.height
        thumb_w = max(1, int(cropped.width * ratio))
        thumbs.append(cropped.resize((thumb_w, CONTACT_THUMB_HEIGHT), Image.Resampling.LANCZOS))

    pad = CONTACT_PADDING
    total_w = sum(t.width + 2 for t in thumbs) + pad * (len(thumbs) + 1)
    total_h = CONTACT_THUMB_HEIGHT + 2 + pad * 2 + CONTACT_LABEL_H

    sheet = Image.new('L', (total_w, total_h), color=220)
    draw = ImageDraw.Draw(sheet)

    x = pad
    for i, thumb in enumerate(thumbs):
        y = pad
        draw.rectangle([x - 1, y - 1, x + thumb.width, y + thumb.height], outline=60)
        sheet.paste(thumb, (x, y))
        draw.text((x + 2, y + CONTACT_THUMB_HEIGHT + 4), f"{i + 1:02d}", fill=40)
        x += thumb.width + 2 + pad

    contacts_dir = output_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(str(contacts_dir / f"{stem}_contact.jpg"), format='JPEG', quality=85)


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process_file(
    filepath: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> ProcessingResult:
    result = ProcessingResult(filepath=filepath)

    try:
        gray = load_image_cv(filepath)
    except Exception as e:
        result.warnings.append(f"Could not read image: {e}")
        return result

    transposed = gray.shape[0] < gray.shape[1]  # landscape TIFF stored sideways
    if transposed:
        gray = gray.T

    H, W = gray.shape

    if args.verbose:
        print(f"  [DEBUG] {filepath.name}: {W}x{H}px, "
              f"brightness=[{gray.min():.1f}, {gray.max():.1f}]")

    row_means = compute_row_means(gray, trim_fraction=EDGE_TRIM_FRACTION)

    threshold = (float(args.threshold) if args.threshold is not None
                 else auto_threshold(row_means))
    result.threshold_used = threshold

    if args.verbose:
        src = "user" if args.threshold is not None else "auto/Otsu"
        print(f"  [DEBUG] Row means: min={row_means.min():.1f}, "
              f"max={row_means.max():.1f}, mean={row_means.mean():.1f}")
        print(f"  [DEBUG] Threshold: {threshold:.1f} ({src})")

    gaps = detect_gaps(row_means, threshold, args.min_gap, args.max_gap, gray=gray)

    if args.verbose:
        dark_fracs = compute_row_dark_fractions(
            gray, threshold * DARK_PIXEL_MULTIPLIER, trim_fraction=EDGE_TRIM_FRACTION,
        )
        print(f"  [DEBUG] Gaps: {len(gaps)}")
        for i, g in enumerate(gaps):
            frac_min = dark_fracs[g.y_start:g.y_end + 1].min()
            frac_max = dark_fracs[g.y_start:g.y_end + 1].max()
            print(f"  [DEBUG]   gap {i}: y=[{g.y_start}:{g.y_end}] h={g.height}px "
                  f"dark_frac=[{frac_min:.2f}..{frac_max:.2f}]")

    frames = gaps_to_frames(gaps, H, args.min_frame)

    if not args.no_holder_trim:
        row_stds = compute_row_stds(gray, trim_fraction=EDGE_TRIM_FRACTION)
        frames = trim_holder_edges(frames, row_means, row_stds, args.min_frame)

    if args.verbose:
        print(f"  [DEBUG] Frames: {len(frames)}")
        for f in frames:
            print(f"  [DEBUG]   frame {f.index}: y=[{f.y_start}:{f.y_end}] h={f.height}px")

    if not frames:
        result.warnings.append(
            f"No frames detected (threshold={threshold:.1f}). "
            "Try a lower --threshold or --min-frame value."
        )
        return result

    # Warn when very few frames are found in a long strip — this usually means
    # the inter-frame gaps are lighter than the default threshold captures.
    # Typical case: scans where the film base between frames sits at ~25-35 instead
    # of near-black (~4-8).  Running with --threshold 36 often fixes this.
    if len(frames) <= 2 and H > 15000 and args.threshold is None:
        result.warnings.append(
            f"Only {len(frames)} frame(s) detected in a {H}px tall strip "
            f"(threshold={threshold:.1f}). "
            "Inter-frame gaps may be lighter than the default cap. "
            "Try --threshold 36 (and --max-gap 750 if over-detection persists) for this file."
        )

    crop_box = CropBox(x_left=0, x_right=W, method="full")
    result.crop_box = crop_box
    result.frames = frames
    result.n_frames = len(frames)

    if args.dry_run:
        for frame in frames:
            filename = f"{filepath.stem}_{frame.index:02d}.tiff"
            print(f"  [DRY RUN] {filename}: "
                  f"y=[{frame.y_start}:{frame.y_end}], "
                  f"x=[{crop_box.x_left}:{crop_box.x_right}], "
                  f"size={crop_box.width}x{frame.height}px")
        return result

    try:
        pil_image = load_image_pil(filepath)
    except Exception as e:
        result.warnings.append(f"PIL could not read image for export: {e}")
        return result

    if transposed:
        pil_image = pil_image.transpose(Image.Transpose.TRANSPOSE)

    export_frames(pil_image, frames, crop_box, output_dir, filepath.stem, args.verbose,
                  frame_trim=args.frame_trim)
    generate_contact_sheet(pil_image, frames, crop_box, output_dir, filepath.stem,
                           frame_trim=args.frame_trim)
    return result


# ---------------------------------------------------------------------------
# Summary and main
# ---------------------------------------------------------------------------

def print_file_summary(result: ProcessingResult) -> None:
    name = result.filepath.name

    if result.n_frames == 0:
        for w in result.warnings:
            print(f"{name} → WARNING: {w}")
        return

    print(f"{name} → {result.n_frames} frame(s)  [threshold={result.threshold_used:.1f}]")

    for w in result.warnings:
        print(f"  [WARN] {w}")


def main() -> None:
    args = parse_args()

    input_folder = args.input_folder.resolve()
    output_dir = args.output.resolve()

    files = collect_input_files(input_folder, args.extensions, output_dir)
    if not files:
        print(f"No image files found in {input_folder} "
              f"(extensions: {', '.join(args.extensions)})")
        sys.exit(0)

    if not args.dry_run:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: could not create output directory {output_dir}: {e}")
            sys.exit(1)

    label = "  [DRY RUN — no files written]" if args.dry_run else ""
    print(f"Input:  {input_folder}")
    print(f"Output: {output_dir}{label}")
    print(f"Files:  {len(files)} found")
    print()

    total_frames = 0
    skipped = 0

    for filepath in files:
        if args.verbose:
            print(f"Processing: {filepath.name}")

        result = process_file(filepath, output_dir, args)
        print_file_summary(result)

        if result.n_frames == 0:
            skipped += 1
        else:
            total_frames += result.n_frames

    print()
    print("---")
    processed = len(files) - skipped
    skip_str = f" ({skipped} skipped)" if skipped else ""
    print(f"Total: {total_frames} frame(s) from {processed} file(s){skip_str}")


if __name__ == "__main__":
    main()
