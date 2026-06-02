# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "Pillow",
# ]
# ///

"""
slice_strips.py — film strip frame slicer (35mm and 120 medium format)

Detects inter-frame gaps in scanned film strips, crops out sprocket holes
(35mm only), and exports each frame as an individual TIFF file.

Usage:
    python slice_strips.py INPUT_FOLDER [options]
    python slice_strips.py INPUT_FOLDER --dry-run -v
    python slice_strips.py INPUT_FOLDER --format 120 --dry-run -v
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_GAP_PX = 10
DEFAULT_MIN_FRAME_PX = 200
DEFAULT_EXTENSIONS = ["tif", "tiff", "jpg", "jpeg", "png"]
DEFAULT_SPROCKET_MARGIN_PX = 5
SPROCKET_BRIGHTNESS_RATIO = 0.7   # fraction of gap-region max to detect holes
GEOMETRIC_TRIM_FRACTION = 0.10    # fallback: trim this fraction from each side
DEFAULT_GAP_THRESHOLD = 36       # max auto-threshold for center-trimmed row means

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
        description="Slice scanned film strips into individual frame TIFFs (35mm and 120 medium format).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slice_strips.py ~/scans/
  python slice_strips.py ~/scans/ --dry-run -v
  python slice_strips.py ~/scans/ --threshold 40 --min-frame 300
  python slice_strips.py ~/scans/ --output ~/frames/ --no-crop-sprockets
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
        help="Gap darkness threshold 0-255. Rows darker than this are gaps. "
             "Default: auto-detected via 1-D Otsu on row brightness distribution.",
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
        "--format",
        choices=["35mm", "120"],
        default="35mm",
        help="Film format. '120' disables sprocket detection and uses full-width row means. Default: 35mm.",
    )
    parser.add_argument(
        "--no-crop-sprockets",
        action="store_true",
        help="Skip sprocket-hole detection; export the full strip width.",
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
) -> List[GapRegion]:
    """Find inter-frame gaps; refine wide gaps caused by dark photographic content."""
    gaps = _find_gap_runs(row_means, threshold, min_gap_px)
    image_height = len(row_means)

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


# ---------------------------------------------------------------------------
# Sprocket-hole detection
# ---------------------------------------------------------------------------

def detect_sprocket_crop(
    gray: np.ndarray,
    gaps: List[GapRegion],
    no_crop: bool,
    verbose: bool,
    margin: int = DEFAULT_SPROCKET_MARGIN_PX,
) -> CropBox:
    """
    Determine left/right image-area bounds by detecting sprocket holes in a gap
    region. Falls back to geometric crop if detection fails.
    """
    W = gray.shape[1]

    if no_crop:
        return CropBox(x_left=0, x_right=W, method="full")

    def geometric_fallback() -> CropBox:
        x_left = int(W * GEOMETRIC_TRIM_FRACTION)
        x_right = W - x_left
        if verbose:
            print(f"  [DEBUG] Sprocket crop: geometric fallback "
                  f"({GEOMETRIC_TRIM_FRACTION*100:.0f}% trim each side), "
                  f"x=[{x_left}:{x_right}]")
        return CropBox(x_left=x_left, x_right=x_right, method="geometric")

    if not gaps:
        return geometric_fallback()

    # Try gaps starting from the middle, expanding outward
    mid = len(gaps) // 2
    order = [mid]
    for step in range(1, len(gaps)):
        for d in (step, -step):
            idx = mid + d
            if 0 <= idx < len(gaps) and idx not in order:
                order.append(idx)

    for gi in order:
        gap = gaps[gi]
        if gap.height < 3:
            continue

        gap_strip = gray[gap.y_start: gap.y_end + 1, :]
        col_profile = gap_strip.mean(axis=0)
        profile_max = float(col_profile.max())
        if profile_max == 0:
            continue

        sprocket_thresh = SPROCKET_BRIGHTNESS_RATIO * profile_max

        left_zone = col_profile[: W // 3]
        left_hits = np.where(left_zone > sprocket_thresh)[0]

        right_start = 2 * W // 3
        right_zone = col_profile[right_start:]
        right_hits = np.where(right_zone > sprocket_thresh)[0]

        if left_hits.size == 0 or right_hits.size == 0:
            if verbose:
                print(f"  [DEBUG] Gap {gi}: no sprocket candidates "
                      f"(left={left_hits.size}, right={right_hits.size}), trying next")
            continue

        left_edge = int(left_hits.max())
        right_edge = int(right_hits.min()) + right_start

        x_left = left_edge + 1 + margin
        x_right = right_edge - margin

        if x_left >= x_right or x_left < 0 or x_right > W:
            if verbose:
                print(f"  [DEBUG] Gap {gi}: invalid crop box "
                      f"x_left={x_left}, x_right={x_right}, trying next")
            continue

        if verbose:
            print(f"  [DEBUG] Sprocket crop: method=sprocket via gap {gi} "
                  f"(y=[{gap.y_start}:{gap.y_end}]), x=[{x_left}:{x_right}]")
        return CropBox(x_left=x_left, x_right=x_right, method="sprocket")

    return geometric_fallback()


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
        thumbs.append(cropped.resize((thumb_w, CONTACT_THUMB_HEIGHT), Image.LANCZOS))

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

    trim = 0.0 if args.format == "120" else GEOMETRIC_TRIM_FRACTION
    row_means = compute_row_means(gray, trim_fraction=trim)

    threshold = (float(args.threshold) if args.threshold is not None
                 else auto_threshold(row_means))
    result.threshold_used = threshold

    if args.verbose:
        src = "user" if args.threshold is not None else "auto/Otsu"
        print(f"  [DEBUG] Row means: min={row_means.min():.1f}, "
              f"max={row_means.max():.1f}, mean={row_means.mean():.1f}")
        print(f"  [DEBUG] Threshold: {threshold:.1f} ({src})")

    gaps = detect_gaps(row_means, threshold, args.min_gap)

    if args.verbose:
        print(f"  [DEBUG] Gaps: {len(gaps)}")
        for i, g in enumerate(gaps):
            print(f"  [DEBUG]   gap {i}: y=[{g.y_start}:{g.y_end}] h={g.height}px")

    frames = gaps_to_frames(gaps, H, args.min_frame)

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

    no_crop = args.no_crop_sprockets or (args.format == "120")
    crop_box = detect_sprocket_crop(gray, gaps, no_crop, args.verbose)
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
        pil_image = pil_image.transpose(Image.TRANSPOSE)

    export_frames(pil_image, frames, crop_box, output_dir, filepath.stem, args.verbose,
                  frame_trim=args.frame_trim)
    generate_contact_sheet(pil_image, frames, crop_box, output_dir, filepath.stem,
                           frame_trim=args.frame_trim)
    return result


# ---------------------------------------------------------------------------
# Summary and main
# ---------------------------------------------------------------------------

def print_file_summary(result: ProcessingResult, verbose: bool) -> None:
    name = result.filepath.name

    if result.n_frames == 0:
        for w in result.warnings:
            print(f"{name} → WARNING: {w}")
        return

    cb = result.crop_box
    crop_str = f", crop={cb.method} x[{cb.x_left}:{cb.x_right}]" if cb else ""
    print(f"{name} → {result.n_frames} frame(s)  "
          f"[threshold={result.threshold_used:.1f}{crop_str}]")

    if verbose:
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
        print_file_summary(result, args.verbose)

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
