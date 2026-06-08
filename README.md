# slice_strips.py

Detects inter-frame gaps in scanned film strips and exports each frame as an
individual LZW-compressed TIFF file, plus a JPEG contact sheet.
Designed for use with **film holders** — sprocket holes are assumed to be
covered and are never used for analysis.

---

## Requirements

- Python 3.9 or later
- `numpy` and `Pillow`

Install dependencies:
```bash
pip install numpy Pillow
# or, if you use uv:
uv run slice_strips.py scans/
```

---

## Quick start

```bash
# Preview what would be cut — no files written
python3 slice_strips.py scans/ --dry-run -v

# Export all frames
python3 slice_strips.py scans/

# Dark rolls that need a higher threshold
python3 slice_strips.py scans/ --threshold 36 --dry-run -v
python3 slice_strips.py scans/ --threshold 36
```

---

## Parameters

### Input / output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_folder` | *(required)* | Path to the folder containing scanned TIFF (or other) files. |
| `--output PATH` | `./output/` | Where to write the exported frames and contact sheets. Created automatically if it does not exist. |
| `--extensions EXTS` | `tif,tiff,jpg,jpeg,png` | Comma-separated list of file extensions to process. Change if your scanner saves in a different format. |

---

### Gap detection (frame-splitting)

These control how the script finds the dark bands between frames.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold INT` | auto (~12) | Brightness value (0–255) below which a row is classified as an inter-frame gap. The default is auto-detected per file using Otsu's method, capped at 12. **Raise to 36** when the script warns that too few frames were detected — this happens for rolls where the film base between frames is a medium dark rather than near-black. |
| `--min-gap INT` | `10` | Minimum height in pixels for a dark band to count as a real gap. Raise (e.g. to `50`) if scanner noise is creating many spurious tiny gaps. |
| `--max-gap INT` | disabled | Maximum height in pixels for a gap. Bands taller than this are discarded as false gaps caused by very dark photographic content. **Recommended value when using `--threshold 36`: `750`.** Do not combine with the default threshold — it is not needed there. |

---

### Frame filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-frame INT` | `1800` | Minimum frame height in pixels. Regions shorter than this (partial leading/trailing fragments) are silently discarded. Raise to a larger value if you want to keep only full-sized frames; lower it (e.g. `600`) if partial frames at the roll edges are being dropped. |
| `--frame-trim INT` | `0` | Pixels to remove from the **top and bottom** of every exported frame. Use this to strip the film rebate (the thin dark border around each image area). |

---

### Holder trimming

Film holders produce a uniformly bright region at both ends of the scan (the
empty holder backing beyond the film). The script automatically detects and
removes this area.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-holder-trim` | off | Disables the automatic holder-edge trimming. Use only if you want the blank holder area included in the exported frames. |

---

### Workflow helpers

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dry-run` | off | Prints the detected frames and their pixel coordinates without writing any files. **Always run this first** to check the result before a full export. |
| `-v` / `--verbose` | off | Prints per-gap and per-frame debug numbers (threshold used, gap heights and `dark_frac` range, frame heights). Most useful combined with `--dry-run` to diagnose a problem file. The `dark_frac` value shows what fraction of pixels in a gap are consistently dark across the full width — values near 1.00 are solid film base; values near 0.80 indicate a narrower or slightly uneven gap. |

---

## Typical workflows

### Standard roll (most files)
```bash
# 1. Check
python3 slice_strips.py scans/ --dry-run -v

# 2. Export
python3 slice_strips.py scans/
```

### Dark roll (e.g. roll 8 in this project)
The inter-frame borders are medium-dark instead of near-black.
The script will print a `[WARN]` message suggesting `--threshold 36`.
```bash
# 1. Check
python3 slice_strips.py scans/ --threshold 36 --dry-run -v

# 2. Export
python3 slice_strips.py scans/ --threshold 36
```

### Dark roll with over-detection
If `--threshold 36` still produces too many frames (large dark photographic
content is being mistaken for gaps), add `--max-gap 750` to discard
oversized false gaps:
```bash
python3 slice_strips.py scans/ --threshold 36 --max-gap 750 --dry-run -v
python3 slice_strips.py scans/ --threshold 36 --max-gap 750
```

### Diagnosing a problem file
```bash
python3 slice_strips.py scans/ --dry-run -v 2>&1 | grep -E "Processing|gap|frame|WARN"
```

---

## Output structure

```
output/
  <filename>_01.tiff
  <filename>_02.tiff
  ...
  contacts/
    <filename>_contact.jpg   ← side-by-side JPEG overview of all frames
```

Frames are exported as LZW-compressed TIFFs at the original bit depth and
colour mode of the source scan. Contact sheets are 8-bit JPEG at 85% quality,
with each frame thumbnail scaled to 300 px tall.

---

## Warnings reference

| Warning | Meaning | Fix |
|---------|---------|-----|
| `Only N frame(s) detected in a Xpx tall strip … Try --threshold 36` | The inter-frame gaps are lighter than the default cap of 12 and are being missed. | Re-run with `--threshold 36`. If over-detection then occurs, add `--max-gap 750`. |
| `No frames detected … Try a lower --threshold or --min-frame value` | No rows fell below the threshold at all. | Lower `--threshold` or reduce `--min-frame`. |
