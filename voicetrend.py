#!/usr/bin/env python3
"""
voicetrend.py - track voice training metrics across many recordings

Analyses every audio file in a folder, sorts by date, and plots how
each metric has changed. Results are cached, so re-running only
analyses files it hasn't seen before.

Usage:
    python voicetrend.py                    # current folder
    python voicetrend.py ~/recordings
    python voicetrend.py --csv out.csv      # also dump a spreadsheet
    python voicetrend.py --rescan           # ignore cache, redo everything
    python voicetrend.py --date-from-name   # parse dates out of filenames

Filename dates: any file containing a date like 23-08-26, 2026-08-23,
20260823 or 26-08-23 is dated from that when --date-from-name is set.
Otherwise file modification time is used.

Requires: ffmpeg on PATH, and:
    pip install praat-parselmouth numpy matplotlib
"""

import argparse
import csv as csvmod
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    sys.exit("Missing dependency. Run: pip install praat-parselmouth numpy matplotlib")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    sys.exit("Missing dependency. Run: pip install matplotlib")


AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}
CACHE_NAME = ".voicetrend_cache.json"

# (label, unit, target_low, target_high, higher_is_better)
METRICS = [
    ("f0_median", "F0 median",  "Hz", 180,  220,  None),
    ("f0_p10",    "F0 floor",   "Hz", 150,  None, True),
    ("creak",     "Creak",      "%",  None, 5,    False),
    ("f3",        "F3",         "Hz", 2900, None, True),
    ("hnr",       "HNR",        "dB", 15,   None, True),
    ("cpps",      "CPPS",       "dB", None, None, True),
]

DATE_PATTERNS = [
    (r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})", "ymd"),   # 2026-08-23 / 20260823
    (r"(\d{2})[-_.](\d{2})[-_.](\d{2})",     "dmy"),   # 23-08-26
]


def parse_date_from_name(name):
    """Best-effort date extraction from a filename. Returns datetime or None."""
    for pattern, order in DATE_PATTERNS:
        m = re.search(pattern, name)
        if not m:
            continue
        a, b, c = (int(x) for x in m.groups())
        try:
            if order == "ymd":
                return datetime(a, b, c)
            # dmy with 2-digit year: 23-08-26 -> 2026-08-23
            return datetime(2000 + c, b, a)
        except ValueError:
            continue
    return None


def to_wav(path):
    if path.lower().endswith(".wav"):
        return path, False
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    r = subprocess.run(
        ["ffmpeg", "-i", path, "-ac", "1", "-ar", "44100", out, "-y", "-loglevel", "error"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        os.unlink(out)
        return None, False
    return out, True


def analyse(path, floor=60, ceiling=600, creak_below=100, max_formant=5500):
    """Return a dict of metrics for one file, or None if it can't be read."""
    wav, temporary = to_wav(path)
    if wav is None:
        return None
    try:
        snd = parselmouth.Sound(wav)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
        f0 = pitch.selected_array["frequency"]
        fv = f0[f0 > 0]
        if len(fv) < 50:
            return None

        # Guard against octave-doubling artifacts skewing the ceiling.
        p90_raw = float(np.percentile(fv, 90))
        clean = fv[fv < 400] if (fv < 400).sum() > 0.8 * len(fv) else fv

        res = {
            "duration":   float(snd.duration),
            "f0_median":  float(np.median(fv)),
            "f0_p10":     float(np.percentile(fv, 10)),
            "f0_p90":     float(np.percentile(clean, 90)),
            "creak":      float(100 * (fv < creak_below).sum() / len(fv)),
            "voiced_pct": float(100 * len(fv) / len(f0)),
        }

        try:
            h = call(snd, "To Harmonicity (cc)", 0.01, floor, 0.1, 1.0)
            hv = h.values[0]
            hv = hv[hv > -100]
            res["hnr"] = float(np.mean(hv)) if len(hv) else None
        except Exception:
            res["hnr"] = None

        try:
            pc = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
            res["cpps"] = float(call(pc, "Get CPPS", False, 0.02, 0.0005, 60, 330,
                                     0.05, "Parabolic", 0.001, 0.05,
                                     "Straight", "Robust"))
        except Exception:
            res["cpps"] = None

        try:
            fm = call(snd, "To Formant (burg)", 0.01, 5, max_formant, 0.025, 50)
            vals = []
            times = pitch.xs()
            for t, f in zip(times, f0):
                if f <= 0:
                    continue
                v = call(fm, "Get value at time", 3, t, "hertz", "linear")
                if not np.isnan(v):
                    vals.append(v)
            res["f3"] = float(np.median(vals)) if vals else None
        except Exception:
            res["f3"] = None

        return res
    except Exception:
        return None
    finally:
        if temporary and os.path.exists(wav):
            os.unlink(wav)


def load_cache(folder):
    path = os.path.join(folder, CACHE_NAME)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def save_cache(folder, cache):
    try:
        with open(os.path.join(folder, CACHE_NAME), "w") as fh:
            json.dump(cache, fh, indent=1)
    except Exception:
        pass


def trend_line(dates, values):
    """Least-squares slope per day. Returns (slope, first_fit, last_fit)."""
    if len(values) < 3:
        return None, None, None
    x = np.array([(d - dates[0]).total_seconds() / 86400 for d in dates])
    y = np.array(values)
    ok = ~np.isnan(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return None, None, None
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return slope, intercept + slope * x[ok][0], intercept + slope * x[ok][-1]


def main():
    ap = argparse.ArgumentParser(
        description="Track voice metrics across a folder of recordings.")
    ap.add_argument("folder", nargs="?", default=".", help="Folder of audio files")
    ap.add_argument("--out", default="voice_progress.png", help="Output image")
    ap.add_argument("--csv", help="Also write a CSV of every measurement")
    ap.add_argument("--rescan", action="store_true", help="Ignore the cache")
    ap.add_argument("--date-from-name", action="store_true",
                    help="Take dates from filenames where possible")
    ap.add_argument("--min-duration", type=float, default=3.0,
                    help="Skip clips shorter than this (default 3 s)")
    ap.add_argument("--max-formant", type=float, default=5500)
    ap.add_argument("--creak-below", type=float, default=100)
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        sys.exit(f"Not a folder: {folder}")

    files = sorted(f for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"No audio files found in {folder}")

    cache = {} if args.rescan else load_cache(folder)
    rows = []

    print(f"Scanning {len(files)} files in {folder}\n")
    for name in files:
        path = os.path.join(folder, name)
        stat = os.stat(path)
        key = f"{name}:{stat.st_size}:{int(stat.st_mtime)}"

        if key in cache:
            res = cache[key]
            status = "cached"
        else:
            res = analyse(path, creak_below=args.creak_below,
                          max_formant=args.max_formant)
            cache[key] = res
            status = "analysed"

        if res is None:
            print(f"  {name:<34} skipped (unreadable)")
            continue
        if res["duration"] < args.min_duration:
            print(f"  {name:<34} skipped (too short)")
            continue

        when = None
        if args.date_from_name:
            when = parse_date_from_name(name)
        if when is None:
            when = datetime.fromtimestamp(stat.st_mtime)

        row = dict(res)
        row["file"] = name
        row["date"] = when
        rows.append(row)
        print(f"  {name:<34} {status:<9} {when:%Y-%m-%d}  "
              f"F0 {res['f0_median']:.0f}  creak {res['creak']:.1f}%")

    save_cache(folder, cache)

    if len(rows) < 2:
        sys.exit("\nNeed at least 2 usable recordings to show a trend.")

    rows.sort(key=lambda r: r["date"])
    dates = [r["date"] for r in rows]

    # ---------------- summary ----------------
    print(f"\n{'='*66}")
    print(f"TREND  ({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}, {len(rows)} recordings)")
    print(f"{'='*66}")
    print(f"{'metric':<12}{'first':>9}{'last':>9}{'change':>10}{'per week':>11}  verdict")

    n = max(1, min(3, len(rows) // 3))
    for key, label, unit, lo, hi, higher_better in METRICS:
        vals = [r.get(key) for r in rows]
        vals = [np.nan if v is None else v for v in vals]
        if np.all(np.isnan(vals)):
            continue
        first = np.nanmean(vals[:n])
        last = np.nanmean(vals[-n:])
        change = last - first
        slope, _, _ = trend_line(dates, vals)
        per_week = slope * 7 if slope is not None else float("nan")

        if higher_better is None:              # band target
            def dist(v):
                return 0 if lo <= v <= hi else min(abs(v - lo), abs(v - hi))
            verdict = "closer to target" if dist(last) < dist(first) else \
                      ("further away" if dist(last) > dist(first) else "flat")
        elif higher_better:
            verdict = "improving" if change > 0 else ("worse" if change < 0 else "flat")
        else:
            verdict = "improving" if change < 0 else ("worse" if change > 0 else "flat")

        if abs(change) < 0.05 * max(1.0, abs(first)):
            verdict = "roughly flat"

        print(f"{label:<12}{first:>9.1f}{last:>9.1f}{change:>+10.1f}"
              f"{per_week:>+11.1f}  {verdict}")

    # ---------------- plot ----------------
    plot_metrics = [m for m in METRICS
                    if not np.all(np.isnan([np.nan if r.get(m[0]) is None
                                            else r[m[0]] for r in rows]))]
    ncols = 2
    nrows = int(np.ceil(len(plot_metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), squeeze=False)
    fig.suptitle(f"Voice training progress   {dates[0]:%d %b} - {dates[-1]:%d %b %Y}"
                 f"   ({len(rows)} recordings)", fontsize=13, y=0.995)

    for idx, (key, label, unit, lo, hi, higher_better) in enumerate(plot_metrics):
        ax = axes[idx // ncols][idx % ncols]
        vals = np.array([np.nan if r.get(key) is None else r[key] for r in rows],
                        dtype=float)

        # target band shading
        ylo, yhi = np.nanmin(vals), np.nanmax(vals)
        pad = max(1.0, (yhi - ylo) * 0.18)
        if lo is not None and hi is not None:
            ax.axhspan(lo, hi, color="tab:green", alpha=0.10, zorder=0)
            ax.axhline(lo, color="tab:green", lw=0.8, ls="--", alpha=0.6)
            ax.axhline(hi, color="tab:green", lw=0.8, ls="--", alpha=0.6)
        elif lo is not None:
            ax.axhspan(lo, max(yhi + pad, lo + pad), color="tab:green",
                       alpha=0.10, zorder=0)
            ax.axhline(lo, color="tab:green", lw=0.9, ls="--", alpha=0.7)
        elif hi is not None:
            ax.axhspan(min(ylo - pad, hi - pad), hi, color="tab:green",
                       alpha=0.10, zorder=0)
            ax.axhline(hi, color="tab:green", lw=0.9, ls="--", alpha=0.7)

        ax.plot(dates, vals, "o-", color="tab:blue", ms=4, lw=1.1,
                alpha=0.75, zorder=3)

        # rolling mean (window 3) to show the underlying direction
        if len(vals) >= 5:
            ser = np.convolve(np.nan_to_num(vals, nan=np.nanmean(vals)),
                              np.ones(3) / 3, mode="valid")
            ax.plot(dates[1:-1], ser, "-", color="tab:orange", lw=2.0,
                    alpha=0.85, zorder=4, label="3-point average")

        slope, y0, y1 = trend_line(dates, vals)
        if slope is not None:
            ax.plot([dates[0], dates[-1]], [y0, y1], "--", color="crimson",
                    lw=1.4, zorder=5,
                    label=f"trend {slope*7:+.1f} {unit}/week")

        ax.set_title(f"{label} ({unit})", fontsize=10, loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(ylo - pad, yhi + pad)
        if slope is not None or len(vals) >= 5:
            ax.legend(fontsize=7, loc="best", framealpha=0.85)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_horizontalalignment("right")
            lbl.set_fontsize(8)

    for j in range(len(plot_metrics), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=140)
    print(f"\nGraph written to {args.out}")

    if args.csv:
        cols = ["file", "date", "duration", "f0_median", "f0_p10", "f0_p90",
                "creak", "hnr", "cpps", "f3", "voiced_pct"]
        with open(args.csv, "w", newline="") as fh:
            w = csvmod.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                out = dict(r)
                out["date"] = r["date"].strftime("%Y-%m-%d %H:%M")
                w.writerow(out)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
