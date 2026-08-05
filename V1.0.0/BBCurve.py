"""
Backbone (envelope) curve extractor + bilinear idealization for cyclic
hysteresis data -- GUI VERSION. Units: displacement in mm, force in kN.

This GUI wraps the ORIGINAL processing logic (backbone extraction,
positive/negative/average branches, EEEP bilinear idealization, energy,
stiffness, ductility, summary CSV).

Requires: numpy, pandas, matplotlib  (tkinter ships with standard Python)

--------------------------------------------------------------------------
Developed by: Tufail Mabood
GitHub:       https://github.com/tufailmab
WhatsApp:     +92 344 0907874  /  +92 340 0740460
--------------------------------------------------------------------------
"""

import csv
import glob
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# numpy >= 2.0 renamed trapz -> trapezoid; support both so this runs anywhere
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Output Folders Names

BACKBONE_PLOTS_DIR = "All Backbone Curves-Plots"
BACKBONE_CSVS_DIR = "All Backbone Curves-CSVs"
PNA_PLOTS_DIR = "Positive-Negative-Average Plots"
PNA_CSVS_DIR = "Positive-Negative-Average CSVs"
BILINEAR_PLOTS_DIR = "Bilinear Idealization Plots"
BILINEAR_CSVS_DIR = "Bilinear Idealization CSVs"
STIFFNESS_PLOTS_DIR = "Stiffness Degradation Plots"
STIFFNESS_CSVS_DIR = "Stiffness Degradation CSVs"
ENERGY_PLOTS_DIR = "Energy Dissipation Plots"
ENERGY_CSVS_DIR = "Energy Dissipation CSVs"
SUMMARY_DIR = "Summary"

# Engineering assumptions (standard, widely-used defaults -- change here if needed)
STIFFNESS_FRACTION = 0.40    # Ke = secant stiffness through 40% of peak force (ASTM E2126)
DEGRADATION_FRACTION = 0.80  # Ultimate point = first drop to 80% of peak force after the peak

PLOT_TYPES = [
    "Backbone Curve",
    "Positive / Negative / Average",
    "Bilinear Idealization",
    "Stiffness Degradation (per Cycle)",
    "Energy Dissipation per Loop",
    "Cumulative Energy Dissipation",
]


# App Branding & Developer Info

APP_NAME = "Backbone Curve & Bilinear Idealization Tool"
APP_SUBTITLE = "Cyclic Hysteresis Analysis Suite  \u2022  ASTM E2126 EEEP Method"
APP_VERSION = "1.0.0"

DEVELOPER_NAME = "Tufail Mabood"
DEVELOPER_GITHUB_URL = "https://github.com/tufailmab"
DEVELOPER_GITHUB_LABEL = "GitHub.com/tufailmab"
DEVELOPER_WHATSAPP_1 = "+92 344 0907874"
DEVELOPER_WHATSAPP_2 = "+92 340 0740460"

# Color Pallete

COLOR_BG = "#f2f4f7"
COLOR_HEADER_BG = "#12233d"
COLOR_HEADER_ACCENT = "#2f8fd6"
COLOR_HEADER_TEXT = "#ffffff"
COLOR_HEADER_SUBTEXT = "#b7c6da"
COLOR_FOOTER_BG = "#12233d"
COLOR_FOOTER_TEXT = "#9db2cc"
COLOR_ACCENT = "#2f8fd6"
COLOR_ACCENT_DARK = "#1f6ea8"
COLOR_LOG_BG = "#0c1220"
COLOR_LOG_DEFAULT = "#d6e2f0"
COLOR_LOG_OK = "#3ddc84"
COLOR_LOG_FAIL = "#ff6b6b"
COLOR_LOG_HEADER = "#5ac8fa"
COLOR_STATUS_BG = "#e7ecf3"
COLOR_PANEL_BG = "#ffffff"
COLOR_TOOLBAR_BG = "#e3e9f1"
COLOR_MAX_HL = "#c8f7d4"
COLOR_MIN_HL = "#ffd6d6"

FONT_FAMILY = "Segoe UI"

# Setting Persistence

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".bbcurve_gui_settings.json")

DEFAULT_SETTINGS = {
    "last_input_dir": "",
    "last_output_dir": "",
    "recent_input_dirs": [],
    "recent_output_dirs": [],
    "window_geometry": "1440x900+80+40",
    "plot_style": "default",
    "line_width": 2.2,
    "marker_size": 5.0,
    "show_grid": True,
    "show_legend": True,
    "show_markers": True,
    "crosshair_enabled": False,
    "export_dpi": 300,
    "max_recent": 6,
}

MPL_STYLES = ["default", "classic", "bmh", "ggplot", "fivethirtyeight", "Solarize_light2"]


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        if os.path.isfile(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                settings.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
    except Exception:
        pass  # fall back silently to defaults; never crash the app on bad settings
    return settings


def save_settings(settings):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass  # persistence is a convenience, never fatal


# Logic

def find_txt_files(input_dir):
    candidates = sorted(f for f in glob.glob(os.path.join(input_dir, "*.txt")))
    return candidates


def load_hysteresis_data(txt_path):
    # Skip the first row (header) and use tab as delimiter
    df = pd.read_csv(txt_path, header=0, names=["disp", "force"], skiprows=1, delimiter='\t')

    # If the above fails, try alternative parsing with whitespace delimiter
    if len(df.columns) != 2 or df.iloc[0].isnull().any():
        # Try reading with any whitespace as delimiter
        df = pd.read_csv(txt_path, header=0, names=["disp", "force"], skiprows=1, delim_whitespace=True)

    df["disp"] = pd.to_numeric(df["disp"], errors="coerce")
    df["force"] = pd.to_numeric(df["force"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if len(df) < 3:
        raise ValueError("TXT does not contain enough valid numeric rows.")

    return df["disp"].to_numpy(), df["force"].to_numpy()


def find_reversal_points(disp, force):
    d = np.diff(disp)
    d = np.where(d == 0, np.nan, d)
    direction = np.sign(d)
    direction = pd.Series(direction).ffill().bfill().to_numpy()

    turn = np.ones(len(disp), dtype=bool)
    turn[1:-1] = direction[1:] != direction[:-1]
    turn[0] = True
    turn[-1] = True

    return disp[turn], force[turn]


def monotonic_envelope(x, y, cluster_tol=0.02):
    if len(x) == 0:
        return np.array([]), np.array([])

    order = np.argsort(np.abs(x))
    x, y = x[order], np.abs(y[order])

    tol = cluster_tol * np.max(np.abs(x))

    groups_x, groups_y = [], []
    cur_x, cur_y = [x[0]], [y[0]]
    for xi, yi in zip(x[1:], y[1:]):
        if xi - cur_x[-1] <= tol:
            cur_x.append(xi)
            cur_y.append(yi)
        else:
            groups_x.append(np.mean(cur_x))
            groups_y.append(np.max(cur_y))
            cur_x, cur_y = [xi], [yi]
    groups_x.append(np.mean(cur_x))
    groups_y.append(np.max(cur_y))

    keep_x, keep_y = [], []
    max_x_seen = -np.inf
    for xi, yi in zip(groups_x, groups_y):
        if xi > max_x_seen:
            keep_x.append(xi)
            keep_y.append(yi)
            max_x_seen = xi

    return np.array(keep_x), np.array(keep_y)


def split_backbone_branches(disp, force):
    """Return the positive branch and negative branch of the backbone
    curve, each INCLUDING the origin (0,0) and sorted ascending in x.
    Signs are kept as-is (negative branch stays negative/negative).
    """
    px, pf = find_reversal_points(disp, force)

    pos_mask = px > 0
    neg_mask = px < 0

    pos_x, pos_y = monotonic_envelope(px[pos_mask], pf[pos_mask])
    neg_x, neg_y = monotonic_envelope(-px[neg_mask], -pf[neg_mask])
    neg_x, neg_y = -neg_x, -neg_y  # restore sign; currently descending (near-0 -> most negative)

    pos_x_full = np.concatenate([[0.0], pos_x])
    pos_y_full = np.concatenate([[0.0], pos_y])
    neg_x_full = np.concatenate([neg_x[::-1], [0.0]])   # ascending: most negative -> 0
    neg_y_full = np.concatenate([neg_y[::-1], [0.0]])

    return pos_x_full, pos_y_full, neg_x_full, neg_y_full


def build_backbone(disp, force):
    pos_x, pos_y, neg_x, neg_y = split_backbone_branches(disp, force)
    bx = np.concatenate([neg_x, pos_x[1:]])
    by = np.concatenate([neg_y, pos_y[1:]])
    return bx, by, pos_x, pos_y, neg_x, neg_y


def mirror_negative_branch(neg_x, neg_y):
    """Reflect the negative branch (which lives in the (-,-) quadrant)
    onto the positive quadrant so it can be compared directly against
    the positive branch. neg_x/neg_y are ascending from (-max, -Fmax) to
    (0, 0); the mirrored output is ascending from (0,0) to (max, Fmax).
    """
    mirror_x = -neg_x[::-1]
    mirror_y = -neg_y[::-1]
    return mirror_x, mirror_y


def average_branch(pos_x, pos_y, mirror_x, mirror_y):
    """Average of the positive branch and the mirrored negative branch,
    interpolated onto a common displacement grid (union of both branches'
    displacement values). Where one branch doesn't have data (extends
    further than the other), that branch's own last value is held
    constant (standard flat extrapolation) rather than crashing.
    """
    grid = np.unique(np.concatenate([pos_x, mirror_x]))
    pos_interp = np.interp(grid, pos_x, pos_y)
    neg_interp = np.interp(grid, mirror_x, mirror_y)
    avg_y = (pos_interp + neg_interp) / 2.0
    return grid, avg_y, pos_interp, neg_interp


def compute_branch_properties(x, y,
                               stiffness_frac=STIFFNESS_FRACTION,
                               degrade_frac=DEGRADATION_FRACTION):
    """Compute initial stiffness, ultimate point, energy (area under
    curve), and the EEEP bilinear idealization (Dy, Fy) for ONE branch
    (x, y must be ascending in x, starting at/near the origin, with y
    generally rising to a peak and then possibly degrading).

    Returns a dict of results, plus 'Notes' if anything had to fall back
    to a default assumption.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    notes = []

    if len(x) < 2 or np.max(y) <= 0:
        return {"error": "Not enough data / no positive force reached."}

    idx_max = int(np.argmax(y))
    Fmax = y[idx_max]
    Dmax = x[idx_max]

    # --- Ultimate point: first post-peak drop to degrade_frac * Fmax ---
    Du, Fu = x[-1], y[-1]
    found = False
    for i in range(idx_max, len(x) - 1):
        if y[i] >= degrade_frac * Fmax and y[i + 1] < degrade_frac * Fmax:
            t = (degrade_frac * Fmax - y[i]) / (y[i + 1] - y[i])
            Du = x[i] + t * (x[i + 1] - x[i])
            Fu = degrade_frac * Fmax
            found = True
            break
    if not found:
        notes.append(
            f"No {int(degrade_frac*100)}% post-peak strength drop found in data; "
            "ultimate point taken as the last available data point."
        )

    # --- Initial stiffness: secant through stiffness_frac * Fmax ---
    asc_x, asc_y = x[:idx_max + 1], y[:idx_max + 1]
    target = stiffness_frac * Fmax
    Ke = None
    for i in range(len(asc_x) - 1):
        if asc_y[i] <= target <= asc_y[i + 1]:
            if asc_y[i + 1] != asc_y[i]:
                t = (target - asc_y[i]) / (asc_y[i + 1] - asc_y[i])
                Dtarget = asc_x[i] + t * (asc_x[i + 1] - asc_x[i])
            else:
                Dtarget = asc_x[i]
            if Dtarget > 0:
                Ke = target / Dtarget
            break
    if Ke is None or not np.isfinite(Ke):
        Ke = (asc_y[1] - asc_y[0]) / (asc_x[1] - asc_x[0]) if asc_x[1] != asc_x[0] else np.nan
        notes.append(f"Could not find {int(stiffness_frac*100)}% point cleanly; "
                      "Ke estimated from first available segment instead.")

    # --- Energy = area under actual curve from 0 to Du ---
    mask = x <= Du
    cx, cy = list(x[mask]), list(y[mask])
    if len(cx) == 0 or cx[-1] < Du:
        cx.append(Du)
        cy.append(Fu)
    Area = float(_trapezoid(cy, cx))

    # --- EEEP bilinear idealization (equal energy, ASTM E2126) ---
    Dy = Fy = ductility = np.nan
    disc = (Ke * Du) ** 2 - 2 * Ke * Area
    if Ke and np.isfinite(Ke) and Ke > 0 and disc >= 0:
        Fy = Ke * Du - np.sqrt(disc)
        Dy = Fy / Ke
        ductility = Du / Dy if Dy > 0 else np.nan
    else:
        notes.append("EEEP bilinear solution not well-defined for this curve "
                      "(check data quality / stiffness estimate).")

    return {
        "Fmax_kN": Fmax, "Dmax_mm": Dmax,
        "Du_mm": Du, "Fu_kN": Fu,
        "Ke_kN_per_mm": Ke,
        "Energy_kN_mm": Area,
        "Dy_mm": Dy, "Fy_kN": Fy,
        "Ductility": ductility,
        "Notes": "; ".join(notes) if notes else "",
    }


def bilinear_curve_points(props):
    """(0,0) -> (Dy,Fy) -> (Du,Fy) -- the EEEP idealized elastic-perfectly
    -plastic curve. Returns None if the yield point could not be solved.
    """
    if not np.isfinite(props.get("Dy_mm", np.nan)):
        return None
    return (
        np.array([0.0, props["Dy_mm"], props["Du_mm"]]),
        np.array([0.0, props["Fy_kN"], props["Fy_kN"]]),
    )


def identify_cycles(disp, force):
    """Segment the RAW, time-ordered hysteresis data into individual
    loading cycles/loops using the reversal (turning) points, in the
    original chronological order (unlike monotonic_envelope, which
    re-sorts by |displacement| to build the envelope -- here we keep
    the true time sequence so each loop is a real physical loop).

    A "cycle" is defined the standard way: from one positive-going
    displacement peak to the next positive-going displacement peak
    (i.e. it contains exactly one positive excursion and one negative
    excursion -- one full loop of the hysteresis).

    Returns a list of dicts (one per detected cycle, in order), each
    holding the loop's raw disp/force sub-arrays plus its positive and
    negative peak (displacement, force) pairs. Returns [] if fewer
    than two positive peaks are found (not enough data to form a
    full loop).
    """
    disp = np.asarray(disp, dtype=float)
    force = np.asarray(force, dtype=float)

    d = np.diff(disp)
    d = np.where(d == 0, np.nan, d)
    direction = np.sign(d)
    direction = pd.Series(direction).ffill().bfill().to_numpy()

    turn = np.ones(len(disp), dtype=bool)
    turn[1:-1] = direction[1:] != direction[:-1]
    turn[0] = True
    turn[-1] = True
    turn_idx = np.where(turn)[0]

    pos_peak_idx = [k for k in turn_idx if disp[k] > 0]

    cycles = []
    for c, (i0, i1) in enumerate(zip(pos_peak_idx[:-1], pos_peak_idx[1:]), start=1):
        seg_d = disp[i0:i1 + 1]
        seg_f = force[i0:i1 + 1]
        if len(seg_d) < 3:
            continue
        idx_pos = int(np.argmax(seg_f))
        idx_neg = int(np.argmin(seg_f))
        cycles.append({
            "cycle": c,
            "disp": seg_d, "force": seg_f,
            "peak_pos_disp": float(seg_d[idx_pos]), "peak_pos_force": float(seg_f[idx_pos]),
            "peak_neg_disp": float(seg_d[idx_neg]), "peak_neg_force": float(seg_f[idx_neg]),
        })
    return cycles


def compute_cycle_metrics(cycles):
    """From the list of per-cycle loops returned by identify_cycles(),
    compute, for every cycle:
        - Secant stiffness through the two peaks of the loop
          (Fpos - Fneg) / (Dpos - Dneg)
        - Stiffness retained relative to the first cycle (%)
        - Dissipated energy = area enclosed by the closed loop
          (shoelace formula on the loop's disp/force path)
        - Cumulative dissipated energy up to and including that cycle

    Returns None if `cycles` is empty, otherwise a dict of numpy
    arrays keyed by: cycle_number, stiffness_kN_per_mm,
    stiffness_pct_of_initial, energy_kN_mm, cumulative_energy_kN_mm.
    """
    if not cycles:
        return None

    cycle_number = np.array([c["cycle"] for c in cycles], dtype=int)

    stiffness = np.array([
        (c["peak_pos_force"] - c["peak_neg_force"]) / (c["peak_pos_disp"] - c["peak_neg_disp"])
        if (c["peak_pos_disp"] - c["peak_neg_disp"]) != 0 else np.nan
        for c in cycles
    ], dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        stiffness_pct = 100.0 * stiffness / stiffness[0] if len(stiffness) else np.array([])

    energy = []
    for c in cycles:
        x, y = c["disp"], c["force"]
        # Shoelace formula for the area enclosed by the closed loop
        # (loop already starts and ends at the same positive peak).
        xc = np.concatenate([x, x[:1]])
        yc = np.concatenate([y, y[:1]])
        area = 0.5 * np.abs(np.sum(xc[:-1] * yc[1:] - xc[1:] * yc[:-1]))
        energy.append(float(area))
    energy = np.array(energy, dtype=float)
    cumulative_energy = np.cumsum(energy)

    return {
        "cycle_number": cycle_number,
        "stiffness_kN_per_mm": stiffness,
        "stiffness_pct_of_initial": stiffness_pct,
        "energy_kN_mm": energy,
        "cumulative_energy_kN_mm": cumulative_energy,
    }


def process_one_file(txt_path, dirs):
    """Processes a single TXT: saves the same plots/CSVs as the original
    script into `dirs`, AND returns a dict with every array needed to
    re-draw the three plot types later inside the GUI (so we don't need
    to re-read PNGs -- we redraw live from data with matplotlib).
    """
    stem = os.path.splitext(os.path.basename(txt_path))[0]

    disp, force = load_hysteresis_data(txt_path)
    bx, by, pos_x, pos_y, neg_x, neg_y = build_backbone(disp, force)

    # ---- (A) raw + full backbone plot/csv ----
    pd.DataFrame({"displacement_mm": bx, "force_kN": by}).to_csv(
        os.path.join(dirs["backbone_csv"], f"{stem}_backbone.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(disp, force, color="lightgray", linewidth=0.8, label="Raw hysteresis data")
    ax.plot(bx, by, color="crimson", linewidth=2.2, marker="o", markersize=4, label="Backbone curve")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.scatter([0], [0], color="black", zorder=5, s=25)
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (kN)")
    ax.set_title(f"Backbone Curve: {stem}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["backbone_plot"], f"{stem}_backbone.png"), dpi=200)
    plt.close(fig)

    # ---- (B) positive / mirrored-negative / average ----
    mirror_x, mirror_y = mirror_negative_branch(neg_x, neg_y)
    grid, avg_y, pos_on_grid, neg_on_grid = average_branch(pos_x, pos_y, mirror_x, mirror_y)

    pd.DataFrame({
        "displacement_mm": grid,
        "positive_force_kN": pos_on_grid,
        "negative_force_mirrored_kN": neg_on_grid,
        "average_force_kN": avg_y,
    }).to_csv(os.path.join(dirs["pna_csv"], f"{stem}_pos_neg_avg.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pos_x, pos_y, color="royalblue", linewidth=2, marker="o", markersize=4, label="Positive branch")
    ax.plot(mirror_x, mirror_y, color="darkorange", linewidth=2, marker="o", markersize=4,
             label="Negative branch (mirrored)")
    ax.plot(grid, avg_y, color="black", linewidth=2.4, linestyle="--", label="Average branch")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (kN)")
    ax.set_title(f"Positive / Negative(mirrored) / Average: {stem}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["pna_plot"], f"{stem}_pos_neg_avg.png"), dpi=200)
    plt.close(fig)

    # ---- (C) engineering properties + bilinear idealization ----
    props_pos = compute_branch_properties(pos_x, pos_y)
    props_neg = compute_branch_properties(mirror_x, mirror_y)
    props_avg = compute_branch_properties(grid, avg_y)

    bl = bilinear_curve_points(props_avg)
    if bl is not None:
        bl_x, bl_y = bl
        pd.DataFrame({"displacement_mm": bl_x, "force_kN": bl_y}).to_csv(
            os.path.join(dirs["bilinear_csv"], f"{stem}_bilinear.csv"), index=False)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(grid, avg_y, color="black", linewidth=2, label="Average backbone")
        ax.plot(bl_x, bl_y, color="seagreen", linewidth=2.2, linestyle="--", marker="s",
                 markersize=6, label="Bilinear idealization (EEEP)")
        ax.scatter([props_avg["Dmax_mm"]], [props_avg["Fmax_kN"]], color="red", zorder=5,
                   label=f"Peak ({props_avg['Dmax_mm']:.2f}, {props_avg['Fmax_kN']:.2f})")
        ax.scatter([props_avg["Du_mm"]], [props_avg["Fu_kN"]], color="purple", zorder=5,
                   label=f"Ultimate ({props_avg['Du_mm']:.2f}, {props_avg['Fu_kN']:.2f})")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Force (kN)")
        ax.set_title(f"Bilinear Idealization (Average Curve): {stem}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(dirs["bilinear_plot"], f"{stem}_bilinear.png"), dpi=200)
        plt.close(fig)

    # ---- (C2) cyclic properties: stiffness degradation + energy dissipation ----
    cycles = identify_cycles(disp, force)
    cycle_metrics = compute_cycle_metrics(cycles)

    if cycle_metrics is not None:
        cm = cycle_metrics
        pd.DataFrame({
            "cycle_number": cm["cycle_number"],
            "stiffness_kN_per_mm": cm["stiffness_kN_per_mm"],
            "stiffness_pct_of_initial": cm["stiffness_pct_of_initial"],
            "energy_dissipated_kN_mm": cm["energy_kN_mm"],
            "cumulative_energy_kN_mm": cm["cumulative_energy_kN_mm"],
        }).to_csv(os.path.join(dirs["stiffness_csv"], f"{stem}_cyclic_properties.csv"), index=False)

        # -- Stiffness degradation bar chart --
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar(cm["cycle_number"], cm["stiffness_kN_per_mm"], color="#2f8fd6", edgecolor="black")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Secant stiffness (kN/mm)")
        ax.set_title(f"Stiffness Degradation per Cycle: {stem}")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(dirs["stiffness_plot"], f"{stem}_stiffness_degradation.png"), dpi=200)
        plt.close(fig)

        # -- Energy dissipated per loop bar chart --
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar(cm["cycle_number"], cm["energy_kN_mm"], color="#e08a2f", edgecolor="black")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Dissipated energy per loop (kN\u00b7mm)")
        ax.set_title(f"Energy Dissipation per Loop: {stem}")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(dirs["energy_plot"], f"{stem}_energy_per_loop.png"), dpi=200)
        plt.close(fig)

        # -- Cumulative energy dissipation chart --
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar(cm["cycle_number"], cm["cumulative_energy_kN_mm"], color="#3d9c6c", edgecolor="black")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Cumulative dissipated energy (kN\u00b7mm)")
        ax.set_title(f"Cumulative Energy Dissipation: {stem}")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(dirs["energy_plot"], f"{stem}_cumulative_energy.png"), dpi=200)
        plt.close(fig)

    # ---- (D) summary row ----
    def flat(prefix, props):
        d = {f"{prefix}_{k}": v for k, v in props.items() if k != "Notes"}
        d[f"{prefix}_Notes"] = props.get("Notes", "")
        return d

    row = {"Specimen": stem}
    row.update(flat("Pos", props_pos))
    row.update(flat("Neg", props_neg))
    row.update(flat("Avg", props_avg))
    if cycle_metrics is not None:
        row["Num_Cycles"] = int(len(cycle_metrics["cycle_number"]))
        row["Total_Energy_Dissipated_kN_mm"] = float(cycle_metrics["cumulative_energy_kN_mm"][-1])
        row["Final_Stiffness_Pct_of_Initial"] = float(cycle_metrics["stiffness_pct_of_initial"][-1])
    else:
        row["Num_Cycles"] = 0
        row["Total_Energy_Dissipated_kN_mm"] = np.nan
        row["Final_Stiffness_Pct_of_Initial"] = np.nan

    # ---- (E) everything the GUI viewer needs to redraw plots live ----
    plot_data = {
        "stem": stem,
        "disp": disp, "force": force,
        "bx": bx, "by": by,
        "pos_x": pos_x, "pos_y": pos_y,
        "mirror_x": mirror_x, "mirror_y": mirror_y,
        "grid": grid, "avg_y": avg_y,
        "props_avg": props_avg,
        "bilinear": bl,  # (bl_x, bl_y) tuple or None
        "cycle_metrics": cycle_metrics,  # dict of arrays or None
    }

    return row, plot_data


# GUI Utility Helper

class ToolTip:
    """Lightweight tooltip: shows a small popup label when the mouse
    hovers over a widget for a short moment, professional-app style.
    """

    def __init__(self, widget, text, delay=450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        except Exception:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        try:
            self._tip.attributes("-topmost", True)
        except Exception:
            pass
        label = tk.Label(self._tip, text=self.text, justify="left", background="#2b2b2b",
                          foreground="white", relief="solid", borderwidth=1,
                          font=(FONT_FAMILY, 9), padx=6, pady=3)
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def add_tooltip(widget, text):
    return ToolTip(widget, text)


def fmt_num(v, digits=4):
    """Format a number safely for display, tolerating NaN/None/strings."""
    try:
        if v is None:
            return "\u2013"
        if isinstance(v, str):
            return v if v else "\u2013"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        fv = float(v)
        if not np.isfinite(fv):
            return "\u2013"
        return f"{fv:.{digits}g}"
    except Exception:
        return "\u2013"


# GUI

class BackboneGUI(tk.Tk):

    NUMERIC_HIGHLIGHT_CANDIDATES = [
        "Avg_Fmax_kN", "Avg_Dmax_mm", "Avg_Fy_kN", "Avg_Dy_mm",
        "Avg_Fu_kN", "Avg_Du_mm", "Avg_Ke_kN_per_mm", "Avg_Energy_kN_mm",
        "Avg_Ductility", "Num_Cycles", "Total_Energy_Dissipated_kN_mm",
        "Final_Stiffness_Pct_of_Initial",
    ]

    def __init__(self):
        super().__init__()

        # ---- persistent settings ----
        self.settings = load_settings()

        self.title(f"{APP_NAME}  v{APP_VERSION}")
        try:
            self.geometry(self.settings.get("window_geometry", DEFAULT_SETTINGS["window_geometry"]))
        except Exception:
            self.geometry(DEFAULT_SETTINGS["window_geometry"])
        self.minsize(1150, 720)
        self.configure(bg=COLOR_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.input_dir = tk.StringVar(value=self.settings.get("last_input_dir", ""))
        self.output_dir = tk.StringVar(value=self.settings.get("last_output_dir", ""))
        self.status_var = tk.StringVar(value="Ready.")
        self.coord_var = tk.StringVar(value="x = \u2013 ,  y = \u2013")
        self.stats_var = tk.StringVar(value="No files processed yet.")
        self.search_var = tk.StringVar()
        self.table_search_var = tk.StringVar()

        # plot preference vars (persisted)
        self.plottype_var = tk.StringVar(value=PLOT_TYPES[0])
        self.specimen_var = tk.StringVar()
        self.style_var = tk.StringVar(value=self.settings.get("plot_style", "default"))
        self.linewidth_var = tk.DoubleVar(value=self.settings.get("line_width", 2.2))
        self.markersize_var = tk.DoubleVar(value=self.settings.get("marker_size", 5.0))
        self.grid_var = tk.BooleanVar(value=self.settings.get("show_grid", True))
        self.legend_var = tk.BooleanVar(value=self.settings.get("show_legend", True))
        self.markers_var = tk.BooleanVar(value=self.settings.get("show_markers", True))
        self.crosshair_var = tk.BooleanVar(value=self.settings.get("crosshair_enabled", False))
        self.dpi_var = tk.IntVar(value=self.settings.get("export_dpi", 300))
        self.highlight_col_var = tk.StringVar(value="(none)")

        self.results = {}          # stem -> plot_data dict
        self.summary_rows = []     # list of dicts (raw, unfiltered)
        self.summary_df = pd.DataFrame()
        self.sort_state = {}       # column -> ascending bool
        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.cancel_event = threading.Event()
        self.progress_dialog = None
        self.process_start_time = None

        self._current_xy = (np.array([]), np.array([]))  # currently plotted "primary" curve for click-inspect
        self._click_marker = None
        self._crosshair_h = None
        self._crosshair_v = None

        self._configure_style()
        self._build_menu()
        self._build_layout()
        self._build_shortcuts()
        self.after(150, self._poll_log_queue)

    # Style
    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=(FONT_FAMILY, 10), background=COLOR_BG)
        style.configure("TFrame", background=COLOR_BG)
        style.configure("TLabel", background=COLOR_BG, foreground="#1c2b3a")
        style.configure("TLabelframe", background=COLOR_BG)
        style.configure("TLabelframe.Label", background=COLOR_BG, foreground="#1c2b3a",
                         font=(FONT_FAMILY, 10, "bold"))
        style.configure("TCheckbutton", background=COLOR_BG)

        style.configure("TButton", font=(FONT_FAMILY, 10), padding=6)
        style.configure("Accent.TButton", font=(FONT_FAMILY, 10, "bold"),
                         padding=8, background=COLOR_ACCENT, foreground="white")
        style.map("Accent.TButton",
                  background=[("active", COLOR_ACCENT_DARK), ("disabled", "#a9b7c6")],
                  foreground=[("disabled", "#eef2f7")])

        style.configure("Toolbar.TFrame", background=COLOR_TOOLBAR_BG)
        style.configure("Toolbar.TButton", font=(FONT_FAMILY, 9), padding=4)
        style.configure("Toolbar.TLabel", background=COLOR_TOOLBAR_BG, foreground="#1c2b3a",
                         font=(FONT_FAMILY, 9, "bold"))
        style.configure("Toolbar.TCheckbutton", background=COLOR_TOOLBAR_BG, font=(FONT_FAMILY, 9))

        style.configure("Panel.TFrame", background=COLOR_PANEL_BG)
        style.configure("Panel.TLabel", background=COLOR_PANEL_BG, foreground="#1c2b3a",
                         font=(FONT_FAMILY, 9))
        style.configure("PanelKey.TLabel", background=COLOR_PANEL_BG, foreground="#5a6a7d",
                         font=(FONT_FAMILY, 9, "bold"))
        style.configure("PanelValue.TLabel", background=COLOR_PANEL_BG, foreground="#0f2a44",
                         font=(FONT_FAMILY, 11, "bold"))
        style.configure("PanelTitle.TLabel", background=COLOR_PANEL_BG, foreground="#12233d",
                         font=(FONT_FAMILY, 12, "bold"))

        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=(FONT_FAMILY, 10, "bold"), padding=(16, 8))
        style.map("TNotebook.Tab",
                  background=[("selected", "white"), ("!selected", "#dfe6ee")],
                  foreground=[("selected", COLOR_ACCENT_DARK), ("!selected", "#5a6a7d")])

        style.configure("Header.TFrame", background=COLOR_HEADER_BG)
        style.configure("Header.TLabel", background=COLOR_HEADER_BG, foreground=COLOR_HEADER_TEXT)
        style.configure("HeaderSub.TLabel", background=COLOR_HEADER_BG, foreground=COLOR_HEADER_SUBTEXT)

        style.configure("Footer.TFrame", background=COLOR_FOOTER_BG)
        style.configure("Footer.TLabel", background=COLOR_FOOTER_BG, foreground=COLOR_FOOTER_TEXT,
                         font=(FONT_FAMILY, 9))
        style.configure("FooterLink.TLabel", background=COLOR_FOOTER_BG, foreground="#8ecbff",
                         font=(FONT_FAMILY, 9, "underline"))

        style.configure("Status.TLabel", background=COLOR_STATUS_BG, foreground="#1c2b3a",
                         font=(FONT_FAMILY, 9))

        style.configure("Treeview", rowheight=24, font=(FONT_FAMILY, 9))
        style.configure("Treeview.Heading", font=(FONT_FAMILY, 9, "bold"),
                         background="#dfe6ee", foreground="#1c2b3a")

        style.configure("TCombobox", padding=4)

        # Matplotlib figures adopt a light, clean look consistent with the app
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4a5a6a",
            "axes.labelcolor": "#1c2b3a",
            "xtick.color": "#1c2b3a",
            "ytick.color": "#1c2b3a",
            "font.size": 10,
        })

    # Menu Bar
    def _build_menu(self):
        menubar = tk.Menu(self)

        # ---- File ----
        self.file_menu = tk.Menu(menubar, tearoff=0)
        self.file_menu.add_command(label="Choose Input Folder...\tCtrl+O", command=self._browse_input)
        self.file_menu.add_command(label="Choose Output Folder...\tCtrl+Shift+O", command=self._browse_output)

        self.recent_input_menu = tk.Menu(self.file_menu, tearoff=0)
        self.recent_output_menu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Recent Input Folders", menu=self.recent_input_menu)
        self.file_menu.add_cascade(label="Recent Output Folders", menu=self.recent_output_menu)

        self.file_menu.add_separator()
        self.file_menu.add_command(label="Run Processing\tCtrl+R", command=self._start_processing)
        self.file_menu.add_command(label="Open Output Folder", command=self._open_output_folder)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Export Processing Log...", command=self._export_log)
        self.file_menu.add_command(label="Export Summary Table (CSV)...", command=self._export_full_summary)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Preferences...\tCtrl+,", command=self._show_preferences)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit\tCtrl+Q", command=self._on_close)
        menubar.add_cascade(label="File", menu=self.file_menu)

        # View
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(label="Show Grid\tCtrl+G", variable=self.grid_var,
                                   command=self._render_plot)
        view_menu.add_checkbutton(label="Show Legend\tCtrl+L", variable=self.legend_var,
                                   command=self._render_plot)
        view_menu.add_checkbutton(label="Show Markers", variable=self.markers_var,
                                   command=self._render_plot)
        view_menu.add_checkbutton(label="Crosshair Cursor", variable=self.crosshair_var,
                                   command=self._toggle_crosshair)
        view_menu.add_separator()
        style_menu = tk.Menu(view_menu, tearoff=0)
        for s in MPL_STYLES:
            style_menu.add_radiobutton(label=s, variable=self.style_var, value=s,
                                        command=self._apply_style)
        view_menu.add_cascade(label="Plot Style", menu=style_menu)
        view_menu.add_separator()
        view_menu.add_command(label="Reset View\tCtrl+0", command=self._reset_view)
        view_menu.add_command(label="Auto-fit Axes", command=self._autofit_axes)
        view_menu.add_separator()
        view_menu.add_command(label="Clear Processing Log", command=self._clear_log)
        menubar.add_cascade(label="View", menu=view_menu)

        # Tools
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Save Current Plot...\tCtrl+S", command=self._save_plot)
        tools_menu.add_command(label="Export Current Plot...\tCtrl+E", command=self._export_plot_dialog)
        tools_menu.add_command(label="Copy Plot to Clipboard\tCtrl+C", command=self._copy_plot_to_clipboard)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Help
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_command(label="About This Tool", command=self._show_about)
        help_menu.add_command(label="Open Developer's GitHub", command=self._open_github)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)
        self._refresh_recent_menus()

    def _open_github(self):
        webbrowser.open(DEVELOPER_GITHUB_URL)

    def _show_shortcuts(self):
        text = (
            "Ctrl+O        Choose input folder\n"
            "Ctrl+Shift+O  Choose output folder\n"
            "Ctrl+R        Run processing\n"
            "Ctrl+S        Save current plot\n"
            "Ctrl+E        Export current plot (PNG/SVG/PDF)\n"
            "Ctrl+C        Copy plot to clipboard\n"
            "Ctrl+G        Toggle grid\n"
            "Ctrl+L        Toggle legend\n"
            "Ctrl+0        Reset view\n"
            "Ctrl+F        Search specimen\n"
            "F5            Refresh current plot\n"
            "Left / Right  Previous / Next specimen\n"
            "Ctrl+,        Preferences\n"
            "Ctrl+Q        Exit"
        )
        messagebox.showinfo("Keyboard Shortcuts", text)

    def _show_about(self):
        win = tk.Toplevel(self)
        win.title("About")
        win.configure(bg="white")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        pad = {"padx": 24, "pady": 4}

        tk.Label(win, text=APP_NAME, font=(FONT_FAMILY, 13, "bold"),
                 bg="white", fg="#1c2b3a").pack(pady=(20, 2), padx=24)
        tk.Label(win, text=f"Version {APP_VERSION}", font=(FONT_FAMILY, 9),
                 bg="white", fg="#5a6a7d").pack()
        tk.Label(win, text=APP_SUBTITLE, font=(FONT_FAMILY, 9, "italic"),
                 bg="white", fg="#5a6a7d", wraplength=340, justify="center").pack(pady=(4, 14), padx=24)

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=24)

        tk.Label(win, text=f"Developed by {DEVELOPER_NAME}", font=(FONT_FAMILY, 10, "bold"),
                 bg="white", fg="#1c2b3a").pack(**pad, pady=(14, 2))

        gh = tk.Label(win, text=DEVELOPER_GITHUB_LABEL, font=(FONT_FAMILY, 10, "underline"),
                      bg="white", fg=COLOR_ACCENT_DARK, cursor="hand2")
        gh.pack(**pad)
        gh.bind("<Button-1>", lambda e: self._open_github())

        tk.Label(win, text=f"WhatsApp: {DEVELOPER_WHATSAPP_1}", font=(FONT_FAMILY, 10),
                 bg="white", fg="#1c2b3a").pack(**pad)
        tk.Label(win, text=f"WhatsApp: {DEVELOPER_WHATSAPP_2}", font=(FONT_FAMILY, 10),
                 bg="white", fg="#1c2b3a").pack(**pad)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(16, 20))

    # Keyboard Shortcuts
    def _build_shortcuts(self):
        self.bind_all("<Control-o>", lambda e: self._browse_input())
        self.bind_all("<Control-O>", lambda e: self._browse_output())
        self.bind_all("<Control-Shift-O>", lambda e: self._browse_output())
        self.bind_all("<Control-r>", lambda e: self._start_processing())
        self.bind_all("<Control-s>", lambda e: self._save_plot())
        self.bind_all("<Control-e>", lambda e: self._export_plot_dialog())
        self.bind_all("<Control-c>", lambda e: self._copy_plot_to_clipboard())
        self.bind_all("<Control-g>", lambda e: self._toggle_var(self.grid_var))
        self.bind_all("<Control-l>", lambda e: self._toggle_var(self.legend_var))
        self.bind_all("<Control-0>", lambda e: self._reset_view())
        self.bind_all("<F5>", lambda e: self._render_plot())
        self.bind_all("<Control-f>", lambda e: self._focus_search())
        self.bind_all("<Control-comma>", lambda e: self._show_preferences())
        self.bind_all("<Control-q>", lambda e: self._on_close())
        self.bind_all("<Left>", self._on_left_key)
        self.bind_all("<Right>", self._on_right_key)

    def _toggle_var(self, var):
        var.set(not var.get())
        self._render_plot()

    def _focus_search(self):
        try:
            self.search_entry.focus_set()
            self.search_entry.select_range(0, "end")
        except Exception:
            pass

    def _on_left_key(self, _event=None):
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        self._prev_specimen()

    def _on_right_key(self, _event=None):
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        self._next_specimen()

    # Layout
    def _build_layout(self):
        # ---- Branded header ----
        header = ttk.Frame(self, style="Header.TFrame", padding=(18, 14))
        header.pack(side="top", fill="x")

        title_box = ttk.Frame(header, style="Header.TFrame")
        title_box.pack(side="left", fill="y")
        ttk.Label(title_box, text=APP_NAME, style="Header.TLabel",
                  font=(FONT_FAMILY, 16, "bold")).pack(anchor="w")
        ttk.Label(title_box, text=APP_SUBTITLE, style="HeaderSub.TLabel",
                  font=(FONT_FAMILY, 9)).pack(anchor="w", pady=(2, 0))

        credit_box = ttk.Frame(header, style="Header.TFrame")
        credit_box.pack(side="right", fill="y")
        ttk.Label(credit_box, text=f"Developed by {DEVELOPER_NAME}", style="HeaderSub.TLabel",
                  font=(FONT_FAMILY, 9, "bold")).pack(anchor="e")
        ttk.Label(credit_box, text=DEVELOPER_GITHUB_LABEL, style="HeaderSub.TLabel",
                  font=(FONT_FAMILY, 9)).pack(anchor="e")

        # Folder selection + run button
        top = ttk.Frame(self, padding=(14, 12))
        top.pack(side="top", fill="x")

        lbl_in = ttk.Label(top, text="Input folder (contains TXT files):")
        lbl_in.grid(row=0, column=0, sticky="w")
        entry_in = ttk.Entry(top, textvariable=self.input_dir, width=70)
        entry_in.grid(row=0, column=1, padx=8, sticky="we")
        btn_in = ttk.Button(top, text="Browse...", command=self._browse_input)
        btn_in.grid(row=0, column=2, padx=4)
        add_tooltip(entry_in, "Folder containing the raw *.txt hysteresis data files (Ctrl+O)")
        add_tooltip(btn_in, "Choose the input folder (Ctrl+O)")

        lbl_out = ttk.Label(top, text="Output folder (results saved here):")
        lbl_out.grid(row=1, column=0, sticky="w", pady=(8, 0))
        entry_out = ttk.Entry(top, textvariable=self.output_dir, width=70)
        entry_out.grid(row=1, column=1, padx=8, pady=(8, 0), sticky="we")
        btn_out = ttk.Button(top, text="Browse...", command=self._browse_output)
        btn_out.grid(row=1, column=2, padx=4, pady=(8, 0))
        add_tooltip(entry_out, "Folder where plots, CSVs and the summary will be written (Ctrl+Shift+O)")
        add_tooltip(btn_out, "Choose the output folder (Ctrl+Shift+O)")

        self.run_btn = ttk.Button(top, text="\u25B6  Process All TXT Files",
                                   style="Accent.TButton", command=self._start_processing)
        self.run_btn.grid(row=0, column=3, rowspan=2, padx=(18, 0), sticky="ns")
        add_tooltip(self.run_btn, "Run backbone extraction + bilinear idealization on every "
                                   "TXT file in the input folder (Ctrl+R)")

        top.columnconfigure(1, weight=1)

        # Notebook: Viewer / Log / Summary
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=14, pady=(4, 8))
        self.notebook = nb

        self.viewer_tab = ttk.Frame(nb)
        self.log_tab = ttk.Frame(nb)
        self.summary_tab = ttk.Frame(nb)
        nb.add(self.viewer_tab, text="  \U0001F4CA  Plot Viewer  ")
        nb.add(self.log_tab, text="  \U0001F4DC  Processing Log  ")
        nb.add(self.summary_tab, text="  \U0001F4CB  Summary Table  ")

        self._build_viewer_tab()
        self._build_log_tab()
        self._build_summary_tab()

        # Status bar
        status_bar = ttk.Frame(self, style="Status.TLabel")
        status_bar.pack(side="bottom", fill="x")
        ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel",
                  relief="sunken", anchor="w", padding=6).pack(side="left", fill="x", expand=True)
        ttk.Label(status_bar, textvariable=self.coord_var, style="Status.TLabel",
                  relief="sunken", anchor="w", padding=6, width=32).pack(side="right")

        # Footer with developer contact info
        footer = ttk.Frame(self, style="Footer.TFrame", padding=(14, 6))
        footer.pack(side="bottom", fill="x")

        ttk.Label(footer, text=f"\u00A9 {DEVELOPER_NAME}  \u2022  {APP_NAME} v{APP_VERSION}",
                  style="Footer.TLabel").pack(side="left")

        contact_box = ttk.Frame(footer, style="Footer.TFrame")
        contact_box.pack(side="right")

        gh_label = ttk.Label(contact_box, text=DEVELOPER_GITHUB_LABEL, style="FooterLink.TLabel",
                              cursor="hand2")
        gh_label.pack(side="left", padx=(0, 16))
        gh_label.bind("<Button-1>", lambda e: self._open_github())

        ttk.Label(contact_box, text=f"WhatsApp: {DEVELOPER_WHATSAPP_1}  |  {DEVELOPER_WHATSAPP_2}",
                  style="Footer.TLabel").pack(side="left")


    # Viewer Tab

    def _build_viewer_tab(self):
        # ---- Row 1: specimen navigation / search / plot type ----
        nav = ttk.Frame(self.viewer_tab, style="Toolbar.TFrame", padding=6)
        nav.pack(side="top", fill="x")

        prev_btn = ttk.Button(nav, text="\u25C0 Prev", style="Toolbar.TButton", width=8,
                               command=self._prev_specimen)
        prev_btn.pack(side="left", padx=(0, 2))
        next_btn = ttk.Button(nav, text="Next \u25B6", style="Toolbar.TButton", width=8,
                               command=self._next_specimen)
        next_btn.pack(side="left", padx=(0, 12))
        add_tooltip(prev_btn, "Previous specimen (Left arrow)")
        add_tooltip(next_btn, "Next specimen (Right arrow)")

        ttk.Label(nav, text="Specimen:", style="Toolbar.TLabel").pack(side="left")
        self.specimen_combo = ttk.Combobox(nav, textvariable=self.specimen_var,
                                            state="readonly", width=28, values=[])
        self.specimen_combo.pack(side="left", padx=(5, 12))
        self.specimen_combo.bind("<<ComboboxSelected>>", lambda e: self._render_plot())

        ttk.Label(nav, text="Search:", style="Toolbar.TLabel").pack(side="left")
        self.search_entry = ttk.Entry(nav, textvariable=self.search_var, width=16)
        self.search_entry.pack(side="left", padx=(5, 2))
        self.search_entry.bind("<Return>", lambda e: self._search_specimen())
        self.search_entry.bind("<KeyRelease>", self._on_search_keyrelease)
        search_btn = ttk.Button(nav, text="Go", style="Toolbar.TButton", width=4,
                                 command=self._search_specimen)
        search_btn.pack(side="left")
        add_tooltip(self.search_entry, "Type part of a specimen name and press Enter (Ctrl+F)")

        ttk.Label(nav, text="  Plot type:", style="Toolbar.TLabel").pack(side="left", padx=(16, 0))
        self.plottype_combo = ttk.Combobox(nav, textvariable=self.plottype_var,
                                            state="readonly", width=30, values=PLOT_TYPES)
        self.plottype_combo.pack(side="left", padx=5)
        self.plottype_combo.bind("<<ComboboxSelected>>", lambda e: self._render_plot())

        refresh_btn = ttk.Button(nav, text="\u21BB Refresh", style="Toolbar.TButton",
                                  command=self._render_plot)
        refresh_btn.pack(side="right")
        add_tooltip(refresh_btn, "Redraw the current plot (F5)")

        # view / style controls
        view_bar = ttk.Frame(self.viewer_tab, style="Toolbar.TFrame", padding=(6, 2))
        view_bar.pack(side="top", fill="x")

        reset_btn = ttk.Button(view_bar, text="Reset View", style="Toolbar.TButton",
                                command=self._reset_view)
        reset_btn.pack(side="left", padx=2)
        add_tooltip(reset_btn, "Return the plot to its default view (Ctrl+0)")

        autofit_btn = ttk.Button(view_bar, text="Auto-fit", style="Toolbar.TButton",
                                  command=self._autofit_axes)
        autofit_btn.pack(side="left", padx=2)
        add_tooltip(autofit_btn, "Auto-scale axes to fit all plotted data")

        zoomin_btn = ttk.Button(view_bar, text="Zoom +", style="Toolbar.TButton",
                                 command=lambda: self._zoom(0.8))
        zoomin_btn.pack(side="left", padx=2)
        zoomout_btn = ttk.Button(view_bar, text="Zoom \u2212", style="Toolbar.TButton",
                                  command=lambda: self._zoom(1.25))
        zoomout_btn.pack(side="left", padx=2)

        pan_btn = ttk.Button(view_bar, text="\u2716 Pan", style="Toolbar.TButton",
                              command=self._toggle_pan)
        pan_btn.pack(side="left", padx=(2, 12))
        add_tooltip(pan_btn, "Toggle click-and-drag panning")

        crosshair_chk = ttk.Checkbutton(view_bar, text="Crosshair", style="Toolbar.TCheckbutton",
                                         variable=self.crosshair_var, command=self._toggle_crosshair)
        crosshair_chk.pack(side="left", padx=4)
        grid_chk = ttk.Checkbutton(view_bar, text="Grid", style="Toolbar.TCheckbutton",
                                    variable=self.grid_var, command=self._render_plot)
        grid_chk.pack(side="left", padx=4)
        legend_chk = ttk.Checkbutton(view_bar, text="Legend", style="Toolbar.TCheckbutton",
                                      variable=self.legend_var, command=self._render_plot)
        legend_chk.pack(side="left", padx=4)
        markers_chk = ttk.Checkbutton(view_bar, text="Markers", style="Toolbar.TCheckbutton",
                                       variable=self.markers_var, command=self._render_plot)
        markers_chk.pack(side="left", padx=4)

        ttk.Label(view_bar, text="  Line:", style="Toolbar.TLabel").pack(side="left", padx=(10, 0))
        lw_spin = ttk.Spinbox(view_bar, from_=0.5, to=6.0, increment=0.25, width=4,
                               textvariable=self.linewidth_var, command=self._render_plot)
        lw_spin.pack(side="left", padx=2)
        lw_spin.bind("<Return>", lambda e: self._render_plot())

        ttk.Label(view_bar, text="Marker:", style="Toolbar.TLabel").pack(side="left", padx=(8, 0))
        ms_spin = ttk.Spinbox(view_bar, from_=1.0, to=14.0, increment=0.5, width=4,
                               textvariable=self.markersize_var, command=self._render_plot)
        ms_spin.pack(side="left", padx=2)
        ms_spin.bind("<Return>", lambda e: self._render_plot())

        ttk.Label(view_bar, text="Style:", style="Toolbar.TLabel").pack(side="left", padx=(8, 0))
        style_combo = ttk.Combobox(view_bar, textvariable=self.style_var, state="readonly",
                                    width=14, values=MPL_STYLES)
        style_combo.pack(side="left", padx=2)
        style_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_style())

        # save / export / clipboard
        export_bar = ttk.Frame(self.viewer_tab, style="Toolbar.TFrame", padding=(6, 2))
        export_bar.pack(side="top", fill="x")

        save_btn = ttk.Button(export_bar, text="\U0001F4BE Save Plot", style="Toolbar.TButton",
                               command=self._save_plot)
        save_btn.pack(side="left", padx=2)
        add_tooltip(save_btn, "Quick-save the current plot (Ctrl+S)")

        export_btn = ttk.Button(export_bar, text="Export (PNG/SVG/PDF)...", style="Toolbar.TButton",
                                 command=self._export_plot_dialog)
        export_btn.pack(side="left", padx=2)
        add_tooltip(export_btn, "Export with a chosen format and DPI (Ctrl+E)")

        copy_btn = ttk.Button(export_bar, text="\U0001F4CB Copy to Clipboard", style="Toolbar.TButton",
                               command=self._copy_plot_to_clipboard)
        copy_btn.pack(side="left", padx=2)
        add_tooltip(copy_btn, "Copy the current plot image to the clipboard (Ctrl+C)")

        ttk.Label(export_bar, text="  Export DPI:", style="Toolbar.TLabel").pack(side="left", padx=(10, 0))
        dpi_combo = ttk.Combobox(export_bar, textvariable=self.dpi_var, state="readonly", width=6,
                                  values=[100, 150, 200, 300, 450, 600])
        dpi_combo.pack(side="left", padx=2)
        add_tooltip(dpi_combo, "High-DPI export resolution")

        # Main body: plot (left) + engineering info panel (right)
        body = ttk.Panedwindow(self.viewer_tab, orient="horizontal")
        body.pack(fill="both", expand=True)

        canvas_frame = ttk.Frame(body)
        info_frame = ttk.Frame(body, style="Panel.TFrame")
        body.add(canvas_frame, weight=4)
        body.add(info_frame, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.canvas.get_tk_widget().bind("<Button-3>", self._show_plot_context_menu)

        mpl_toolbar_frame = ttk.Frame(canvas_frame)
        mpl_toolbar_frame.pack(side="bottom", fill="x")
        self.mpl_toolbar = NavigationToolbar2Tk(self.canvas, mpl_toolbar_frame)
        self.mpl_toolbar.update()

        self._build_info_panel(info_frame)
        self._build_plot_context_menu()
        self._draw_placeholder()

    def _build_info_panel(self, parent):
        wrapper = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        wrapper.pack(fill="both", expand=True)

        ttk.Label(wrapper, text="Engineering Information", style="PanelTitle.TLabel").pack(
            anchor="w", pady=(0, 4))
        ttk.Separator(wrapper, orient="horizontal").pack(fill="x", pady=(0, 10))

        self.info_vars = {}
        fields = [
            ("Specimen", "specimen"),
            ("Peak Force (kN)", "peak_force"),
            ("Peak Displacement (mm)", "peak_disp"),
            ("Yield Force (kN)", "yield_force"),
            ("Yield Displacement (mm)", "yield_disp"),
            ("Ultimate Force (kN)", "ult_force"),
            ("Ultimate Displacement (mm)", "ult_disp"),
            ("Initial Stiffness (kN/mm)", "stiffness"),
            ("Energy Dissipation (kN\u00b7mm)", "energy"),
            ("Ductility", "ductility"),
            ("Number of Cycles", "cycles"),
            ("Processing Status", "status"),
        ]
        grid = ttk.Frame(wrapper, style="Panel.TFrame")
        grid.pack(fill="x")
        for row_i, (label, key) in enumerate(fields):
            ttk.Label(grid, text=label + ":", style="PanelKey.TLabel").grid(
                row=row_i, column=0, sticky="w", pady=4, padx=(0, 8))
            var = tk.StringVar(value="\u2013")
            self.info_vars[key] = var
            ttk.Label(grid, textvariable=var, style="PanelValue.TLabel", wraplength=190,
                      justify="left").grid(row=row_i, column=1, sticky="w", pady=4)
        grid.columnconfigure(1, weight=1)

        ttk.Separator(wrapper, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(wrapper, text="Selected point (click on plot):", style="PanelKey.TLabel").pack(anchor="w")
        self.selected_point_var = tk.StringVar(value="\u2013")
        ttk.Label(wrapper, textvariable=self.selected_point_var, style="PanelValue.TLabel",
                  wraplength=210, justify="left").pack(anchor="w", pady=(2, 0))

        ttk.Separator(wrapper, orient="horizontal").pack(fill="x", pady=10)
        notes_title = ttk.Label(wrapper, text="Notes / Warnings:", style="PanelKey.TLabel")
        notes_title.pack(anchor="w")
        self.notes_text = tk.Text(wrapper, height=6, wrap="word", font=(FONT_FAMILY, 9),
                                   relief="solid", borderwidth=1, bg="#fbfcfe")
        self.notes_text.pack(fill="both", expand=True, pady=(4, 0))
        self.notes_text.config(state="disabled")

    def _build_plot_context_menu(self):
        self.plot_menu = tk.Menu(self, tearoff=0)
        self.plot_menu.add_command(label="Save Plot...", command=self._save_plot)
        self.plot_menu.add_command(label="Export Plot...", command=self._export_plot_dialog)
        self.plot_menu.add_command(label="Copy to Clipboard", command=self._copy_plot_to_clipboard)
        self.plot_menu.add_separator()
        self.plot_menu.add_command(label="Reset View", command=self._reset_view)
        self.plot_menu.add_command(label="Auto-fit Axes", command=self._autofit_axes)
        self.plot_menu.add_separator()
        self.plot_menu.add_checkbutton(label="Grid", variable=self.grid_var, command=self._render_plot)
        self.plot_menu.add_checkbutton(label="Legend", variable=self.legend_var, command=self._render_plot)
        self.plot_menu.add_checkbutton(label="Markers", variable=self.markers_var, command=self._render_plot)
        self.plot_menu.add_checkbutton(label="Crosshair", variable=self.crosshair_var,
                                        command=self._toggle_crosshair)

    def _show_plot_context_menu(self, event):
        try:
            self.plot_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.plot_menu.grab_release()

    def _build_log_tab(self):
        wrapper = ttk.Frame(self.log_tab, padding=8)
        wrapper.pack(fill="both", expand=True)

        toolbar = ttk.Frame(wrapper)
        toolbar.pack(side="top", fill="x", pady=(0, 6))
        ttk.Label(toolbar, text="Live Processing Log", font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        ttk.Label(toolbar, textvariable=self.stats_var, font=(FONT_FAMILY, 9)).pack(side="left", padx=16)
        ttk.Button(toolbar, text="Export Log...", command=self._export_log).pack(side="right", padx=(6, 0))
        ttk.Button(toolbar, text="Clear Log", command=self._clear_log).pack(side="right")

        self.log_text = tk.Text(wrapper, wrap="word", state="disabled", bg=COLOR_LOG_BG,
                                 fg=COLOR_LOG_DEFAULT, insertbackground=COLOR_LOG_DEFAULT,
                                 font=("Consolas", 10), relief="flat", padx=10, pady=10)
        self.log_text.pack(fill="both", expand=True)

        # Color tags for a clearer, more readable log
        self.log_text.tag_configure("ok", foreground=COLOR_LOG_OK, font=("Consolas", 10, "bold"))
        self.log_text.tag_configure("fail", foreground=COLOR_LOG_FAIL, font=("Consolas", 10, "bold"))
        self.log_text.tag_configure("header", foreground=COLOR_LOG_HEADER, font=("Consolas", 10, "bold"))
        self.log_text.tag_configure("dim", foreground="#6b7c93")

    # Summary Tab
    def _build_summary_tab(self):
        container = ttk.Frame(self.summary_tab, padding=8)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container)
        toolbar.pack(side="top", fill="x", pady=(0, 6))

        ttk.Label(toolbar, text="Search:").pack(side="left")
        search_entry = ttk.Entry(toolbar, textvariable=self.table_search_var, width=28)
        search_entry.pack(side="left", padx=(4, 12))
        self.table_search_var.trace_add("write", lambda *a: self._filter_summary_table())
        add_tooltip(search_entry, "Filter rows by any column text (e.g. specimen name)")

        ttk.Label(toolbar, text="Highlight max/min column:").pack(side="left", padx=(0, 4))
        self.highlight_combo = ttk.Combobox(toolbar, textvariable=self.highlight_col_var,
                                             state="readonly", width=26, values=["(none)"])
        self.highlight_combo.pack(side="left")
        self.highlight_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_summary_highlight())

        ttk.Button(toolbar, text="Copy Selected", command=self._copy_selected_rows).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Export Selected...", command=self._export_selected_rows).pack(
            side="right", padx=2)
        ttk.Button(toolbar, text="View in Plot Viewer", command=self._view_selected_in_plotter).pack(
            side="right", padx=2)

        table_frame = ttk.Frame(container)
        table_frame.pack(fill="both", expand=True)

        yscroll = ttk.Scrollbar(table_frame, orient="vertical")
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal")
        self.summary_tree = ttk.Treeview(table_frame, show="headings", selectmode="extended",
                                          yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.config(command=self.summary_tree.yview)
        xscroll.config(command=self.summary_tree.xview)

        self.summary_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.summary_tree.tag_configure("maxval", background=COLOR_MAX_HL)
        self.summary_tree.tag_configure("minval", background=COLOR_MIN_HL)

        self.summary_tree.bind("<Double-1>", self._on_summary_double_click)
        self.summary_tree.bind("<Button-3>", self._show_summary_context_menu)

        self._build_summary_context_menu()

    def _build_summary_context_menu(self):
        self.summary_menu = tk.Menu(self, tearoff=0)
        self.summary_menu.add_command(label="View in Plot Viewer", command=self._view_selected_in_plotter)
        self.summary_menu.add_separator()
        self.summary_menu.add_command(label="Copy Selected Rows", command=self._copy_selected_rows)
        self.summary_menu.add_command(label="Export Selected Rows...", command=self._export_selected_rows)

    def _show_summary_context_menu(self, event):
        row_id = self.summary_tree.identify_row(event.y)
        if row_id and row_id not in self.summary_tree.selection():
            self.summary_tree.selection_set(row_id)
        try:
            self.summary_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.summary_menu.grab_release()

    # Folder Browsing
    def _browse_input(self):
        folder = filedialog.askdirectory(title="Select folder containing your hysteresis TXT files",
                                          initialdir=self.input_dir.get() or os.path.expanduser("~"))
        if folder:
            self.input_dir.set(folder)
            self._remember_recent("recent_input_dirs", folder)

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select folder where results should be saved",
                                          initialdir=self.output_dir.get() or os.path.expanduser("~"))
        if folder:
            self.output_dir.set(folder)
            self._remember_recent("recent_output_dirs", folder)

    def _remember_recent(self, key, folder):
        recents = list(self.settings.get(key, []))
        if folder in recents:
            recents.remove(folder)
        recents.insert(0, folder)
        max_recent = self.settings.get("max_recent", 6)
        self.settings[key] = recents[:max_recent]
        self._refresh_recent_menus()

    def _refresh_recent_menus(self):
        self.recent_input_menu.delete(0, "end")
        recents_in = self.settings.get("recent_input_dirs", [])
        if not recents_in:
            self.recent_input_menu.add_command(label="(none)", state="disabled")
        else:
            for folder in recents_in:
                self.recent_input_menu.add_command(
                    label=folder, command=lambda f=folder: self.input_dir.set(f))

        self.recent_output_menu.delete(0, "end")
        recents_out = self.settings.get("recent_output_dirs", [])
        if not recents_out:
            self.recent_output_menu.add_command(label="(none)", state="disabled")
        else:
            for folder in recents_out:
                self.recent_output_menu.add_command(
                    label=folder, command=lambda f=folder: self.output_dir.set(f))

    def _open_output_folder(self):
        out = self.output_dir.get().strip()
        if not out or not os.path.isdir(out):
            messagebox.showinfo("Output folder", "Set and process an output folder first.")
            return
        try:
            os.startfile(out)  # Windows
        except AttributeError:
            if sys.platform == "darwin":
                subprocess.Popen(["open", out])
            else:
                subprocess.Popen(["xdg-open", out])

    # Preference Dialog
    def _show_preferences(self):
        win = tk.Toplevel(self)
        win.title("Preferences")
        win.configure(bg="white")
        win.transient(self)
        win.grab_set()
        win.resizable(False, False)

        pad = {"padx": 20, "pady": 6}
        row = 0

        tk.Label(win, text="Application Preferences", font=(FONT_FAMILY, 12, "bold"),
                 bg="white", fg="#1c2b3a").grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        tk.Label(win, text="Default plot style:", bg="white").grid(row=row, column=0, sticky="w", **pad)
        style_var = tk.StringVar(value=self.style_var.get())
        ttk.Combobox(win, textvariable=style_var, state="readonly", values=MPL_STYLES, width=20).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        tk.Label(win, text="Default line width:", bg="white").grid(row=row, column=0, sticky="w", **pad)
        lw_var = tk.DoubleVar(value=self.linewidth_var.get())
        ttk.Spinbox(win, from_=0.5, to=6.0, increment=0.25, textvariable=lw_var, width=10).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        tk.Label(win, text="Default marker size:", bg="white").grid(row=row, column=0, sticky="w", **pad)
        ms_var = tk.DoubleVar(value=self.markersize_var.get())
        ttk.Spinbox(win, from_=1.0, to=14.0, increment=0.5, textvariable=ms_var, width=10).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        tk.Label(win, text="Default export DPI:", bg="white").grid(row=row, column=0, sticky="w", **pad)
        dpi_var = tk.IntVar(value=self.dpi_var.get())
        ttk.Combobox(win, textvariable=dpi_var, state="readonly", width=8,
                     values=[100, 150, 200, 300, 450, 600]).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        grid_var = tk.BooleanVar(value=self.grid_var.get())
        ttk.Checkbutton(win, text="Show grid by default", variable=grid_var).grid(
            row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        legend_var = tk.BooleanVar(value=self.legend_var.get())
        ttk.Checkbutton(win, text="Show legend by default", variable=legend_var).grid(
            row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        markers_var = tk.BooleanVar(value=self.markers_var.get())
        ttk.Checkbutton(win, text="Show markers by default", variable=markers_var).grid(
            row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        tk.Label(win, text="Max recent folders remembered:", bg="white").grid(
            row=row, column=0, sticky="w", **pad)
        recent_var = tk.IntVar(value=self.settings.get("max_recent", 6))
        ttk.Spinbox(win, from_=1, to=15, textvariable=recent_var, width=10).grid(
            row=row, column=1, sticky="w", **pad)
        row += 1

        def apply_and_close():
            self.style_var.set(style_var.get())
            self.linewidth_var.set(lw_var.get())
            self.markersize_var.set(ms_var.get())
            self.dpi_var.set(dpi_var.get())
            self.grid_var.set(grid_var.get())
            self.legend_var.set(legend_var.get())
            self.markers_var.set(markers_var.get())
            self.settings["max_recent"] = max(1, int(recent_var.get()))
            self._apply_style()
            win.destroy()

        btn_row = ttk.Frame(win)
        btn_row.grid(row=row, column=0, columnspan=2, pady=(14, 16))
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="right", padx=6)
        ttk.Button(btn_row, text="Apply", style="Accent.TButton", command=apply_and_close).pack(
            side="right", padx=6)

    
    # Processing (runs in a background thread so the GUI doesn't freeze)
    
    def _start_processing(self):
        input_dir = self.input_dir.get().strip()
        output_dir = self.output_dir.get().strip()

        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Missing input folder", "Please choose a valid input folder first.")
            return
        if not output_dir:
            messagebox.showerror("Missing output folder", "Please choose an output folder first.")
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Cannot create output folder", f"Could not create output folder:\n{e}")
            return

        try:
            txt_files = find_txt_files(input_dir)
        except Exception as e:
            messagebox.showerror("Error scanning input folder", str(e))
            return

        if not txt_files:
            messagebox.showwarning("No TXT files", f"No .txt files found in:\n{input_dir}")
            return

        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "Processing is already running. Please wait.")
            return

        self._remember_recent("recent_input_dirs", input_dir)
        self._remember_recent("recent_output_dirs", output_dir)

        self.run_btn.config(state="disabled")
        self._clear_log()
        self.results.clear()
        self.summary_rows = []
        self.summary_df = pd.DataFrame()
        self.specimen_combo["values"] = []
        self.specimen_var.set("")
        self._draw_placeholder()
        self._clear_info_panel()
        self.status_var.set("Processing...")
        self.cancel_event.clear()
        self.process_start_time = time.time()

        self._open_progress_dialog(len(txt_files))

        self.worker_thread = threading.Thread(
            target=self._process_all_worker, args=(txt_files, output_dir), daemon=True
        )
        self.worker_thread.start()

    def _open_progress_dialog(self, total):
        win = tk.Toplevel(self)
        win.title("Processing...")
        win.configure(bg="white")
        win.transient(self)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", lambda: None)  # block the [x] button; use Cancel instead

        tk.Label(win, text="Processing hysteresis TXT files", font=(FONT_FAMILY, 11, "bold"),
                 bg="white", fg="#1c2b3a").pack(padx=24, pady=(18, 4))

        self.progress_file_var = tk.StringVar(value="Starting...")
        tk.Label(win, textvariable=self.progress_file_var, font=(FONT_FAMILY, 9),
                 bg="white", fg="#5a6a7d", wraplength=360).pack(padx=24, pady=(0, 10))

        self.progress_bar = ttk.Progressbar(win, mode="determinate", maximum=max(total, 1), length=360)
        self.progress_bar.pack(padx=24, pady=(0, 6))

        self.progress_count_var = tk.StringVar(value=f"0 / {total}")
        tk.Label(win, textvariable=self.progress_count_var, font=(FONT_FAMILY, 9),
                 bg="white", fg="#5a6a7d").pack(pady=(0, 4))

        self.progress_timer_var = tk.StringVar(value="Elapsed: 0.0 s")
        tk.Label(win, textvariable=self.progress_timer_var, font=(FONT_FAMILY, 9),
                 bg="white", fg="#5a6a7d").pack(pady=(0, 10))

        ttk.Button(win, text="Cancel", command=self._cancel_processing).pack(pady=(0, 16))

        win.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{max(x,0)}+{max(y,0)}")
        win.grab_set()

        self.progress_dialog = win
        self._tick_progress_timer()

    def _tick_progress_timer(self):
        if self.progress_dialog is None or not self.progress_dialog.winfo_exists():
            return
        elapsed = time.time() - self.process_start_time if self.process_start_time else 0.0
        self.progress_timer_var.set(f"Elapsed: {elapsed:.1f} s")
        self.progress_dialog.after(100, self._tick_progress_timer)

    def _cancel_processing(self):
        self.cancel_event.set()
        if self.progress_dialog is not None and self.progress_dialog.winfo_exists():
            self.progress_file_var.set("Cancelling... finishing current file.")

    def _close_progress_dialog(self):
        if self.progress_dialog is not None and self.progress_dialog.winfo_exists():
            self.progress_dialog.grab_release()
            self.progress_dialog.destroy()
        self.progress_dialog = None

    def _process_all_worker(self, txt_files, output_dir):
        dirs = {
            "backbone_plot": os.path.join(output_dir, BACKBONE_PLOTS_DIR),
            "backbone_csv": os.path.join(output_dir, BACKBONE_CSVS_DIR),
            "pna_plot": os.path.join(output_dir, PNA_PLOTS_DIR),
            "pna_csv": os.path.join(output_dir, PNA_CSVS_DIR),
            "bilinear_plot": os.path.join(output_dir, BILINEAR_PLOTS_DIR),
            "bilinear_csv": os.path.join(output_dir, BILINEAR_CSVS_DIR),
            "stiffness_plot": os.path.join(output_dir, STIFFNESS_PLOTS_DIR),
            "stiffness_csv": os.path.join(output_dir, STIFFNESS_CSVS_DIR),
            "energy_plot": os.path.join(output_dir, ENERGY_PLOTS_DIR),
            "energy_csv": os.path.join(output_dir, ENERGY_CSVS_DIR),
            "summary": os.path.join(output_dir, SUMMARY_DIR),
        }
        try:
            for d in dirs.values():
                os.makedirs(d, exist_ok=True)
        except Exception as e:
            self.log_queue.put(("fail", f"[FAIL] Could not create output directories: {e}\n"))
            self.log_queue.put(("__DONE__", {}, []))
            return

        self.log_queue.put(("header", f"Found {len(txt_files)} TXT file(s) in input folder.\n"))

        succeeded, failed = [], []
        summary_rows = []
        results = {}
        t0 = time.time()

        for i, txt_path in enumerate(txt_files, start=1):
            if self.cancel_event.is_set():
                self.log_queue.put(("dim", f"\nCancelled by user after {i - 1} of {len(txt_files)} file(s).\n"))
                break

            name = os.path.basename(txt_path)
            self.log_queue.put(("__PROGRESS__", i, len(txt_files), name))
            try:
                row, plot_data = process_one_file(txt_path, dirs)
                summary_rows.append(row)
                results[plot_data["stem"]] = plot_data
                self.log_queue.put(("ok", f"[OK]   {name}\n"))
                succeeded.append(name)
            except Exception as e:
                self.log_queue.put(("fail", f"[FAIL] {name}: {e}\n"))
                self.log_queue.put(("dim", traceback.format_exc(limit=3) + "\n"))
                failed.append(name)

        elapsed = time.time() - t0
        summary_path = None
        if summary_rows:
            try:
                summary_df = pd.DataFrame(summary_rows)
                summary_path = os.path.join(dirs["summary"], "backbone_summary.csv")
                summary_df.to_csv(summary_path, index=False)
            except Exception as e:
                self.log_queue.put(("fail", f"[FAIL] Could not write master summary CSV: {e}\n"))

        n_total = len(txt_files)
        avg = elapsed / max(len(succeeded) + len(failed), 1)
        self.log_queue.put(("dim", "\n--------------------------------------------------\n"))
        self.log_queue.put(("header",
            f"Done. {len(succeeded)} succeeded, {len(failed)} failed "
            f"out of {n_total} TXT file(s) in {elapsed:.2f} s "
            f"({avg:.2f} s/file average).\n"
        ))
        if summary_path:
            self.log_queue.put(("normal", f"Master summary saved to: {summary_path}\n"))
        for label, d in dirs.items():
            self.log_queue.put(("dim", f"  {label:14s}: {d}\n"))
        if failed:
            self.log_queue.put(("fail", "Failed files:\n"))
            for f in failed:
                self.log_queue.put(("fail", f"   - {f}\n"))

        # Hand results back to the GUI thread
        self.log_queue.put(("__STATS__", len(succeeded), len(failed), n_total, elapsed))
        self.log_queue.put(("__DONE__", results, summary_rows))


    # Log queue plotting (safe cross-thread GUI updates)

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    _, results, summary_rows = item
                    self._on_processing_done(results, summary_rows)
                elif isinstance(item, tuple) and item and item[0] == "__PROGRESS__":
                    _, i, total, name = item
                    self._on_progress_update(i, total, name)
                elif isinstance(item, tuple) and item and item[0] == "__STATS__":
                    _, n_ok, n_fail, n_total, elapsed = item
                    self.stats_var.set(
                        f"Last run: {n_ok} OK, {n_fail} failed, {n_total} total \u2022 "
                        f"{elapsed:.2f} s elapsed"
                    )
                elif isinstance(item, tuple) and len(item) == 2:
                    tag, text = item
                    self._append_log(text, tag)
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)

    def _on_progress_update(self, i, total, name):
        if self.progress_dialog is not None and self.progress_dialog.winfo_exists():
            self.progress_bar["value"] = i
            self.progress_count_var.set(f"{i} / {total}")
            self.progress_file_var.set(f"Processing: {name}")
        self.status_var.set(f"Processing {i} of {total}: {name}")

    def _append_log(self, text, tag="normal"):
        self.log_text.config(state="normal")
        if tag and tag != "normal":
            self.log_text.insert("end", text, tag)
        else:
            self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _export_log(self):
        content = self.log_text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Export Log", "The processing log is empty.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Processing Log", defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("Log file", "*.log"), ("All files", "*.*")],
            initialfile=f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.status_var.set(f"Processing log exported to: {path}")
        except Exception as e:
            messagebox.showerror("Export failed", f"Could not export the log:\n{e}")

    def _on_processing_done(self, results, summary_rows):
        self.results = results
        self.summary_rows = summary_rows
        self.run_btn.config(state="normal")
        self._close_progress_dialog()

        if self.cancel_event.is_set():
            self.status_var.set(f"Cancelled. {len(results)} specimen(s) processed before cancellation.")
        else:
            self.status_var.set(f"Done. {len(results)} specimen(s) processed successfully.")

        stems = sorted(results.keys())
        self.specimen_combo["values"] = stems
        if stems:
            self.specimen_var.set(stems[0])
            self._render_plot()
        else:
            self._clear_info_panel()

        self._populate_summary_table(summary_rows)

        if not stems:
            messagebox.showwarning("Nothing processed",
                                    "No specimens were processed successfully. Check the Processing Log tab.")

    # Summary Table
    def _populate_summary_table(self, summary_rows):
        self.summary_tree.delete(*self.summary_tree.get_children())
        self.sort_state = {}
        if not summary_rows:
            self.summary_tree["columns"] = []
            self.summary_df = pd.DataFrame()
            self.highlight_combo["values"] = ["(none)"]
            self.highlight_col_var.set("(none)")
            return

        self.summary_df = pd.DataFrame(summary_rows)
        cols = list(self.summary_df.columns)
        self.summary_tree["columns"] = cols
        for c in cols:
            self.summary_tree.heading(c, text=c, command=lambda col=c: self._sort_summary_by(col))
            self.summary_tree.column(c, width=130, anchor="center", stretch=False)

        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.summary_df[c])]
        self.highlight_combo["values"] = ["(none)"] + numeric_cols
        preferred = [c for c in self.NUMERIC_HIGHLIGHT_CANDIDATES if c in numeric_cols]
        self.highlight_col_var.set(preferred[0] if preferred else "(none)")

        self._render_summary_rows(self.summary_df)
        self._apply_summary_highlight()

    def _render_summary_rows(self, df):
        self.summary_tree.delete(*self.summary_tree.get_children())
        cols = list(df.columns)
        for idx, r in df.iterrows():
            values = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    values.append("" if pd.isna(v) else f"{v:.4g}")
                else:
                    values.append(v)
            self.summary_tree.insert("", "end", iid=str(idx), values=values)

    def _sort_summary_by(self, col):
        if self.summary_df.empty:
            return
        ascending = not self.sort_state.get(col, False)
        self.sort_state = {col: ascending}
        try:
            df_sorted = self.summary_df.sort_values(by=col, ascending=ascending, kind="mergesort",
                                                      na_position="last")
        except Exception:
            df_sorted = self.summary_df
        self._render_summary_rows(df_sorted)
        self._apply_summary_highlight(df_sorted)
        for c in self.summary_tree["columns"]:
            base = c
            arrow = " \u25B2" if (c == col and ascending) else (" \u25BC" if c == col else "")
            self.summary_tree.heading(c, text=base + arrow)

    def _filter_summary_table(self):
        if self.summary_df.empty:
            return
        query = self.table_search_var.get().strip().lower()
        if not query:
            self._render_summary_rows(self.summary_df)
            self._apply_summary_highlight(self.summary_df)
            return
        mask = self.summary_df.apply(
            lambda row: row.astype(str).str.lower().str.contains(query, regex=False).any(), axis=1)
        filtered = self.summary_df[mask]
        self._render_summary_rows(filtered)
        self._apply_summary_highlight(filtered)

    def _apply_summary_highlight(self, df=None):
        if df is None:
            df = self.summary_df
        col = self.highlight_col_var.get()
        # clear existing highlight tags on all currently displayed rows
        for iid in self.summary_tree.get_children():
            tags = tuple(t for t in self.summary_tree.item(iid, "tags") if t not in ("maxval", "minval"))
            self.summary_tree.item(iid, tags=tags)
        if not col or col == "(none)" or df.empty or col not in df.columns:
            return
        series = pd.to_numeric(df[col], errors="coerce")
        if series.dropna().empty:
            return
        max_idx = series.idxmax()
        min_idx = series.idxmin()
        for idx, tag in ((max_idx, "maxval"), (min_idx, "minval")):
            iid = str(idx)
            if self.summary_tree.exists(iid):
                existing = self.summary_tree.item(iid, "tags")
                self.summary_tree.item(iid, tags=tuple(existing) + (tag,))

    def _copy_selected_rows(self):
        sel = self.summary_tree.selection()
        if not sel:
            messagebox.showinfo("Copy Selected", "Select one or more rows first.")
            return
        cols = self.summary_tree["columns"]
        lines = ["\t".join(cols)]
        for iid in sel:
            values = self.summary_tree.item(iid, "values")
            lines.append("\t".join(str(v) for v in values))
        text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set(f"Copied {len(sel)} row(s) to clipboard.")

    def _export_selected_rows(self):
        sel = self.summary_tree.selection()
        if not sel:
            messagebox.showinfo("Export Selected", "Select one or more rows first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Selected Rows", defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
            initialfile="selected_specimens.csv")
        if not path:
            return
        cols = self.summary_tree["columns"]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                for iid in sel:
                    writer.writerow(self.summary_tree.item(iid, "values"))
            self.status_var.set(f"Exported {len(sel)} row(s) to: {path}")
        except Exception as e:
            messagebox.showerror("Export failed", f"Could not export rows:\n{e}")

    def _export_full_summary(self):
        if self.summary_df.empty:
            messagebox.showinfo("Export Summary", "No summary data available yet. Process files first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Summary Table", defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
            initialfile="backbone_summary_export.csv")
        if not path:
            return
        try:
            self.summary_df.to_csv(path, index=False)
            self.status_var.set(f"Full summary exported to: {path}")
        except Exception as e:
            messagebox.showerror("Export failed", f"Could not export summary:\n{e}")

    def _on_summary_double_click(self, _event=None):
        self._view_selected_in_plotter()

    def _view_selected_in_plotter(self):
        sel = self.summary_tree.selection()
        if not sel:
            return
        values = self.summary_tree.item(sel[0], "values")
        cols = self.summary_tree["columns"]
        if "Specimen" not in cols:
            return
        stem = values[list(cols).index("Specimen")]
        if stem in self.results:
            self.specimen_var.set(stem)
            self.notebook.select(self.viewer_tab)
            self._render_plot()
        else:
            messagebox.showinfo("Not available", f"No plot data cached for specimen '{stem}'.")

    # Specimen Navigation
    def _prev_specimen(self):
        stems = list(self.specimen_combo["values"])
        if not stems:
            return
        cur = self.specimen_var.get()
        idx = stems.index(cur) if cur in stems else 0
        self.specimen_var.set(stems[(idx - 1) % len(stems)])
        self._render_plot()

    def _next_specimen(self):
        stems = list(self.specimen_combo["values"])
        if not stems:
            return
        cur = self.specimen_var.get()
        idx = stems.index(cur) if cur in stems else -1
        self.specimen_var.set(stems[(idx + 1) % len(stems)])
        self._render_plot()

    def _on_search_keyrelease(self, event):
        if event.keysym in ("Return", "Up", "Down", "Left", "Right"):
            return
        query = self.search_var.get().strip().lower()
        if not query:
            return
        stems = list(self.specimen_combo["values"])
        match = next((s for s in stems if query in s.lower()), None)
        if match:
            self.specimen_combo.set(match)  # preview only; Enter/Go commits + renders

    def _search_specimen(self):
        query = self.search_var.get().strip().lower()
        stems = list(self.specimen_combo["values"])
        if not stems:
            return
        if not query:
            messagebox.showinfo("Search", "Type part of a specimen name first.")
            return
        match = next((s for s in stems if query in s.lower()), None)
        if match:
            self.specimen_var.set(match)
            self._render_plot()
            self.status_var.set(f"Found specimen: {match}")
        else:
            messagebox.showinfo("Search", f"No specimen found matching '{self.search_var.get()}'.")

    # View Controlls
    def _apply_style(self):
        style = self.style_var.get()
        try:
            plt.style.use(style if style != "default" else "default")
        except Exception:
            pass
        # Re-assert our own palette on top of the chosen style for readability
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })
        self._render_plot()

    def _toggle_pan(self):
        try:
            self.mpl_toolbar.pan()
        except Exception:
            pass

    def _reset_view(self):
        try:
            self.mpl_toolbar.home()
        except Exception:
            self._render_plot()

    def _autofit_axes(self):
        self.ax.relim()
        self.ax.autoscale()
        self.canvas.draw_idle()

    def _zoom(self, factor):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xc = (xlim[0] + xlim[1]) / 2.0
        yc = (ylim[0] + ylim[1]) / 2.0
        xh = (xlim[1] - xlim[0]) * factor / 2.0
        yh = (ylim[1] - ylim[0]) * factor / 2.0
        self.ax.set_xlim(xc - xh, xc + xh)
        self.ax.set_ylim(yc - yh, yc + yh)
        self.canvas.draw_idle()

    def _toggle_crosshair(self):
        if not self.crosshair_var.get():
            if self._crosshair_h is not None:
                try:
                    self._crosshair_h.remove()
                    self._crosshair_v.remove()
                except Exception:
                    pass
                self._crosshair_h = None
                self._crosshair_v = None
                self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax:
            self.coord_var.set("x = \u2013 ,  y = \u2013")
            return
        self.coord_var.set(f"x = {event.xdata:.4g} mm ,  y = {event.ydata:.4g} kN")

        if not self.crosshair_var.get():
            return
        if self._crosshair_h is None:
            self._crosshair_h = self.ax.axhline(event.ydata, color="#888888", linewidth=0.7,
                                                 linestyle=":", zorder=10)
            self._crosshair_v = self.ax.axvline(event.xdata, color="#888888", linewidth=0.7,
                                                 linestyle=":", zorder=10)
        else:
            self._crosshair_h.set_ydata([event.ydata, event.ydata])
            self._crosshair_v.set_xdata([event.xdata, event.xdata])
        self.canvas.draw_idle()

    def _on_plot_click(self, event):
        if event.inaxes != self.ax:
            return
        x_arr, y_arr = self._current_xy
        if len(x_arr) == 0:
            return
        # nearest point by pixel distance (scale-independent)
        try:
            disp_xy = self.ax.transData.transform(np.column_stack([x_arr, y_arr]))
            click_xy = np.array([event.x, event.y])
            dists = np.hypot(disp_xy[:, 0] - click_xy[0], disp_xy[:, 1] - click_xy[1])
            idx = int(np.argmin(dists))
        except Exception:
            return

        px, py = float(x_arr[idx]), float(y_arr[idx])
        self.selected_point_var.set(f"Displacement = {px:.4g} mm\nForce = {py:.4g} kN")

        if self._click_marker is not None:
            try:
                self._click_marker.remove()
            except Exception:
                pass
        self._click_marker = self.ax.scatter([px], [py], s=90, facecolors="none",
                                              edgecolors="#ff2d55", linewidths=2, zorder=20)
        self.canvas.draw_idle()

    # Save / Export / Clipboard
    def _default_export_name(self, ext):
        stem = self.specimen_var.get() or "plot"
        plot_type = self.plottype_var.get().replace(" ", "_").replace("/", "-")
        return f"{stem}_{plot_type}.{ext}"

    def _save_plot(self):
        path = filedialog.asksaveasfilename(
            title="Save Current Plot", defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("SVG vector", "*.svg"), ("PDF document", "*.pdf"),
                       ("All files", "*.*")],
            initialfile=self._default_export_name("png"))
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=self.dpi_var.get())
            self.status_var.set(f"Plot saved to: {path}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save the plot:\n{e}")

    def _export_plot_dialog(self):
        win = tk.Toplevel(self)
        win.title("Export Plot")
        win.configure(bg="white")
        win.transient(self)
        win.grab_set()
        win.resizable(False, False)

        pad = {"padx": 20, "pady": 8}
        tk.Label(win, text="Export Current Plot", font=(FONT_FAMILY, 12, "bold"),
                 bg="white", fg="#1c2b3a").grid(row=0, column=0, columnspan=2, sticky="w", **pad)

        tk.Label(win, text="Format:", bg="white").grid(row=1, column=0, sticky="w", **pad)
        fmt_var = tk.StringVar(value="PNG")
        ttk.Combobox(win, textvariable=fmt_var, state="readonly", width=14,
                     values=["PNG", "SVG", "PDF"]).grid(row=1, column=1, sticky="w", **pad)

        tk.Label(win, text="DPI (raster formats):", bg="white").grid(row=2, column=0, sticky="w", **pad)
        dpi_var = tk.IntVar(value=self.dpi_var.get())
        ttk.Combobox(win, textvariable=dpi_var, state="readonly", width=14,
                     values=[100, 150, 200, 300, 450, 600, 1200]).grid(row=2, column=1, sticky="w", **pad)

        def do_export():
            fmt = fmt_var.get().lower()
            ext = fmt
            path = filedialog.asksaveasfilename(
                title="Export Plot", defaultextension=f".{ext}",
                filetypes=[(f"{fmt_var.get()} file", f"*.{ext}"), ("All files", "*.*")],
                initialfile=self._default_export_name(ext))
            if not path:
                return
            try:
                self.fig.savefig(path, format=ext, dpi=dpi_var.get())
                self.status_var.set(f"Plot exported ({fmt_var.get()}, {dpi_var.get()} DPI) to: {path}")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Export failed", f"Could not export the plot:\n{e}")

        btn_row = ttk.Frame(win)
        btn_row.grid(row=3, column=0, columnspan=2, pady=(10, 16))
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="right", padx=6)
        ttk.Button(btn_row, text="Export", style="Accent.TButton", command=do_export).pack(
            side="right", padx=6)

    def _copy_plot_to_clipboard(self):
        buf = io.BytesIO()
        try:
            self.fig.savefig(buf, format="png", dpi=self.dpi_var.get())
        except Exception as e:
            messagebox.showerror("Copy failed", f"Could not render the plot for copying:\n{e}")
            return

        # Best effort: true bitmap-to-clipboard is OS specific. Try Windows first.
        if sys.platform.startswith("win"):
            try:
                import tempfile
                from PIL import Image
                import win32clipboard  # type: ignore

                buf.seek(0)
                image = Image.open(buf).convert("RGB")
                output = io.BytesIO()
                image.save(output, "BMP")
                data = output.getvalue()[14:]  # strip BMP file header for CF_DIB
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                self.status_var.set("Plot image copied to clipboard.")
                return
            except Exception:
                pass  # fall through to file-based fallback

        # Cross-platform fallback: save to a temp file and put its path on the clipboard.
        try:
            import tempfile
            tmp_path = os.path.join(tempfile.gettempdir(),
                                     f"bbcurve_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            with open(tmp_path, "wb") as f:
                f.write(buf.getvalue())
            self.clipboard_clear()
            self.clipboard_append(tmp_path)
            self.status_var.set(
                f"Direct image clipboard copy isn't available on this platform. "
                f"Plot saved and its path copied to clipboard: {tmp_path}")
        except Exception as e:
            messagebox.showerror("Copy failed", f"Could not copy the plot:\n{e}")

    # Plot Viewer
    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Process TXT files to view plots here",
                     ha="center", va="center", fontsize=12, color="gray",
                     transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._current_xy = (np.array([]), np.array([]))
        self._click_marker = None
        self.canvas.draw_idle()

    def _clear_info_panel(self):
        for var in self.info_vars.values():
            var.set("\u2013")
        self.selected_point_var.set("\u2013")
        self.notes_text.config(state="normal")
        self.notes_text.delete("1.0", "end")
        self.notes_text.config(state="disabled")

    def _update_info_panel(self, stem, d):
        props = d.get("props_avg", {})
        cm = d.get("cycle_metrics")

        self.info_vars["specimen"].set(stem)
        self.info_vars["peak_force"].set(fmt_num(props.get("Fmax_kN")))
        self.info_vars["peak_disp"].set(fmt_num(props.get("Dmax_mm")))
        self.info_vars["yield_force"].set(fmt_num(props.get("Fy_kN")))
        self.info_vars["yield_disp"].set(fmt_num(props.get("Dy_mm")))
        self.info_vars["ult_force"].set(fmt_num(props.get("Fu_kN")))
        self.info_vars["ult_disp"].set(fmt_num(props.get("Du_mm")))
        self.info_vars["stiffness"].set(fmt_num(props.get("Ke_kN_per_mm")))
        self.info_vars["energy"].set(fmt_num(props.get("Energy_kN_mm")))
        self.info_vars["ductility"].set(fmt_num(props.get("Ductility")))
        self.info_vars["cycles"].set(str(len(cm["cycle_number"])) if cm is not None else "0")

        notes = props.get("Notes", "")
        status = "OK" if not notes else "OK (see notes below)"
        if "error" in props:
            status = "FAILED"
        self.info_vars["status"].set(status)

        self.notes_text.config(state="normal")
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("1.0", notes if notes else "No warnings for this specimen.")
        self.notes_text.config(state="disabled")

        self.selected_point_var.set("\u2013")
        self._click_marker = None

    def _render_plot(self):
        stem = self.specimen_var.get()
        plot_type = self.plottype_var.get()
        if not stem or stem not in self.results:
            self._draw_placeholder()
            self._clear_info_panel()
            return

        d = self.results[stem]
        self._update_info_panel(stem, d)

        self.ax.clear()
        self._click_marker = None

        lw = max(self.linewidth_var.get(), 0.1)
        ms = max(self.markersize_var.get(), 0.5)
        marker_o = "o" if self.markers_var.get() else None
        marker_s = "s" if self.markers_var.get() else None

        if plot_type == "Backbone Curve":
            self.ax.plot(d["disp"], d["force"], color="lightgray", linewidth=max(lw * 0.4, 0.4),
                         label="Raw hysteresis data")
            self.ax.plot(d["bx"], d["by"], color="crimson", linewidth=lw,
                         marker=marker_o, markersize=ms, label="Backbone curve")
            self.ax.scatter([0], [0], color="black", zorder=5, s=25)
            self.ax.set_title(f"Backbone Curve: {stem}")
            self._current_xy = (d["bx"], d["by"])

        elif plot_type == "Positive / Negative / Average":
            self.ax.plot(d["pos_x"], d["pos_y"], color="royalblue", linewidth=lw,
                         marker=marker_o, markersize=ms, label="Positive branch")
            self.ax.plot(d["mirror_x"], d["mirror_y"], color="darkorange", linewidth=lw,
                         marker=marker_o, markersize=ms, label="Negative branch (mirrored)")
            self.ax.plot(d["grid"], d["avg_y"], color="black", linewidth=lw + 0.2,
                         linestyle="--", label="Average branch")
            self.ax.set_xlim(left=0)
            self.ax.set_ylim(bottom=0)
            self.ax.set_title(f"Positive / Negative(mirrored) / Average: {stem}")
            self._current_xy = (d["grid"], d["avg_y"])

        elif plot_type == "Bilinear Idealization":
            props_avg = d["props_avg"]
            self.ax.plot(d["grid"], d["avg_y"], color="black", linewidth=lw,
                         label="Average backbone")
            if d["bilinear"] is not None:
                bl_x, bl_y = d["bilinear"]
                self.ax.plot(bl_x, bl_y, color="seagreen", linewidth=lw + 0.2, linestyle="--",
                             marker=marker_s, markersize=ms + 1, label="Bilinear idealization (EEEP)")
                self.ax.scatter([props_avg["Dmax_mm"]], [props_avg["Fmax_kN"]], color="red", zorder=5,
                                label=f"Peak ({props_avg['Dmax_mm']:.2f}, {props_avg['Fmax_kN']:.2f})")
                self.ax.scatter([props_avg["Du_mm"]], [props_avg["Fu_kN"]], color="purple", zorder=5,
                                label=f"Ultimate ({props_avg['Du_mm']:.2f}, {props_avg['Fu_kN']:.2f})")
                self._current_xy = (bl_x, bl_y)
            else:
                self.ax.text(0.5, 0.05, "Bilinear solution not well-defined for this curve",
                             ha="center", transform=self.ax.transAxes, color="red", fontsize=9)
                self._current_xy = (d["grid"], d["avg_y"])
            self.ax.set_xlim(left=0)
            self.ax.set_ylim(bottom=0)
            self.ax.set_title(f"Bilinear Idealization (Average Curve): {stem}")

        elif plot_type in ("Stiffness Degradation (per Cycle)",
                            "Energy Dissipation per Loop",
                            "Cumulative Energy Dissipation"):
            cm = d.get("cycle_metrics")
            if cm is None:
                self.ax.text(0.5, 0.5, "Could not identify individual load cycles for this file\n"
                                        "(need at least 2 positive displacement peaks).",
                             ha="center", va="center", fontsize=10, color="gray",
                             transform=self.ax.transAxes)
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self._current_xy = (np.array([]), np.array([]))
            else:
                if plot_type == "Stiffness Degradation (per Cycle)":
                    self.ax.bar(cm["cycle_number"], cm["stiffness_kN_per_mm"],
                                color="#2f8fd6", edgecolor="black")
                    self.ax.set_xlabel("Cycle number")
                    self.ax.set_ylabel("Secant stiffness (kN/mm)")
                    self.ax.set_title(f"Stiffness Degradation per Cycle: {stem}")
                    self._current_xy = (cm["cycle_number"].astype(float), cm["stiffness_kN_per_mm"])
                elif plot_type == "Energy Dissipation per Loop":
                    self.ax.bar(cm["cycle_number"], cm["energy_kN_mm"],
                                color="#e08a2f", edgecolor="black")
                    self.ax.set_xlabel("Cycle number")
                    self.ax.set_ylabel("Dissipated energy per loop (kN\u00b7mm)")
                    self.ax.set_title(f"Energy Dissipation per Loop: {stem}")
                    self._current_xy = (cm["cycle_number"].astype(float), cm["energy_kN_mm"])
                else:  # Cumulative Energy Dissipation
                    self.ax.bar(cm["cycle_number"], cm["cumulative_energy_kN_mm"],
                                color="#3d9c6c", edgecolor="black")
                    self.ax.set_xlabel("Cycle number")
                    self.ax.set_ylabel("Cumulative dissipated energy (kN\u00b7mm)")
                    self.ax.set_title(f"Cumulative Energy Dissipation: {stem}")
                    self._current_xy = (cm["cycle_number"].astype(float), cm["cumulative_energy_kN_mm"])
                if self.grid_var.get():
                    self.ax.grid(alpha=0.3, axis="y")
                self.fig.tight_layout()
                self.canvas.draw_idle()
                return

        self.ax.axhline(0, color="black", linewidth=0.6)
        self.ax.axvline(0, color="black", linewidth=0.6)
        self.ax.set_xlabel("Displacement (mm)")
        self.ax.set_ylabel("Force (kN)")
        if self.legend_var.get():
            self.ax.legend(fontsize=8)
        if self.grid_var.get():
            self.ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # Shutdown / Persist Settings
    def _on_close(self):
        try:
            self.settings["last_input_dir"] = self.input_dir.get().strip()
            self.settings["last_output_dir"] = self.output_dir.get().strip()
            self.settings["window_geometry"] = self.geometry()
            self.settings["plot_style"] = self.style_var.get()
            self.settings["line_width"] = self.linewidth_var.get()
            self.settings["marker_size"] = self.markersize_var.get()
            self.settings["show_grid"] = self.grid_var.get()
            self.settings["show_legend"] = self.legend_var.get()
            self.settings["show_markers"] = self.markers_var.get()
            self.settings["crosshair_enabled"] = self.crosshair_var.get()
            self.settings["export_dpi"] = self.dpi_var.get()
            save_settings(self.settings)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = BackboneGUI()
    app.mainloop()
