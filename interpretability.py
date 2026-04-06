#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-impact post-processing for descriptors.csv

Improvements:
- High DPI (600)
- Large and bold fonts
- Thicker axes and ticks
- PNG + PDF + SVG export
- NO regression fitting in scatter plots
- Figure 01 uses default matplotlib colors
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

INPUT_CSV = "descriptors.csv"
OUT_DIR = "postprocessing"

DPI = 600
EXPORT_PDF = True
EXPORT_SVG = True

BASE_FONT = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 15

LINEWIDTH = 2.2
SPINEWIDTH = 2.0
MARKER_SIZE = 70


# =========================
# GLOBAL STYLE
# =========================

mpl.rcParams.update({

    "figure.dpi": DPI,
    "savefig.dpi": DPI,

    "font.family": "DejaVu Sans",
    "font.size": BASE_FONT,
    "font.weight": "bold",

    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": LABEL_SIZE,
    "axes.labelweight": "bold",
    "axes.linewidth": SPINEWIDTH,

    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,

    "xtick.major.width": LINEWIDTH,
    "ytick.major.width": LINEWIDTH,

    "legend.fontsize": LEGEND_SIZE,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


INTERPRETABLE_COLS = [
    "pR5","pR6","pR7","pR8plus",
    "topo_entropy",
    "mean_q","mean_abs_q","var_q",
    "mean_area","median_area","area_std","area_cv",
    "area_p25","area_p75",
    "frac_area_Rge10","max_area",
    "frac_adj_5_7","frac_adj_defect_defect",
    "n_faces",
    "bond_d0","bond_max",
]

RING_FRAC_COLS = ["pR5","pR6","pR7","pR8plus"]


LABELS = {
    "pR5": r"$p_{R5}$",
    "pR6": r"$p_{R6}$",
    "pR7": r"$p_{R7}$",
    "pR8plus": "p(R>=8)",
    "topo_entropy": r"$H_{topo}$",
    "mean_q": r"$\langle q \rangle$",
    "mean_abs_q": r"$\langle |q| \rangle$",
    "var_q": r"$Var(q)$",
    "mean_area": r"$\langle A \rangle$",
    "median_area": r"$A_{med}$",
    "area_std": r"$\sigma_A$",
    "area_cv": r"$CV(A)$",
    "area_p25": r"$A_{25\%}$",
    "area_p75": r"$A_{75\%}$",
    "frac_area_Rge10": "Frac area (R>=10)",
    "max_area": r"$A_{max}$",
    "frac_adj_5_7": r"$f_{adj}(5,7)$",
    "frac_adj_defect_defect": r"$f_{adj}(def,def)$",
    "n_faces": r"$N_{faces}$",
    "bond_d0": r"$d_0$ (Å)",
    "bond_max": r"$d_{max}$ (Å)",
}


# =========================
# HELPERS
# =========================

def numeric_key(s: str):
    m = re.match(r"(\d+)", str(s))
    if m:
        return (0, int(m.group(1)))
    return (1, str(s))


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def apply_axis_style(ax):

    for spine in ax.spines.values():
        spine.set_linewidth(SPINEWIDTH)

    ax.tick_params(width=LINEWIDTH, length=7)

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def robust_savefig(path):

    fig = plt.gcf()

    try:
        fig.tight_layout()
    except:
        pass

    fig.savefig(path + ".png", dpi=DPI)

    if EXPORT_PDF:
        fig.savefig(path + ".pdf")

    if EXPORT_SVG:
        fig.savefig(path + ".svg")

    plt.close()


def require_columns(df, cols):

    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing columns:\n" + "\n".join(missing)
        )


# =========================
# PLOTS
# =========================

def plot_ring_stacked(df_plot, outbase, title):

    require_columns(df_plot, ["system"] + RING_FRAC_COLS)

    systems = df_plot["system"].astype(str).tolist()

    x = np.arange(len(systems))

    bottom = np.zeros(len(systems))

    fig, ax = plt.subplots(figsize=(max(14, 0.4*len(systems)),6))

    for col in RING_FRAC_COLS:

        vals = df_plot[col].values

        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=LABELS[col],
            edgecolor="black",
        )

        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=90)

    ax.set_ylabel("Fraction of faces")

    ax.set_ylim(0,1.02)

    ax.set_title(title)

    apply_axis_style(ax)

    ax.legend(loc="upper left", bbox_to_anchor=(1.01,1))

    robust_savefig(outbase)


def plot_scatter(df_plot, xcol, ycol, outbase, title, xlabel=None, ylabel=None):

    require_columns(df_plot, [xcol,ycol])

    fig, ax = plt.subplots(figsize=(7,6))

    ax.scatter(
        df_plot[xcol],
        df_plot[ycol],
        s=MARKER_SIZE,
        edgecolor="black"
    )

    ax.set_xlabel(xlabel if xlabel else LABELS.get(xcol,xcol))
    ax.set_ylabel(ylabel if ylabel else LABELS.get(ycol,ycol))

    ax.set_title(title)

    apply_axis_style(ax)

    robust_savefig(outbase)


def plot_boxplots(df_plot, cols, outbase, title):

    cols=[c for c in cols if c in df_plot.columns]

    data=[df_plot[c].dropna().values for c in cols]

    fig, ax = plt.subplots(figsize=(11,6))

    ax.boxplot(
        data,
        showfliers=True
    )

    ax.set_xticklabels(
        [LABELS.get(c,c) for c in cols],
        rotation=25
    )

    ax.set_ylabel("Value")

    ax.set_title(title)

    apply_axis_style(ax)

    robust_savefig(outbase)


def plot_hist_1d(df_plot,col,outbase,title,bins=30):

    require_columns(df_plot,[col])

    fig, ax = plt.subplots(figsize=(7,5))

    ax.hist(
        df_plot[col].dropna(),
        bins=bins,
        edgecolor="black"
    )

    ax.set_xlabel(LABELS.get(col,col))

    ax.set_ylabel("Count")

    ax.set_title(title)

    apply_axis_style(ax)

    robust_savefig(outbase)


def plot_dual_hist(df_plot,col1,col2,outbase,title,bins=30):

    require_columns(df_plot,[col1,col2])

    fig, ax = plt.subplots(figsize=(10,6))

    ax.hist(
        df_plot[col1].dropna(),
        bins=bins,
        alpha=0.7,
        label=LABELS.get(col1,col1),
        edgecolor="black"
    )

    ax.hist(
        df_plot[col2].dropna(),
        bins=bins,
        alpha=0.7,
        label=LABELS.get(col2,col2),
        edgecolor="black"
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

    ax.set_title(title)

    ax.legend()

    apply_axis_style(ax)

    robust_savefig(outbase)


def plot_corr_heatmap(df_plot,cols,outbase,title):

    require_columns(df_plot,cols)

    corr=df_plot[cols].corr()

    fig, ax = plt.subplots(figsize=(9,8))

    im=ax.imshow(corr.values,vmin=-1,vmax=1)

    cbar=fig.colorbar(im)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))

    ax.set_xticklabels([LABELS.get(c,c) for c in cols],rotation=45,ha="right")
    ax.set_yticklabels([LABELS.get(c,c) for c in cols])

    ax.set_title(title)

    apply_axis_style(ax)

    robust_savefig(outbase)


# =========================
# MAIN
# =========================

def main():

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    safe_mkdir(OUT_DIR)

    df=pd.read_csv(INPUT_CSV)

    require_columns(df,INTERPRETABLE_COLS)

    df=df.sort_values(
        by="system",
        key=lambda s:s.map(numeric_key)
    )

    for c in INTERPRETABLE_COLS:
        df[c]=pd.to_numeric(df[c],errors="coerce")

    df_interpret=df[["system"]+INTERPRETABLE_COLS]

    df_interpret.to_csv(
        os.path.join(OUT_DIR,"interpretable_per_system.csv"),
        index=False
    )

    stats=df_interpret[INTERPRETABLE_COLS].agg(["mean","std","min","median","max"]).T

    stats.to_csv(
        os.path.join(OUT_DIR,"dataset_stats_interpretable.csv")
    )

    dataset_mean=df_interpret[INTERPRETABLE_COLS].mean().to_frame("dataset_mean")

    dataset_mean.to_csv(
        os.path.join(OUT_DIR,"dataset_mean_interpretable.csv")
    )

    # FIGURES

    plot_ring_stacked(
        df_interpret,
        os.path.join(OUT_DIR,"fig01_ring_fractions_stacked_all"),
        "Ring-size fractions per system"
    )

    plot_scatter(
        df_interpret,
        "topo_entropy",
        "mean_abs_q",
        os.path.join(OUT_DIR,"fig02_entropy_vs_absq"),
        "Topology vs defect intensity"
    )

    plot_scatter(
        df_interpret,
        "mean_area",
        "max_area",
        os.path.join(OUT_DIR,"fig03_meanarea_vs_maxarea"),
        "Ring area vs largest ring area"
    )

    plot_boxplots(
        df_interpret,
        ["topo_entropy","mean_abs_q","area_cv","frac_area_Rge10"],
        os.path.join(OUT_DIR,"fig04_boxplots"),
        "Dataset distribution of key features"
    )

    plot_dual_hist(
        df_interpret,
        "bond_d0",
        "bond_max",
        os.path.join(OUT_DIR,"fig05_bond_histograms"),
        "Bond inference sanity check"
    )

    plot_corr_heatmap(
        df_interpret,
        ["pR5","pR6","pR7","pR8plus","topo_entropy","mean_abs_q"],
        os.path.join(OUT_DIR,"fig06_correlation"),
        "Correlation matrix"
    )

    plot_hist_1d(
        df_interpret,
        "topo_entropy",
        os.path.join(OUT_DIR,"fig07_hist_entropy"),
        "Histogram: topological entropy"
    )

    plot_hist_1d(
        df_interpret,
        "mean_abs_q",
        os.path.join(OUT_DIR,"fig08_hist_absq"),
        "Histogram: mean |q|"
    )

    print("\nPost-processing complete")


if __name__ == "__main__":
    main()
