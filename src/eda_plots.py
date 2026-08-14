"""
Exploratory data analysis plots — joint scatter/regression plots for each
input feature vs. compressive strength (Figures 8–17 in the paper).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import seaborn as sns

from .data import Dataset

PLOTS_INFO = {
    8:  ("Alk: Binder",      "Alkaline-to-binder ratio"),
    9:  ("All Binders",      "Binder Content ($kg/m^3$)"),
    10: ("Fly ash",          "Fly Ash (FA) Content ($kg/m^3$)"),
    11: ("Fine Aggregate",   "Fine Aggregate ($kg/m^3$)"),
    12: ("NaOH",             "NaOH (SH) Content ($kg/m^3$)"),
    13: ("Na2SiO3",          "Na$_2$SiO$_3$ (SS) Content ($kg/m^3$)"),
    14: ("Molarity (M)",     "Molarity of NaOH (M)"),
    15: ("Na2SiO3/NaOH",    "Na$_2$SiO$_3$/NaOH ratio"),
    16: ("Nano Silica (Kg)", "Nano-silica (nS) Content ($kg/m^3$)"),
    17: ("Curing Condition", "Curing Temperature ($^\\circ$C)"),
}

Y_LABEL = "Compressive Strength at 28 Days (MPa)"


def _add_counts(g) -> None:
    for p in g.ax_marg_x.patches:
        h = p.get_height()
        if h > 0:
            g.ax_marg_x.text(
                p.get_x() + p.get_width() / 2., h + 0.2,
                f"{int(h)}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
    for p in g.ax_marg_y.patches:
        w = p.get_width()
        if w > 0:
            g.ax_marg_y.text(
                w + 0.2, p.get_y() + p.get_height() / 2.,
                f"{int(w)}", ha="left", va="center", fontsize=8, fontweight="bold",
            )


def plot_eda_joint_plots(dataset: Dataset, output_dir: str = "Plots_Dataset_EDA") -> None:
    """
    Generate joint scatter+regression plots for each input feature vs. target.

    Parameters
    ----------
    dataset    : Dataset object returned by load_data().
    output_dir : Folder where PNG files are saved (created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)

    df = dataset.data.rename(columns=dataset.display_name_map)
    df["All Binders"] = (
        df.get("Fly ash", 0) + df.get("Slag", 0) + df.get("Metakaoline", 0)
    )
    y_col = "Comp.(MPa, 28 days)"

    for fig_num, (x_col, x_label) in PLOTS_INFO.items():
        plot_df = df.dropna(subset=[x_col, y_col])
        if len(plot_df) < 2:
            print(f"Skipping Figure {fig_num}: not enough data for {x_col}")
            continue

        g = sns.jointplot(
            data=plot_df, x=x_col, y=y_col, kind="reg",
            joint_kws={
                "line_kws":    {"color": "red", "linewidth": 1.5},
                "scatter_kws": {"alpha": 0.6, "color": "#2c3e50"},
            },
            marginal_kws={"bins": 10, "fill": True, "color": "#bdc3c7", "edgecolor": "black"},
        )
        g.set_axis_labels(x_label, Y_LABEL, fontsize=10, fontweight="medium", labelpad=10)
        _add_counts(g)
        plt.subplots_adjust(top=0.9, right=0.9, left=0.15, bottom=0.15)
        plt.savefig(f"{output_dir}/Figure_{fig_num}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: Figure_{fig_num}.png  ({x_col})")

    print(f"\nAll dataset joint plots saved to {output_dir}/")
