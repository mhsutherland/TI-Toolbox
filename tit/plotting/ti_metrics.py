"""
Simple TI-Toolbox metric visualizations (matplotlib).

Centralizes plots that were previously implemented in optimizer/analyzer modules.
"""

from pathlib import Path
from typing import Sequence

from ._common import SaveFigOptions, ensure_headless_matplotlib_backend, savefig_close


def plot_montage_distributions(
    *,
    timax_values: Sequence[float],
    timean_values: Sequence[float],
    focality_values: Sequence[float],
    output_file: str,
    dpi: int = 300,
) -> str | None:
    """
    Create 3 side-by-side histograms for TImax, TImean and Focality distributions.
    """
    if (not timax_values) and (not timean_values) and (not focality_values):
        return None

    ensure_headless_matplotlib_backend()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    configs = [
        (timax_values, axes[0], "TImax (V/m)", "TImax Distribution", "#2196F3"),
        (timean_values, axes[1], "TImean (V/m)", "TImean Distribution", "#4CAF50"),
        (focality_values, axes[2], "Focality", "Focality Distribution", "#FF9800"),
    ]

    for values, ax, xlabel, title, color in configs:
        if values:
            ax.hist(values, bins=20, color=color, edgecolor="black", alpha=0.7)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return savefig_close(fig, output_file, opts=SaveFigOptions(dpi=dpi))


def plot_intensity_vs_focality(
    *,
    intensity: Sequence[float],
    focality: Sequence[float],
    composite: Sequence[float] | None,
    output_file: str,
    dpi: int = 300,
) -> str | None:
    """
    Scatter plot of intensity vs focality, optionally colored by composite index.
    """
    if (not intensity) or (not focality):
        return None

    ensure_headless_matplotlib_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    if composite and any(c is not None for c in composite):
        sc = ax.scatter(
            intensity,
            focality,
            c=composite,
            cmap="viridis",
            s=40,
            edgecolor="black",
            alpha=0.7,
        )
        fig.colorbar(sc, ax=ax).set_label("Composite Index", fontsize=12)
    else:
        ax.scatter(intensity, focality, s=40, edgecolor="black", alpha=0.7)

    ax.set_xlabel("TImean_ROI (V/m)", fontsize=12)
    ax.set_ylabel("Focality", fontsize=12)
    ax.set_title("Intensity vs Focality", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return savefig_close(fig, output_file, opts=SaveFigOptions(dpi=dpi))


def plot_top_montages(
    *,
    points: Sequence[dict],
    selected: Sequence[dict],
    min_intensity: float,
    separation_applied: bool,
    output_file: str,
    dpi: int = 300,
) -> str | None:
    """Intensity-vs-focality scatter with the selected montages called out.

    The companion figure to ``intensity_vs_focality_scatter.png``, which stays a
    plain view of every evaluation.  Here the same cloud is split by the rules
    :mod:`tit.opt.ex.ranking` applies, so a run's picks can be checked against
    what they were picked out of: hollow markers are montages rejected for
    having electrodes of different channels too close together, faint filled
    markers fall below the ``TImean_ROI`` floor, and the picks are numbered in
    rank order.

    Parameters
    ----------
    points : sequence of dict
        One entry per evaluation, with ``timean``, ``focality``,
        ``passes_intensity`` and ``passes_separation``.
    selected : sequence of dict
        The picks, each with ``rank``, ``timean`` and ``focality``.
    min_intensity : float
        The ``TImean_ROI`` floor, drawn as a vertical line.
    separation_applied : bool
        Whether electrode coordinates were available.  When false the
        separation legend entry is dropped rather than shown as all-passing.
    output_file : str
        Destination PNG.
    dpi : int, default 300
        Resolution.

    Returns
    -------
    str or None
        The written path, or ``None`` when there is nothing to plot.
    """
    if not points:
        return None

    ensure_headless_matplotlib_backend()
    import matplotlib.pyplot as plt

    below = [p for p in points if not p["passes_intensity"]]
    too_close = [
        p for p in points if p["passes_intensity"] and not p["passes_separation"]
    ]
    eligible = [p for p in points if p["passes_intensity"] and p["passes_separation"]]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    if below:
        ax.scatter(
            [p["timean"] for p in below],
            [p["focality"] for p in below],
            s=26,
            color="#BDBDBD",
            alpha=0.5,
            linewidths=0,
            label=f"below {min_intensity:g} V/m (n={len(below)})",
        )
    if too_close:
        ax.scatter(
            [p["timean"] for p in too_close],
            [p["focality"] for p in too_close],
            s=34,
            facecolors="none",
            edgecolors="#E57373",
            linewidths=0.9,
            label=f"channels too close (n={len(too_close)})",
        )
    if eligible:
        ax.scatter(
            [p["timean"] for p in eligible],
            [p["focality"] for p in eligible],
            s=34,
            color="#1976D2",
            alpha=0.55,
            edgecolor="black",
            linewidths=0.3,
            label=f"eligible (n={len(eligible)})",
        )
    if selected:
        ax.scatter(
            [item["timean"] for item in selected],
            [item["focality"] for item in selected],
            s=150,
            marker="*",
            color="#FF9800",
            edgecolor="black",
            linewidths=0.7,
            zorder=5,
            label=f"selected (n={len(selected)})",
        )
        # The picks sit close together at the top of the cloud -- offset labels
        # would overlap each other, so the rank goes inside its own star.
        for item in selected:
            ax.annotate(
                str(item["rank"]),
                (item["timean"], item["focality"]),
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="#212121",
                zorder=6,
            )

    ax.axvline(
        min_intensity,
        color="#424242",
        linestyle="--",
        linewidth=1.0,
        label=f"TImean_ROI = {min_intensity:g} V/m",
    )

    ax.set_xlabel("TImean_ROI (V/m)", fontsize=12)
    ax.set_ylabel("Focality (TImean_ROI / TImean_GM)", fontsize=12)
    title = (
        "Top montages"
        if separation_applied
        else "Top montages (no electrode coordinates)"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    return savefig_close(fig, output_file, opts=SaveFigOptions(dpi=dpi))


# Channel colours for the per-montage maps. Slots 1, 2, 3 and 7 of the data-viz
# categorical palette (blue, orange, aqua, violet), which is the widest set of
# four that clears the *all-pairs* separation gates rather than the adjacent-pair
# ones -- every channel of a montage is on screen at once, so any two of them can
# end up side by side. Validated at worst all-pairs CVD dE 9.2 and normal-vision
# dE 16.3 on a light surface; aqua sits at 2.74:1 contrast, below the 3:1 line,
# which the rule covers with mandatory relief -- here the direct electrode labels
# and the legend, so no channel is identified by colour alone.
_CHANNEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")


def plot_top_montage_maps(
    *,
    eeg_positions_csv: str,
    selected: Sequence[dict],
    output_file: str,
    dpi: int = 200,
    title_prefix: str = "Ex-Search",
) -> str | None:
    """Draw each selected montage on its own page of a multi-page PDF.

    The companion to ``montage_strength_map.png``, which stacks 150 montages on
    one cap and answers "where does the search keep going?".  This answers the
    other question -- "what exactly would I place on the head for pick 3?" -- so
    each page carries one montage only: its channels as arcs in the channel
    colours, anodes filled and cathodes hollow, every electrode labelled, and
    the montage's own numbers in the title.

    A PDF rather than five PNGs or one grid: the desktop Results pane rasterises
    a PDF into the same tile a PNG gets and pages a multi-page one with prev/next
    (``desktop/.../preview/PdfCanvas.tsx``), so one artifact stays one artifact
    and each map still gets a full page to be read at.

    Parameters
    ----------
    eeg_positions_csv : str
        Any EEG-position CSV; a known net name resolves to the bundled head
        template, anything else is drawn as a plain RAS x/y scatter
        (:func:`_resolve_layout`).
    selected : sequence of dict
        The picks from :func:`tit.opt.ex.ranking.select_top_montages`, each with
        ``rank``, ``electrodes``, ``focality``, ``timean`` and
        ``min_separation_mm``.
    output_file : str
        Destination ``.pdf``.
    dpi : int, default 200
        Rasterisation resolution of the head template inside the PDF; the
        vector overlay is unaffected.
    title_prefix : str, default "Ex-Search"
        ``"mTI Search"`` for a four-channel run.

    Returns
    -------
    str or None
        The written path, or ``None`` when there is nothing plottable.
    """
    layout = _resolve_layout(eeg_positions_csv)
    positions = layout["positions"]
    if not positions or not selected:
        return None

    plottable = [
        item
        for item in selected
        if len(_montage_electrodes(item)) >= 2
        and all(e in positions for e in _montage_electrodes(item))
    ]
    if not plottable:
        return None

    net_name = Path(layout["eeg_net_name"]).stem
    # Labels are pushed radially outward from the middle of the cap rather than
    # always down-right: the montages worth looking hardest at are the ones with
    # two channels on neighbouring electrodes, and a fixed offset stacks those
    # two labels on top of each other. Outward also keeps them off the dense
    # centre of the template.
    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]
    centre = (sum(xs) / len(xs), sum(ys) / len(ys))

    ensure_headless_matplotlib_backend()
    import matplotlib as mpl
    import matplotlib.image as mpimg
    import matplotlib.lines  # noqa: F401  (attribute access below)
    import matplotlib.patches  # noqa: F401
    import matplotlib.path  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Not savefig_close(): every page goes into one open PdfPages, so the pages
    # are saved through it and closed here instead.
    with PdfPages(output_file) as pdf:
        for item in plottable:
            fig, ax = plt.subplots(figsize=(11, 9))
            label_offset = _draw_layout_background(ax, layout, mpimg)
            handles = []

            pairs = _montage_pairs(_montage_electrodes(item))
            for index, (plus, minus) in enumerate(pairs):
                color = _CHANNEL_COLORS[index % len(_CHANNEL_COLORS)]
                _draw_pair_curve(
                    ax,
                    positions[plus],
                    positions[minus],
                    color=color,
                    alpha=0.95,
                    linewidth=3.0,
                    curve_side=1 if index % 2 == 0 else -1,
                    mpl=mpl,
                )
                for label, is_anode in ((plus, True), (minus, False)):
                    x, y = positions[label]
                    # Hoisted out of the f-string below: a backslash escape
                    # inside an f-string expression needs Python 3.12, and the
                    # container runs 3.11.
                    polarity = "+" if is_anode else "\u2212"
                    # A surface ring under the marker rather than a stroke
                    # around it: these sit on top of the printed electrode
                    # circles of the template.
                    ax.scatter([x], [y], s=430, color="white", zorder=4)
                    ax.scatter(
                        [x],
                        [y],
                        s=330,
                        facecolors=color if is_anode else "white",
                        edgecolors=color,
                        linewidths=3,
                        zorder=5,
                    )
                    dx, dy = x - centre[0], y - centre[1]
                    radius = (dx**2 + dy**2) ** 0.5
                    if radius < 1e-6:
                        ux, uy = 1.0, -1.0
                    else:
                        ux, uy = dx / radius, dy / radius
                    ax.text(
                        x + ux * label_offset * 1.6,
                        y + uy * label_offset * 1.6,
                        f"{label} {polarity}",
                        color="#212121",
                        fontsize=11,
                        weight="bold",
                        ha="left" if ux >= 0 else "right",
                        va="center",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.7,
                            "pad": 1,
                        },
                        zorder=6,
                    )
                handles.append(
                    mpl.lines.Line2D(
                        [],
                        [],
                        color=color,
                        linewidth=3.0,
                        label=f"Ch{index + 1}   {plus} +  \u2192  {minus} \u2212",
                    )
                )

            # Polarity is carried by marker fill as well as by the labels, so it
            # survives a greyscale print.
            handles += [
                mpl.lines.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=10,
                    markerfacecolor="#555555",
                    markeredgecolor="#555555",
                    label="anode (+)",
                ),
                mpl.lines.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=10,
                    markerfacecolor="white",
                    markeredgecolor="#555555",
                    markeredgewidth=2,
                    label="cathode \u2212",
                ),
            ]
            legend = ax.legend(
                handles=handles,
                loc="lower left",
                fontsize=11,
                framealpha=0.9,
                labelcolor="#212121",
            )
            legend.set_zorder(7)

            ax.set_title(
                f"{title_prefix} pick {item.get('rank', '?')} of {len(plottable)}"
                f"  \u00b7  {net_name}\n{_map_subtitle(item)}",
                fontsize=15,
                pad=12,
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        pdf.infodict()["Title"] = f"{title_prefix} top {len(plottable)} montages"

    return output_file


def _map_subtitle(item: dict) -> str:
    """The metric line under a montage-map title."""
    parts = []
    for key, fmt in (
        ("focality", "focality {:.4f}"),
        ("timean", "TImean_ROI {:.4f} V/m"),
    ):
        try:
            parts.append(fmt.format(float(item[key])))
        except (KeyError, TypeError, ValueError):
            continue
    separation = item.get("min_separation_mm")
    if separation is not None:
        parts.append(f"closest cross-channel pair {float(separation):.1f} mm")
    return "  \u00b7  ".join(parts)


# ── Electrode-map figures (ported from Larissa Albantakis's branch) ──────────
#
# Both maps draw on a 2-D electrode layout resolved by ``_resolve_layout``:
# known template nets (GSN, 10-10 families) use the bundled head template
# image + pixel coordinates in ``resources/amv``; any other EEG-position CSV
# is drawn as a plain x/y (SimNIBS RAS) scatter.  A montage record is a dict
# with ``electrodes`` (4 or 8 labels, consecutive pairs) and the metrics
# ``composite``, ``timean``, ``focality``.

_AMV_DIR = Path(__file__).resolve().parents[2] / "resources" / "amv"
_TEMPLATE_COORD_FILES = {
    "GSN-HydroCel-185": "GSN-256.csv",
    "GSN-HydroCel-256": "GSN-256.csv",
    "GSN-256": "GSN-256.csv",
    "EEG10-10_UI_Jurak_2007": "10-10.csv",
    "EEG10-10_Cutini_2011": "10-10.csv",
    "EEG10-20_Okamoto_2004": "10-10.csv",
    "EEG10-10_Neuroelectrics": "10-10.csv",
    "10-10": "10-10.csv",
}
_TEMPLATE_IMAGE = "GSN-256.png"
_MONTAGE_KEYS = ("e1_plus", "e1_minus", "e2_plus", "e2_minus")


def _montage_electrodes(item: dict) -> list[str]:
    """Electrode labels of a montage record (``electrodes`` or ``e1_plus``...)."""
    electrodes = item.get("electrodes")
    if electrodes:
        return [str(e) for e in electrodes]
    return [item[key] for key in _MONTAGE_KEYS if key in item]


def _montage_pairs(electrodes: Sequence[str]) -> list[tuple[str, str]]:
    from tit.opt.ex.buckets import electrode_pairs

    return electrode_pairs(electrodes)


def _load_eeg_position_csv(path: str) -> dict[str, tuple[float, ...]]:
    """Read ``label -> (x, y[, z])`` from any supported EEG-position CSV layout."""
    from tit.opt.ex.buckets import read_eeg_positions

    return read_eeg_positions(path)


def _load_template_projection(path: str) -> dict | None:
    """Template pixel coordinates + head image for known net names, else None."""
    name = Path(path).name
    coord_file = _TEMPLATE_COORD_FILES.get(name) or _TEMPLATE_COORD_FILES.get(
        Path(name).stem
    )
    if coord_file is None:
        return None
    coord_path = _AMV_DIR / coord_file
    template_path = _AMV_DIR / _TEMPLATE_IMAGE
    if not coord_path.is_file() or not template_path.is_file():
        return None
    positions = {
        label: (xy[0], xy[1])
        for label, xy in _load_eeg_position_csv(str(coord_path)).items()
    }
    return {
        "positions": positions,
        "template_path": template_path,
        "eeg_net_name": name,
    }


def _resolve_layout(eeg_positions_csv: str) -> dict:
    """Return ``{"positions", "template_path" (or None), "eeg_net_name"}``."""
    template = _load_template_projection(eeg_positions_csv)
    if template:
        return template
    positions = {
        label: (xyz[0], xyz[1])
        for label, xyz in _load_eeg_position_csv(eeg_positions_csv).items()
    }
    return {
        "positions": positions,
        "template_path": None,
        "eeg_net_name": Path(eeg_positions_csv).name,
    }


def _rank_plottable_montages(
    *,
    montage_scores: Sequence[dict],
    positions: dict,
    metric_key: str,
    top_n: int,
) -> list[dict]:
    plottable = []
    for item in montage_scores:
        electrodes = _montage_electrodes(item)
        if len(electrodes) < 2 or not all(e in positions for e in electrodes):
            continue
        try:
            float(item[metric_key])
        except (KeyError, TypeError, ValueError):
            continue
        plottable.append(item)
    return sorted(plottable, key=lambda item: float(item[metric_key]), reverse=True)[
        :top_n
    ]


def _get_montage_metric_cmap(plt, mpl, cmap_name: str | tuple[str, str]):
    if isinstance(cmap_name, tuple):
        return mpl.colors.LinearSegmentedColormap.from_list(
            "montage_metric_cmap", list(cmap_name)
        )
    return plt.get_cmap(cmap_name)


def _draw_layout_background(ax, layout: dict, mpimg) -> float:
    """Draw the template head image or a grey electrode scatter.

    Returns the label offset (in data units) appropriate for the layout.
    """
    positions = layout["positions"]
    if layout["template_path"] is not None:
        ax.imshow(
            mpimg.imread(layout["template_path"]), cmap="gray", vmin=0, vmax=1, zorder=0
        )
        ax.set_axis_off()
        return 18.0
    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]
    ax.scatter(xs, ys, s=8, color="#d0d0d0", edgecolors="none", zorder=1)
    ax.set_xlabel("Left / Right (SimNIBS RAS x)")
    ax.set_ylabel("Posterior / Anterior (SimNIBS RAS y)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    return 2.0


def _draw_pair_curve(ax, start, end, *, color, alpha, linewidth, curve_side, mpl):
    """Draw one electrode pair as a quadratic curve bowed to ``curve_side``."""
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    dist = (dx**2 + dy**2) ** 0.5
    if dist == 0:
        return
    bow = min(dist * 0.22, 55)
    control = (
        (x1 + x2) / 2 + curve_side * (-dy / dist) * bow,
        (y1 + y2) / 2 + curve_side * (dx / dist) * bow,
    )
    path_cls = mpl.path.Path
    path = path_cls(
        [(x1, y1), control, (x2, y2)],
        [path_cls.MOVETO, path_cls.CURVE3, path_cls.CURVE3],
    )
    ax.add_patch(
        mpl.patches.PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
    )


def plot_montage_score_map(
    *,
    eeg_positions_csv: str,
    montage_scores: Sequence[dict],
    output_file: str,
    top_n: int = 50,
    dpi: int = 300,
    metric_key: str = "composite",
    metric_label: str = "Composite Index (TImean_ROI x Focality)",
    title_metric: str = "Composite Score",
    cmap_name: str | tuple[str, str] = "cividis",
    title_prefix: str = "Ex-Search",
) -> str | None:
    """Draw the top-``top_n`` montages as electrode-pair curves on the EEG layout.

    Every ``montage_scores`` record needs ``electrodes`` (4 or 8 labels as
    consecutive pairs; the legacy ``e1_plus``..``e2_minus`` keys are also
    accepted) and the requested ``metric_key``.  Curves are coloured by the
    metric; the best montage's electrodes are highlighted and labelled.
    """
    layout = _resolve_layout(eeg_positions_csv)
    positions = layout["positions"]
    if not positions or not montage_scores:
        return None
    ranked = _rank_plottable_montages(
        montage_scores=montage_scores,
        positions=positions,
        metric_key=metric_key,
        top_n=top_n,
    )
    if not ranked:
        return None

    ensure_headless_matplotlib_backend()
    import matplotlib as mpl
    import matplotlib.image as mpimg
    import matplotlib.patches  # noqa: F401  (attribute access below)
    import matplotlib.path  # noqa: F401
    import matplotlib.pyplot as plt

    values = [float(item[metric_key]) for item in ranked]
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        vmax = vmin + 1e-12
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = _get_montage_metric_cmap(plt, mpl, cmap_name)

    fig, ax = plt.subplots(figsize=(11, 9))
    label_offset = _draw_layout_background(ax, layout, mpimg)

    for rank, item in enumerate(reversed(ranked), 1):
        color = cmap(norm(float(item[metric_key])))
        alpha = 0.18 + 0.62 * rank / len(ranked)
        linewidth = 1.0 + 3.0 * rank / len(ranked)
        for pair_idx, (a, b) in enumerate(_montage_pairs(_montage_electrodes(item))):
            _draw_pair_curve(
                ax,
                positions[a],
                positions[b],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                curve_side=1 if pair_idx % 2 == 0 else -1,
                mpl=mpl,
            )

    best = ranked[0]
    for label in _montage_electrodes(best):
        x, y = positions[label]
        ax.scatter(
            [x],
            [y],
            s=430,
            facecolors="none",
            edgecolors="#ffea00",
            linewidths=4,
            zorder=5,
        )
        ax.text(
            x + label_offset,
            y - label_offset,
            label,
            color="black",
            fontsize=11,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1},
            zorder=6,
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01).set_label(metric_label)
    ax.set_title(
        f"Top {len(ranked)} {title_prefix} Montages by {title_metric} "
        f"({layout['eeg_net_name']})",
        fontsize=16,
        pad=12,
    )
    fig.tight_layout()
    return savefig_close(fig, output_file, opts=SaveFigOptions(dpi=dpi))


def plot_electrode_score_heatmap(
    *,
    eeg_positions_csv: str,
    montage_scores: Sequence[dict],
    output_file: str,
    top_n: int = 50,
    dpi: int = 300,
    title_prefix: str = "Ex-Search",
) -> str | None:
    """Plot electrode participation in the top-``top_n`` montages.

    Each electrode receives the sum of the composite index of the top
    montages it appears in (colour) and its frequency (marker size); the
    12 highest-scoring electrodes are labelled.
    """
    layout = _resolve_layout(eeg_positions_csv)
    positions = layout["positions"]
    if not positions or not montage_scores:
        return None

    ranked = _rank_plottable_montages(
        montage_scores=montage_scores,
        positions=positions,
        metric_key="composite",
        top_n=top_n,
    )
    if not ranked:
        return None

    scores = {label: 0.0 for label in positions}
    counts = {label: 0 for label in positions}
    for item in ranked:
        composite = float(item["composite"])
        for label in _montage_electrodes(item):
            scores[label] += composite
            counts[label] += 1
    active = [label for label, count in counts.items() if count > 0]

    ensure_headless_matplotlib_backend()
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9))
    label_offset = _draw_layout_background(ax, layout, mpimg)
    if layout["template_path"] is not None:
        inactive = [label for label in positions if label not in active]
        ax.scatter(
            [positions[label][0] for label in inactive],
            [positions[label][1] for label in inactive],
            s=12,
            color="#b0b0b0",
            edgecolors="none",
            zorder=1,
        )

    size_scale = 60 if layout["template_path"] is not None else 22
    sc = ax.scatter(
        [positions[label][0] for label in active],
        [positions[label][1] for label in active],
        c=[scores[label] for label in active],
        s=[60 + size_scale * counts[label] for label in active],
        cmap="inferno",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    for label in sorted(active, key=lambda lb: scores[lb], reverse=True)[:12]:
        x, y = positions[label]
        ax.text(
            x + label_offset,
            y - label_offset,
            label,
            fontsize=10,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1},
            zorder=4,
        )

    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label(
        "Summed Composite Index Across Top Montages"
    )
    ax.set_title(
        f"{title_prefix} Electrode Contribution (Top {len(ranked)} Montages, "
        f"{layout['eeg_net_name']})",
        fontsize=16,
        pad=12,
    )

    freq = [counts[label] for label in active]
    ax.text(
        0.02,
        0.02,
        "Color = summed composite index; marker size = frequency in the top "
        f"montages ({min(freq)}-{max(freq)} of {len(ranked)}).",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
    )
    fig.tight_layout()
    return savefig_close(fig, output_file, opts=SaveFigOptions(dpi=dpi))
