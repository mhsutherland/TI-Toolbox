"""Top-montage selection for ex-search and m-ex-search runs.

``final_output.csv`` is written in enumeration order, so choosing a montage to
run meant sorting it by ``Focality`` by hand and checking against
``intensity_vs_focality_scatter.png`` whether the focal rows also reach a usable
intensity.  This module encodes that judgement as three rules:

1. **Intensity floor** -- keep only montages with
   ``TImean_ROI >= min_intensity`` (default 0.6 V/m).
2. **Focality first** -- rank the survivors by ``Focality``
   (``TImean_ROI / TImean_GM``) descending, tie-breaking on ``TImean_ROI``.
3. **Electrode separation** -- reject a montage when two electrodes belonging to
   *different* channels sit closer than ``min_separation_mm`` (default 20 mm) on
   the subject's cap.  Two energised electrodes that are immediate neighbours
   force the usable current amplitude down, because participants feel the
   stimulation far more.  The rule mostly bites on multipolar runs (two channel
   pairs per hemisphere); on a 2-pair run the channels are metres apart in
   electrode terms and it is a no-op.

Nothing is relaxed silently: when fewer than ``top_n`` montages qualify the
selection is short, and the counts removed by each filter are written to
``top_montages.json``.

Outputs, written next to ``final_output.csv``
---------------------------------------------
``top_montages.csv``
    The picks, with the source file's own columns plus ``Rank`` and
    ``MinSeparation_mm``.
``top_montages.json``
    Thresholds, filter counts, the resolved coordinate source, and the per-pick
    electrode breakdown.
``top_montages_scatter.png``
    The intensity/focality scatter with the picks called out.
``top_montages_map.pdf``
    One page per pick: that montage alone drawn on the electrode cap.

Public API
----------
read_run_config
    Load a run's ``run_config.json`` (missing file -> ``{}``).
electrode_positions_for_run
    Resolve millimetre electrode coordinates for a run directory.
min_cross_channel_separation
    Smallest distance between two electrodes of different channels
    (grouped with :func:`tit.opt.ex.buckets.electrode_pairs`).
select_top_montages
    Apply the three rules to ``final_output.csv`` records.
write_top_montages
    Write the three artifacts.
rank_run
    One call: resolve, select, write.

Also runnable::

    simnibs_python -m tit.opt.ex.ranking <run_dir> [--min-intensity 0.6] ...

See Also
--------
tit.opt.ex.results.process_and_save : Calls :func:`rank_run` after every search.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .buckets import electrode_pairs, read_eeg_positions
from .results import montage_score_records_from_csv
from .symmetry import net_name_from_leadfield

DEFAULT_MIN_INTENSITY = 0.6
"""V/m floor on ``TImean_ROI``; below this a montage is not worth running."""

DEFAULT_MIN_SEPARATION_MM = 20.0
"""Minimum centre-to-centre distance between electrodes of different channels.

On GSN-HydroCel-256 the median nearest-neighbour spacing is ~17 mm in subject
space and ~18.5 mm on the MNI template, so a 20 mm cutoff excludes a median of
two to four neighbours per electrode, depending on the cap -- the closest part
of the first ring, not all of it.  The full ring reaches ~24 mm, so 25 mm
excludes every immediate neighbour and 35 mm starts on the second ring; both are
available through ``--min-separation-mm`` when a stricter run is wanted.
"""

DEFAULT_TOP_N = 5

_MNI_CAP_DIR = Path(__file__).resolve().parents[3] / "resources" / "ElectrodeCaps_MNI"

TOP_MONTAGES_CSV = "top_montages.csv"
TOP_MONTAGES_JSON = "top_montages.json"
TOP_MONTAGES_PNG = "top_montages_scatter.png"
TOP_MONTAGES_MAP = "top_montages_map.pdf"

_SEPARATION_COLUMN = "MinSeparation_mm"


# ── coordinates ──────────────────────────────────────────────────────────────


def read_run_config(run_dir: str | Path) -> dict:
    """Return a run's ``run_config.json``, or ``{}`` when it is absent/unreadable."""
    path = Path(run_dir) / "run_config.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _subject_cap_candidates(run_dir: Path, net_name: str) -> list[Path]:
    """Subject cap files for *net_name*, searched upward from a run directory.

    A run lives at ``<project>/derivatives/SimNIBS/sub-<id>/ex-search/<run>/``,
    so the subject's ``m2m_*/eeg_positions/`` is two levels up.  The search walks
    up rather than reading ``leadfield_hdf`` from ``run_config.json``, because
    that path is the container-internal ``/mnt/...`` one and does not resolve
    when this runs on the host.
    """
    candidates = []
    for parent in list(run_dir.parents)[:4]:
        for eeg_dir in sorted(parent.glob("m2m_*/eeg_positions")):
            candidate = eeg_dir / f"{net_name}.csv"
            if candidate.is_file():
                candidates.append(candidate)
    return candidates


def electrode_positions_for_run(
    run_dir: str | Path, *, eeg_csv: str | Path | None = None
) -> tuple[dict[str, tuple[float, ...]] | None, Path | None, str]:
    """Resolve millimetre electrode coordinates for a run.

    Parameters
    ----------
    run_dir : str or Path
        Run directory holding ``final_output.csv`` and ``run_config.json``.
    eeg_csv : str or Path, optional
        Explicit cap file; skips the search when given.

    Returns
    -------
    positions : dict or None
        ``{label: (x, y, z)}`` in millimetres, or ``None`` when no cap file was
        found -- in which case the separation rule cannot be applied.
    cap_path : Path or None
        The file the coordinates were read from.  The montage maps need the
        path rather than the coordinates: ``_resolve_layout`` keys off its name
        to find the bundled head template to draw on.
    source : str
        Human-readable description of where the coordinates came from (or why
        there are none), recorded in ``top_montages.json``.

    Notes
    -----
    This deliberately does not call
    :func:`tit.opt.ex.symmetry.resolve_eeg_positions_csv`.  For the GSN nets that
    returns the bundled 2-D *pixel* template under ``resources/amv``, which is a
    flattened cap figure -- distances measured there are not millimetres and not
    even proportional to them.  Only the SimNIBS cap files (subject-space
    ``m2m_*/eeg_positions`` or the ``resources/ElectrodeCaps_MNI`` templates)
    carry usable geometry.
    """
    run_dir = Path(run_dir)

    if eeg_csv is not None:
        path = Path(eeg_csv)
        if not path.is_file():
            raise FileNotFoundError(f"EEG-position CSV not found: {path}")
        positions = read_eeg_positions(path)
        if not positions:
            return (
                None,
                None,
                f"no electrode positions in {path}; separation rule skipped",
            )
        return positions, path, f"explicit: {path}"

    leadfield = read_run_config(run_dir).get("leadfield_hdf")
    net_name = net_name_from_leadfield(leadfield) if leadfield else ""
    if not net_name:
        return (
            None,
            None,
            "no leadfield_hdf in run_config.json; separation rule skipped",
        )

    subject_caps = _subject_cap_candidates(run_dir, net_name)
    if subject_caps:
        positions = read_eeg_positions(subject_caps[0])
        if positions:
            return positions, subject_caps[0], f"subject cap: {subject_caps[0]}"

    template = _MNI_CAP_DIR / f"{net_name}.csv"
    if template.is_file():
        positions = read_eeg_positions(template)
        if positions:
            return positions, template, f"MNI template: {template}"

    return (
        None,
        None,
        (
            f"no cap file for net {net_name!r} (looked for the subject's "
            f"m2m_*/eeg_positions/{net_name}.csv and {template}); "
            "separation rule skipped"
        ),
    )


# ── separation ───────────────────────────────────────────────────────────────


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    return math.dist(a[:n], b[:n])


def min_cross_channel_separation(
    pairs: Sequence[tuple[str, str]], positions: dict[str, tuple[float, ...]]
) -> float | None:
    """Smallest distance between two electrodes belonging to different channels.

    Returns ``None`` when there is no such distance to measure: a montage with
    a single channel, or one whose labels are not all in *positions*.  The
    caller separates those two cases -- a single channel has no constraint to
    violate, an unmeasurable distance is never treated as a pass.

    Straight-line rather than geodesic: at 20 mm on a head of radius ~90 mm the
    chord is within 1% of the arc, and no scalp surface is needed.
    """
    if len(pairs) < 2:
        return None
    best = None
    for index, first in enumerate(pairs):
        for second in pairs[index + 1 :]:
            for a in first:
                for b in second:
                    coord_a, coord_b = positions.get(a), positions.get(b)
                    if coord_a is None or coord_b is None:
                        return None
                    distance = _distance(coord_a, coord_b)
                    if best is None or distance < best:
                        best = distance
    return best


def min_within_channel_separation(
    pairs: Sequence[tuple[str, str]], positions: dict[str, tuple[float, ...]]
) -> float | None:
    """Smallest anode-to-cathode distance within a single channel.

    Reported in the diagnostics but never filtered on: that distance follows
    from the electrode buckets the search was given, not from the ranking.
    """
    best = None
    for plus, minus in pairs:
        coord_a, coord_b = positions.get(plus), positions.get(minus)
        if coord_a is None or coord_b is None:
            return None
        distance = _distance(coord_a, coord_b)
        if best is None or distance < best:
            best = distance
    return best


# ── selection ────────────────────────────────────────────────────────────────


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_top_montages(
    csv_path: str | Path,
    positions: dict[str, tuple[float, ...]] | None,
    *,
    min_intensity: float = DEFAULT_MIN_INTENSITY,
    min_separation_mm: float = DEFAULT_MIN_SEPARATION_MM,
    top_n: int = DEFAULT_TOP_N,
    collapse_current_splits: bool = True,
    min_distinct_electrodes: int | None = None,
) -> dict:
    """Apply the three selection rules to a ``final_output.csv``.

    Parameters
    ----------
    csv_path : str or Path
        The run's ``final_output.csv``.
    positions : dict or None
        Millimetre electrode coordinates from
        :func:`electrode_positions_for_run`.  ``None`` disables the separation
        rule, which is then reported as not applied rather than as passed.
    min_intensity : float, default 0.6
        Inclusive ``TImean_ROI`` floor in V/m.
    min_separation_mm : float, default 20.0
        Inclusive minimum cross-channel electrode distance.
    top_n : int, default 5
        How many montages to select.  Fewer are returned when fewer qualify.
    collapse_current_splits : bool, default True
        Keep only the best-scoring current split of each electrode set, so a
        top-5 cannot degenerate into five current variants of one montage.
    min_distinct_electrodes : int, optional
        When set, a candidate is skipped unless it differs from every
        already-selected montage by at least this many electrodes.  Off by
        default: the ranking stays a plain focality ranking unless asked.

    Returns
    -------
    dict
        ``selected`` (the picks, best first), ``dropped`` (counts per filter),
        ``points`` (every evaluation, for the scatter), ``columns`` (the source
        file's column order) and the run-level counts.
    """
    records = montage_score_records_from_csv(csv_path)
    columns = list(records[0]["row"].keys()) if records else []
    separation_applied = positions is not None

    points: list[dict] = []
    candidates: list[dict] = []
    dropped = {"intensity": 0, "separation": 0, "current_split": 0, "diversity": 0}

    for record in records:
        row = record["row"]
        pairs = electrode_pairs(record["electrodes"])
        separation = (
            min_cross_channel_separation(pairs, positions) if positions else None
        )
        passes_intensity = record["timean"] >= min_intensity
        if not separation_applied or len(pairs) < 2:
            # Nothing to measure: no coordinates, or a single channel, whose two
            # poles are the buckets' business rather than the ranking's.
            passes_separation = True
        else:
            # An unknown label leaves the distance unmeasured, which is never a pass.
            passes_separation = (
                separation is not None and separation >= min_separation_mm
            )
        points.append(
            {
                "timean": record["timean"],
                "focality": record["focality"],
                "passes_intensity": passes_intensity,
                "passes_separation": passes_separation,
            }
        )

        if not passes_intensity:
            dropped["intensity"] += 1
            continue
        if not passes_separation:
            dropped["separation"] += 1
            continue

        candidates.append(
            {
                "montage": record["montage"],
                "electrodes": list(record["electrodes"]),
                "pairs": [list(pair) for pair in pairs],
                "timax": _float(row.get("TImax_ROI")),
                "timean": record["timean"],
                "timean_gm": _float(row.get("TImean_GM")),
                "focality": record["focality"],
                "composite": record["composite"],
                "min_separation_mm": separation,
                "min_within_channel_mm": (
                    min_within_channel_separation(pairs, positions)
                    if positions
                    else None
                ),
                "row": row,
            }
        )

    candidates.sort(key=lambda item: (-item["focality"], -item["timean"]))

    selected: list[dict] = []
    seen_electrodes: set[tuple[str, ...]] = set()
    for candidate in candidates:
        if len(selected) >= top_n:
            break
        key = tuple(candidate["electrodes"])
        if collapse_current_splits and key in seen_electrodes:
            dropped["current_split"] += 1
            continue
        if min_distinct_electrodes is not None and any(
            len(set(candidate["electrodes"]) - set(chosen["electrodes"]))
            < min_distinct_electrodes
            for chosen in selected
        ):
            dropped["diversity"] += 1
            continue
        seen_electrodes.add(key)
        candidate["rank"] = len(selected) + 1
        selected.append(candidate)

    n_pass_intensity = len(records) - dropped["intensity"]
    return {
        "thresholds": {
            "min_intensity_v_per_m": min_intensity,
            "min_separation_mm": min_separation_mm if separation_applied else None,
            "top_n": top_n,
            "collapse_current_splits": collapse_current_splits,
            "min_distinct_electrodes": min_distinct_electrodes,
        },
        "separation_applied": separation_applied,
        "n_total": len(records),
        "n_pass_intensity": n_pass_intensity,
        "n_pass_separation": len(candidates),
        "n_selected": len(selected),
        "dropped": dropped,
        "selected": selected,
        "points": points,
        "columns": columns,
    }


# ── output ───────────────────────────────────────────────────────────────────


def _format_cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column == _SEPARATION_COLUMN:
        return f"{value:.1f}"
    return str(value)


def _write_csv(path: Path, selection: dict) -> None:
    columns = ["Rank", *selection["columns"], _SEPARATION_COLUMN]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for item in selection["selected"]:
            row = {
                **item["row"],
                "Rank": item["rank"],
                _SEPARATION_COLUMN: item["min_separation_mm"],
            }
            writer.writerow(
                [_format_cell(column, row.get(column)) for column in columns]
            )


def _json_payload(selection: dict, positions_source: str) -> dict:
    return {
        "thresholds": selection["thresholds"],
        "separation_applied": selection["separation_applied"],
        "positions_source": positions_source,
        "n_total": selection["n_total"],
        "n_pass_intensity": selection["n_pass_intensity"],
        "n_pass_separation": selection["n_pass_separation"],
        "n_selected": selection["n_selected"],
        "dropped": selection["dropped"],
        "selected": [
            {
                "rank": item["rank"],
                "montage": item["montage"],
                "pairs": item["pairs"],
                "TImax_ROI": item["timax"],
                "TImean_ROI": item["timean"],
                "TImean_GM": item["timean_gm"],
                "Focality": item["focality"],
                "Composite_Index": item["composite"],
                "min_separation_mm": item["min_separation_mm"],
                "min_within_channel_mm": item["min_within_channel_mm"],
            }
            for item in selection["selected"]
        ],
    }


def write_top_montages(
    run_dir: str | Path,
    selection: dict,
    positions_source: str,
    logger: Any,
    cap_path: Path | None = None,
) -> list[str]:
    """Write ``top_montages.{csv,json}``, the scatter and the per-montage maps.

    The CSV is skipped when nothing qualified -- an empty table is a worse
    signal than its absence, and ``top_montages.json`` says why in either case.
    The map PDF needs *cap_path* (the file the coordinates came from) because
    the layout is resolved from its name, so it is skipped when no cap resolved.
    """
    from tit.plotting.ti_metrics import plot_top_montage_maps, plot_top_montages

    run_dir = Path(run_dir)
    written = []

    json_path = run_dir / TOP_MONTAGES_JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_payload(selection, positions_source), f, indent=2)
        f.write("\n")
    written.append(str(json_path))

    if selection["selected"]:
        csv_path = run_dir / TOP_MONTAGES_CSV
        _write_csv(csv_path, selection)
        written.append(str(csv_path))
        logger.info(
            "Top %d montage(s): %s",
            selection["n_selected"],
            csv_path,
        )
    else:
        logger.warning(
            "No montage passed TImean_ROI >= %.3f%s; see %s",
            selection["thresholds"]["min_intensity_v_per_m"],
            (
                ""
                if not selection["separation_applied"]
                else f" and {selection['thresholds']['min_separation_mm']:.0f} mm separation"
            ),
            json_path,
        )

    try:
        plot_path = plot_top_montages(
            points=selection["points"],
            selected=selection["selected"],
            min_intensity=selection["thresholds"]["min_intensity_v_per_m"],
            separation_applied=selection["separation_applied"],
            output_file=str(run_dir / TOP_MONTAGES_PNG),
            dpi=300,
        )
    except Exception as exc:  # matplotlib failures must not cost the tables
        logger.warning("Top-montage scatter skipped: %s", exc)
        plot_path = None
    if plot_path:
        written.append(plot_path)

    if selection["selected"] and cap_path is not None:
        try:
            map_path = plot_top_montage_maps(
                eeg_positions_csv=str(cap_path),
                selected=selection["selected"],
                output_file=str(run_dir / TOP_MONTAGES_MAP),
                dpi=200,
                title_prefix=_title_prefix(selection),
            )
        except Exception as exc:  # as above -- a figure is not worth the tables
            logger.warning("Top-montage maps skipped: %s", exc)
            map_path = None
        if map_path:
            written.append(map_path)
            logger.info("Top-montage maps: %s", map_path)

    return written


def _title_prefix(selection: dict) -> str:
    """``"mTI Search"`` for a four-channel run, ``"Ex-Search"`` otherwise."""
    first = selection["selected"][0] if selection["selected"] else None
    return "mTI Search" if first and len(first["electrodes"]) > 4 else "Ex-Search"


def rank_run(
    run_dir: str | Path,
    *,
    eeg_csv: str | Path | None = None,
    min_intensity: float = DEFAULT_MIN_INTENSITY,
    min_separation_mm: float = DEFAULT_MIN_SEPARATION_MM,
    top_n: int = DEFAULT_TOP_N,
    collapse_current_splits: bool = True,
    min_distinct_electrodes: int | None = None,
    logger: Any | None = None,
) -> dict:
    """Select and write the top montages of an ex-/m-ex-search run.

    Returns the :func:`select_top_montages` result with the written file paths
    added under ``"paths"`` and the coordinate source under
    ``"positions_source"``.
    """
    logger = logger or logging.getLogger(__name__)
    run_dir = Path(run_dir)
    csv_path = run_dir / "final_output.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No final_output.csv in {run_dir}")

    positions, cap_path, positions_source = electrode_positions_for_run(
        run_dir, eeg_csv=eeg_csv
    )
    if positions is None:
        logger.warning("Electrode separation rule not applied: %s", positions_source)

    selection = select_top_montages(
        csv_path,
        positions,
        min_intensity=min_intensity,
        min_separation_mm=min_separation_mm,
        top_n=top_n,
        collapse_current_splits=collapse_current_splits,
        min_distinct_electrodes=min_distinct_electrodes,
    )
    selection["positions_source"] = positions_source
    selection["paths"] = write_top_montages(
        run_dir, selection, positions_source, logger, cap_path
    )
    return selection


def main(argv: list[str] | None = None) -> None:
    """``simnibs_python -m tit.opt.ex.ranking <run_dir> [options]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Select the top montages of an ex-/m-ex-search run: TImean_ROI floor, "
            "then focality ranking, rejecting montages whose channels share "
            "neighbouring electrodes."
        )
    )
    parser.add_argument("run_dir", help="Run directory containing final_output.csv")
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=DEFAULT_MIN_INTENSITY,
        help=f"TImean_ROI floor in V/m (default: {DEFAULT_MIN_INTENSITY})",
    )
    parser.add_argument(
        "--min-separation-mm",
        type=float,
        default=DEFAULT_MIN_SEPARATION_MM,
        help=(
            "minimum distance between electrodes of different channels "
            f"(default: {DEFAULT_MIN_SEPARATION_MM:g})"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"how many montages to select (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--eeg-csv",
        default=None,
        help="cap file to measure distances on (default: inferred from the run)",
    )
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="keep every current split instead of the best one per electrode set",
    )
    parser.add_argument(
        "--min-distinct-electrodes",
        type=int,
        default=None,
        help="require each pick to differ from the ones above it by N electrodes",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    selection = rank_run(
        args.run_dir,
        eeg_csv=args.eeg_csv,
        min_intensity=args.min_intensity,
        min_separation_mm=args.min_separation_mm,
        top_n=args.top,
        collapse_current_splits=not args.no_collapse,
        min_distinct_electrodes=args.min_distinct_electrodes,
    )

    print(
        f"{selection['n_total']} evaluations -> "
        f"{selection['n_pass_intensity']} above TImean_ROI {args.min_intensity:g} -> "
        f"{selection['n_pass_separation']} passing separation -> "
        f"{selection['n_selected']} selected"
    )
    for item in selection["selected"]:
        separation = item["min_separation_mm"]
        gap = "n/a" if separation is None else f"{separation:.1f} mm"
        print(
            f"  {item['rank']}. focality {item['focality']:.4f}  "
            f"TImean_ROI {item['timean']:.4f}  min separation {gap}  "
            f"{item['montage']}"
        )
    for path in selection["paths"]:
        print(path)


if __name__ == "__main__":
    main()
