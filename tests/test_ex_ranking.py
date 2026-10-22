"""Top-montage selection for ex-/m-ex-search runs.

Covers tit/opt/ex/ranking.py: the TImean_ROI floor, the focality ranking, the
cross-channel electrode-separation rule, coordinate resolution (subject cap ->
bundled MNI template -> none) and the written artifacts.
"""

import csv
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# matplotlib is a MagicMock in conftest; the ranking module only reaches it
# through tit.plotting.ti_metrics, whose submodule imports must still resolve.
for _mod in ("matplotlib.image", "matplotlib.patches", "matplotlib.path"):
    sys.modules.setdefault(_mod, MagicMock())

from tit.opt.ex.buckets import electrode_pairs  # noqa: E402
from tit.opt.ex.ranking import (  # noqa: E402
    DEFAULT_MIN_INTENSITY,
    DEFAULT_MIN_SEPARATION_MM,
    electrode_positions_for_run,
    min_cross_channel_separation,
    min_within_channel_separation,
    rank_run,
    select_top_montages,
)

HEADER = [
    "Montage",
    "Current_Ch1_mA",
    "Current_Ch2_mA",
    "TImax_ROI",
    "TImean_ROI",
    "TImean_GM",
    "Focality",
    "Composite_Index",
]

# Eight electrodes with deliberately extreme geometry: L1A and L1B are 10 mm
# apart (closer than any threshold) and everything else is 1000 mm or more away,
# so every expected distance below is exact and unmistakable.
POSITIONS = {
    "L1A": (0.0, 0.0, 0.0),
    "L1B": (-10.0, 0.0, 0.0),
    "L2A": (1000.0, 0.0, 0.0),
    "L2B": (2000.0, 0.0, 0.0),
    "R1A": (0.0, 1000.0, 0.0),
    "R1B": (0.0, 2000.0, 0.0),
    "R2A": (1000.0, 1000.0, 0.0),
    "R2B": (2000.0, 2000.0, 0.0),
}


def _row(montage, timean, focality, *, ch1="8.0", ch2="8.0"):
    """A ``final_output.csv`` row with TImean_GM back-computed from *focality*."""
    gm = timean / focality
    return [
        montage,
        ch1,
        ch2,
        f"{timean * 2:.4f}",
        f"{timean:.4f}",
        f"{gm:.4f}",
        f"{focality:.4f}",
        f"{timean * focality:.4f}",
    ]


def _write_csv(path: Path, rows) -> Path:
    csv_path = path / "final_output.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return csv_path


# ── separation ───────────────────────────────────────────────────────────────


def test_the_shipped_default_separation_is_25mm():
    # The wiki quotes this number and argues for it from the cap's neighbour
    # spacing; changing it is a decision, not a tweak.
    assert DEFAULT_MIN_SEPARATION_MM == 25.0


def test_electrode_pairs_groups_consecutive_labels():
    assert electrode_pairs(["A", "B", "C", "D"]) == [("A", "B"), ("C", "D")]
    assert len(electrode_pairs(list("ABCDEFGH"))) == 4


def test_min_cross_channel_separation_ignores_same_channel_distance():
    # L1A/L1B are 10 mm apart but belong to the *same* channel, so the
    # cross-channel minimum is the 1000 mm gap to the other channel.
    pairs = [("L1A", "L1B"), ("L2A", "L2B")]
    assert min_cross_channel_separation(pairs, POSITIONS) == pytest.approx(1000.0)
    assert min_within_channel_separation(pairs, POSITIONS) == pytest.approx(10.0)


def test_min_cross_channel_separation_finds_the_closest_cross_pair():
    # The same eight electrodes, re-dealt so L1A and L1B land in different
    # channels: now the 10 mm gap is the one that counts.
    pairs = [("L1A", "L2A"), ("L1B", "L2B")]
    assert min_cross_channel_separation(pairs, POSITIONS) == pytest.approx(10.0)


def test_min_cross_channel_separation_is_none_when_a_label_is_unknown():
    pairs = [("L1A", "L1B"), ("GHOST", "L2B")]
    assert min_cross_channel_separation(pairs, POSITIONS) is None


def test_min_cross_channel_separation_is_none_for_a_single_channel():
    assert min_cross_channel_separation([("L1A", "L1B")], POSITIONS) is None


# ── selection ────────────────────────────────────────────────────────────────


def test_intensity_floor_is_inclusive(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        [
            _row("L1A_L1B <> L2A_L2B", 0.6, 1.5),
            _row("L1A_L1B <> R2A_R2B", 0.5999, 2.0),
        ],
    )
    selection = select_top_montages(csv_path, POSITIONS)
    assert selection["n_pass_intensity"] == 1
    assert selection["dropped"]["intensity"] == 1
    assert [item["montage"] for item in selection["selected"]] == ["L1A_L1B <> L2A_L2B"]


def test_ranking_is_by_focality_not_composite(tmp_path):
    # The second row wins on Composite_Index (1.0 * 1.4 > 0.61 * 1.9) and still
    # must come second: focality is the primary key.
    csv_path = _write_csv(
        tmp_path,
        [
            _row("L1A_L1B <> L2A_L2B", 1.0, 1.4),
            _row("R1A_R1B <> R2A_R2B", 0.61, 1.9),
        ],
    )
    selection = select_top_montages(csv_path, POSITIONS)
    assert [item["focality"] for item in selection["selected"]] == [1.9, 1.4]


def test_focality_ties_break_on_intensity(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        [
            _row("L1A_L1B <> L2A_L2B", 0.70, 1.5),
            _row("R1A_R1B <> R2A_R2B", 0.90, 1.5),
        ],
    )
    selection = select_top_montages(csv_path, POSITIONS)
    assert [item["timean"] for item in selection["selected"]] == [0.90, 0.70]


def test_adjacent_channels_are_rejected_and_distant_ones_kept(tmp_path):
    # Same electrodes, different channel assignment: the higher-focality row
    # splits L1A and L1B across the two channels, 10 mm apart.
    csv_path = _write_csv(
        tmp_path,
        [
            _row("L1A_L2A <> L1B_L2B", 0.8, 2.0),
            _row("L1A_L1B <> L2A_L2B", 0.8, 1.5),
        ],
    )
    selection = select_top_montages(csv_path, POSITIONS)
    assert selection["dropped"]["separation"] == 1
    assert [item["montage"] for item in selection["selected"]] == ["L1A_L1B <> L2A_L2B"]


def test_close_electrodes_within_one_channel_are_kept(tmp_path):
    csv_path = _write_csv(tmp_path, [_row("L1A_L1B <> L2A_L2B", 0.8, 1.5)])
    selection = select_top_montages(csv_path, POSITIONS)
    assert selection["n_selected"] == 1
    assert selection["selected"][0]["min_within_channel_mm"] == pytest.approx(10.0)


def test_unknown_electrode_labels_never_pass_the_separation_rule(tmp_path):
    csv_path = _write_csv(tmp_path, [_row("L1A_L1B <> GHOST_L2B", 0.8, 1.5)])
    selection = select_top_montages(csv_path, POSITIONS)
    assert selection["n_selected"] == 0
    assert selection["dropped"]["separation"] == 1


def test_a_single_channel_montage_has_no_separation_to_violate(tmp_path):
    # ex-search never writes one, but a one-channel row must not be silently
    # dropped by a rule that has nothing to measure.
    csv_path = _write_csv(tmp_path, [_row("L1A_L1B", 0.8, 1.5)])
    selection = select_top_montages(csv_path, POSITIONS)
    assert selection["n_selected"] == 1
    assert selection["selected"][0]["min_separation_mm"] is None


def test_without_positions_the_rule_is_reported_as_not_applied(tmp_path):
    csv_path = _write_csv(tmp_path, [_row("L1A_L2A <> L1B_L2B", 0.8, 1.5)])
    selection = select_top_montages(csv_path, None)
    assert selection["separation_applied"] is False
    assert selection["dropped"]["separation"] == 0
    assert selection["n_selected"] == 1
    assert selection["selected"][0]["min_separation_mm"] is None


def test_shortfall_returns_fewer_without_relaxing_anything(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        [
            _row("L1A_L1B <> L2A_L2B", 0.8, 1.5),
            _row("R1A_R1B <> R2A_R2B", 0.4, 3.0),
            _row("L1A_L2A <> L1B_L2B", 0.9, 2.5),
        ],
    )
    selection = select_top_montages(csv_path, POSITIONS, top_n=5)
    assert selection["n_selected"] == 1
    assert selection["dropped"] == {
        "intensity": 1,
        "separation": 1,
        "current_split": 0,
        "diversity": 0,
    }
    assert selection["thresholds"]["min_intensity_v_per_m"] == DEFAULT_MIN_INTENSITY
    assert selection["thresholds"]["min_separation_mm"] == DEFAULT_MIN_SEPARATION_MM


def test_current_splits_of_one_electrode_set_collapse_to_the_best(tmp_path):
    rows = [
        _row("L1A_L1B <> L2A_L2B", 0.8, 1.9, ch1="7.0", ch2="9.0"),
        _row("L1A_L1B <> L2A_L2B", 0.8, 1.7, ch1="8.0", ch2="8.0"),
        _row("R1A_R1B <> R2A_R2B", 0.8, 1.8),
    ]
    # The montage name carries the split, as tit.opt.ex writes it.
    rows[0][0] = "L1A_L1B <> L2A_L2B_I1-7.0mA_I2-9.0mA"
    rows[1][0] = "L1A_L1B <> L2A_L2B_I1-8.0mA_I2-8.0mA"
    rows[2][0] = "R1A_R1B <> R2A_R2B_I1-8.0mA_I2-8.0mA"
    csv_path = _write_csv(tmp_path, rows)

    collapsed = select_top_montages(csv_path, POSITIONS)
    assert [item["focality"] for item in collapsed["selected"]] == [1.9, 1.8]
    assert collapsed["dropped"]["current_split"] == 1

    kept = select_top_montages(csv_path, POSITIONS, collapse_current_splits=False)
    assert [item["focality"] for item in kept["selected"]] == [1.9, 1.8, 1.7]


def test_min_distinct_electrodes_drops_near_duplicates(tmp_path):
    rows = [
        _row("L1A_L1B <> L2A_L2B", 0.8, 1.9),
        _row("L1A_L1B <> L2A_R2B", 0.8, 1.8),  # one electrode different
        _row("R1A_R1B <> R2A_R2B", 0.8, 1.7),  # nothing in common
    ]
    csv_path = _write_csv(tmp_path, rows)
    assert select_top_montages(csv_path, POSITIONS)["n_selected"] == 3
    diverse = select_top_montages(csv_path, POSITIONS, min_distinct_electrodes=2)
    assert [item["focality"] for item in diverse["selected"]] == [1.9, 1.7]
    assert diverse["dropped"]["diversity"] == 1


def test_an_eight_electrode_montage_is_parsed_as_four_channels(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        [_row("L1A_L1B <> R1A_R1B <> L2A_L2B <> R2A_R2B", 0.8, 1.5)],
    )
    selection = select_top_montages(csv_path, POSITIONS)
    assert len(selection["selected"][0]["pairs"]) == 4


# ── coordinate resolution ────────────────────────────────────────────────────


def _write_run_config(run_dir: Path, leadfield: str) -> None:
    (run_dir / "run_config.json").write_text(json.dumps({"leadfield_hdf": leadfield}))


def test_positions_come_from_the_subject_cap_next_to_the_run(tmp_path):
    subject = tmp_path / "sub-X"
    run_dir = subject / "ex-search" / "run"
    run_dir.mkdir(parents=True)
    eeg_dir = subject / "m2m_X" / "eeg_positions"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "TestNet.csv").write_text("Electrode,1,2,3,A1\nElectrode,4,5,6,A2\n")
    # The recorded leadfield path is the container's; only its net name is used.
    _write_run_config(run_dir, "/mnt/TIToolbox/.../X_leadfield_TestNet.hdf5")

    positions, cap_path, source = electrode_positions_for_run(run_dir)
    assert positions == {"A1": (1.0, 2.0, 3.0), "A2": (4.0, 5.0, 6.0)}
    assert cap_path == eeg_dir / "TestNet.csv"
    assert source.startswith("subject cap:")


def test_positions_fall_back_to_the_bundled_mni_template(tmp_path):
    run_dir = tmp_path / "sub-X" / "ex-search" / "run"
    run_dir.mkdir(parents=True)
    _write_run_config(run_dir, "/mnt/nowhere/X_leadfield_GSN-HydroCel-256.hdf5")

    positions, cap_path, source = electrode_positions_for_run(run_dir)
    assert source.startswith("MNI template:")
    assert positions is not None and "E001" in positions
    assert len(positions["E001"]) == 3
    # The maps resolve their layout from this name, so it has to be the net's.
    assert cap_path is not None and cap_path.name == "GSN-HydroCel-256.csv"


def test_an_unknown_net_yields_no_positions_rather_than_bad_ones(tmp_path):
    run_dir = tmp_path / "sub-X" / "ex-search" / "run"
    run_dir.mkdir(parents=True)
    _write_run_config(run_dir, "/mnt/nowhere/X_leadfield_NoSuchNet.hdf5")

    positions, cap_path, source = electrode_positions_for_run(run_dir)
    assert positions is None
    assert cap_path is None
    assert "NoSuchNet" in source


def test_an_explicit_cap_file_wins(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cap = tmp_path / "cap.csv"
    cap.write_text("Electrode,0,0,0,Z1\n")
    positions, cap_path, source = electrode_positions_for_run(run_dir, eeg_csv=cap)
    assert positions == {"Z1": (0.0, 0.0, 0.0)}
    assert cap_path == cap
    assert source.startswith("explicit:")


def test_an_unparseable_cap_file_counts_as_no_coordinates(tmp_path):
    # An empty parse must disable the rule, not make every label unknown and
    # silently drop the whole run.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cap = tmp_path / "cap.csv"
    cap.write_text("not,a,cap\n")
    positions, cap_path, source = electrode_positions_for_run(run_dir, eeg_csv=cap)
    assert positions is None
    assert cap_path is None
    assert "no electrode positions" in source


def test_a_missing_explicit_cap_file_is_an_error(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        electrode_positions_for_run(run_dir, eeg_csv=tmp_path / "absent.csv")


# ── artifacts ────────────────────────────────────────────────────────────────


# matplotlib is a MagicMock in the host suite, so the real figure code cannot
# run here; ``tests/numerical`` is where plotting is exercised for real.
@contextmanager
def _no_plot():
    """Stub both figure writers, yielding the two mocks."""
    with (
        patch(
            "tit.plotting.ti_metrics.plot_top_montages", return_value=None
        ) as scatter,
        patch(
            "tit.plotting.ti_metrics.plot_top_montage_maps", return_value=None
        ) as maps,
    ):
        yield scatter, maps


def test_rank_run_writes_the_csv_json_and_scatter(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cap = tmp_path / "cap.csv"
    cap.write_text(
        "".join(
            f"Electrode,{x},{y},{z},{label}\n" for label, (x, y, z) in POSITIONS.items()
        )
    )
    _write_csv(
        run_dir,
        [
            _row("L1A_L1B <> L2A_L2B", 0.8, 1.9),
            _row("R1A_R1B <> R2A_R2B", 0.7, 1.6),
            _row("L1A_L2A <> L1B_L2B", 0.9, 2.5),
        ],
    )

    with _no_plot() as (_, maps):
        selection = rank_run(run_dir, eeg_csv=cap, top_n=2)

    assert selection["n_selected"] == 2
    csv_rows = list(csv.DictReader(open(run_dir / "top_montages.csv")))
    assert [row["Rank"] for row in csv_rows] == ["1", "2"]
    assert csv_rows[0]["Montage"] == "L1A_L1B <> L2A_L2B"
    # The source file's own columns survive, plus the two the ranking adds.
    assert csv_rows[0]["Focality"] == "1.9000"
    assert float(csv_rows[0]["MinSeparation_mm"]) == pytest.approx(1000.0, abs=0.1)

    payload = json.loads((run_dir / "top_montages.json").read_text())
    assert payload["separation_applied"] is True
    assert payload["dropped"]["separation"] == 1
    assert payload["positions_source"].startswith("explicit:")
    assert payload["selected"][0]["pairs"] == [["L1A", "L1B"], ["L2A", "L2B"]]


def test_the_maps_are_drawn_from_the_cap_file_the_distances_used(tmp_path):
    # plot_top_montage_maps resolves its head-template layout from the cap's
    # file NAME, so it needs the path, not the coordinates the ranking parsed.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cap = tmp_path / "GSN-HydroCel-256.csv"
    cap.write_text(
        "".join(
            f"Electrode,{x},{y},{z},{label}\n" for label, (x, y, z) in POSITIONS.items()
        )
    )
    _write_csv(run_dir, [_row("L1A_L1B <> L2A_L2B", 0.8, 1.9)])

    with _no_plot() as (_, maps):
        rank_run(run_dir, eeg_csv=cap)

    kwargs = maps.call_args.kwargs
    assert kwargs["eeg_positions_csv"] == str(cap)
    assert [item["rank"] for item in kwargs["selected"]] == [1]
    assert kwargs["output_file"].endswith("top_montages_map.pdf")


def test_the_maps_are_skipped_when_no_cap_resolved(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_csv(run_dir, [_row("L1A_L1B <> L2A_L2B", 0.8, 1.9)])

    with _no_plot() as (_, maps):
        selection = rank_run(run_dir)

    maps.assert_not_called()
    assert selection["n_selected"] == 1
    assert not any(Path(p).name == "top_montages_map.pdf" for p in selection["paths"])


def test_a_four_channel_run_titles_its_maps_as_mti(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cap = tmp_path / "GSN-HydroCel-256.csv"
    cap.write_text(
        "".join(
            f"Electrode,{x},{y},{z},{label}\n" for label, (x, y, z) in POSITIONS.items()
        )
    )
    _write_csv(run_dir, [_row("L1A_L1B <> L2A_L2B <> R1A_R1B <> R2A_R2B", 0.8, 1.9)])

    with _no_plot() as (_, maps):
        rank_run(run_dir, eeg_csv=cap)

    assert maps.call_args.kwargs["title_prefix"] == "mTI Search"


def test_rank_run_writes_json_but_no_csv_when_nothing_qualifies(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_csv(run_dir, [_row("L1A_L1B <> L2A_L2B", 0.2, 3.0)])

    with _no_plot():
        selection = rank_run(run_dir)

    assert selection["n_selected"] == 0
    assert not (run_dir / "top_montages.csv").exists()
    payload = json.loads((run_dir / "top_montages.json").read_text())
    assert payload["n_total"] == 1
    assert payload["dropped"]["intensity"] == 1
    assert payload["separation_applied"] is False


def test_rank_run_needs_a_final_output_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        rank_run(tmp_path)


def test_a_failed_figure_does_not_cost_the_tables(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_csv(run_dir, [_row("L1A_L1B <> L2A_L2B", 0.8, 1.5)])

    with (
        patch(
            "tit.plotting.ti_metrics.plot_top_montages",
            side_effect=AttributeError("boom"),
        ),
        patch(
            "tit.plotting.ti_metrics.plot_top_montage_maps",
            side_effect=AttributeError("boom"),
        ),
    ):
        selection = rank_run(run_dir)

    assert (run_dir / "top_montages.csv").is_file()
    assert (run_dir / "top_montages.json").is_file()
    assert not any(
        Path(p).name == "top_montages_scatter.png" for p in selection["paths"]
    )


def test_the_pipeline_hook_never_loses_a_finished_search(tmp_path):
    # _rank_top_montages is called after final_output.csv is already on disk, so
    # whatever goes wrong in it must stay a warning.
    from tit.opt.ex.results import _rank_top_montages

    logger = MagicMock()
    with patch("tit.opt.ex.ranking.rank_run", side_effect=RuntimeError("boom")):
        assert _rank_top_montages(str(tmp_path), logger) == []
    assert any(
        "Top-montage ranking skipped" in str(call.args[0])
        for call in logger.warning.call_args_list
    )
