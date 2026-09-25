"""Tests for tit/opt/mex/ -- multipolar (4-pair, 8-electrode) exhaustive search.

Covers candidate generation (logic.py), the electrode mirror map and
generalized bucket loader (ex/buckets.py), MExConfig round trips through
config_io, the engine's ROI metric computation, and the carrier-grouping
behavior that replaces the rejected recursive mTI dispatch.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure simnibs.utils.TI_utils is mocked before tit.opt.mex.engine (or
# tit.opt.ex.engine, which it subclasses) is imported.
for mod_name in ("simnibs.utils.TI_utils",):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


# ---------------------------------------------------------------------------
# tit.opt.mex.logic -- candidate generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateMultipolarCombinations:
    def test_bucket_cartesian_filters_non_distinct_electrodes(self):
        from tit.opt.mex.logic import generate_multipolar_combinations

        # e1_plus has two options; one ("X") collides with e4_minus, so only
        # the combination built from "A" survives the all-distinct filter.
        buckets = {
            "e1_plus": ["A", "X"],
            "e1_minus": ["B"],
            "e2_plus": ["C"],
            "e2_minus": ["D"],
            "e3_plus": ["E"],
            "e3_minus": ["F"],
            "e4_plus": ["G"],
            "e4_minus": ["X"],
        }
        combos = list(generate_multipolar_combinations(buckets, all_combinations=False))
        assert combos == [("A", "B", "C", "D", "E", "F", "G", "X")]

    def test_pool_mode_count_matches_enumeration_and_closed_form(self):
        """Pool mode yields one montage per mTI symmetry class, counted analytically.

        With two independent channels the score is invariant under a group
        of order 64 (8 slot automorphisms x 8 group-flip patterns), so
        P(n, 8) / 64 canonical montages remain.
        """
        from tit.opt.mex.logic import (
            count_multipolar_combinations,
            generate_multipolar_combinations,
        )

        pool = [f"E{i}" for i in range(1, 10)]
        count = count_multipolar_combinations(pool, all_combinations=True)
        assert count == math.perm(9, 8) // 64
        combos = list(generate_multipolar_combinations(pool, all_combinations=True))
        assert len(combos) == count
        assert len(set(combos)) == count
        for combo in combos[:50]:
            assert len(combo) == 8
            assert len(set(combo)) == 8
            assert set(combo) <= set(pool)

    @pytest.mark.parametrize(
        "channels, group_order",
        [
            (None, 64),
            ([([0, 2], [1, 3])], 32),
            ([([0], [1, 2]), ([3], [])], 16),
            ([([0, 1, 2, 3], [])], 48),
        ],
    )
    def test_pool_mode_is_a_transversal_of_the_symmetry_classes(
        self, channels, group_order
    ):
        """No two enumerated montages are equivalent, and every class is hit.

        The symmetry group is built independently here from the envelope
        math in ``tit.calc`` (``Q = |sum_k a_k*b_k|`` over beating channels,
        ``psi=0``): slot automorphisms of the channel structure, and
        group-wise polarity flips whose per-channel product sign agrees.
        """
        from itertools import permutations, product

        from tit.opt.mex.logic import (
            _normalize_channels,
            _slot_automorphisms,
            count_multipolar_combinations,
            generate_multipolar_combinations,
        )

        norm = _normalize_channels(channels)
        autos = _slot_automorphisms(norm)
        groups = [g for a, b in norm for g in (a, b) if g]
        signs = []
        for bits in product((1, -1), repeat=len(groups)):
            gs = dict(zip(groups, bits))
            products = {gs[a] * gs[b] for a, b in norm if a and b}
            if len(products) <= 1:
                signs.append({slot: gs[g] for g in groups for slot in g})
        assert len(autos) * len(signs) == group_order

        def canon(montage):
            pairs = [montage[2 * k : 2 * k + 2] for k in range(4)]
            images = []
            for perm in autos:
                for sg in signs:
                    placed = [None] * 4
                    for slot, pair in enumerate(pairs):
                        placed[perm[slot]] = pair[::-1] if sg[slot] < 0 else pair
                    images.append(tuple(e for pair in placed for e in pair))
            return min(images)

        pool = [f"E{i}" for i in range(8)]
        combos = list(
            generate_multipolar_combinations(
                pool, all_combinations=True, channels=channels
            )
        )
        classes = [canon(c) for c in combos]
        assert len(set(classes)) == len(classes), "duplicate symmetry class"
        assert set(classes) == {canon(p) for p in permutations(pool, 8)}
        assert len(combos) == math.factorial(8) // group_order
        assert count_multipolar_combinations(
            pool, all_combinations=True, channels=channels
        ) == len(combos)

    def test_pool_mode_polarity_flip_inside_a_beating_channel_is_kept(self):
        """A lone flip within a beating channel is a pi phase shift, not a symmetry."""
        from tit.opt.mex.logic import generate_multipolar_combinations

        pool = [f"E{i}" for i in range(8)]
        combos = set(generate_multipolar_combinations(pool, all_combinations=True))
        aligned = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
        shifted = ("E0", "E1", "E2", "E3", "E4", "E5", "E7", "E6")
        assert aligned in combos
        assert shifted in combos
        # Flipping slot 1 instead of slot 3 is the same class as ``shifted``
        # (Q = |a0*b0 - a1*b1| either way), and flipping both pairs of a
        # channel is the same class as ``aligned`` -- neither is re-emitted.
        assert ("E0", "E1", "E3", "E2", "E4", "E5", "E6", "E7") not in combos
        assert ("E1", "E0", "E3", "E2", "E4", "E5", "E6", "E7") not in combos

    def test_pool_mode_rejects_channels_missing_a_slot(self):
        from tit.opt.mex.logic import count_multipolar_combinations

        with pytest.raises(ValueError, match="four pair slots"):
            count_multipolar_combinations(
                [f"E{i}" for i in range(8)],
                all_combinations=True,
                channels=[([0], [1]), ([2], [])],
            )

    def test_bucket_mode_count_still_walks_the_generator(self):
        from tit.opt.mex.logic import (
            count_multipolar_combinations,
            generate_multipolar_combinations,
        )

        buckets = {
            "e1_plus": ["A", "B"],
            "e1_minus": ["C"],
            "e2_plus": ["D"],
            "e2_minus": ["E"],
            "e3_plus": ["F"],
            "e3_minus": ["G"],
            "e4_plus": ["H"],
            "e4_minus": ["A", "I"],
        }
        combos = list(generate_multipolar_combinations(buckets))
        assert count_multipolar_combinations(buckets) == len(combos) == 3

    def test_symmetric_within_pairs_collapses_cartesian_to_linear(self):
        """|e1+| x |e1-| candidates collapse to ~|e1+| under symmetric mode."""
        from tit.opt.mex.logic import count_multipolar_combinations

        # Every bucket's electrodes need a mirror partner: the within-pair
        # matcher requires all four pairs to resolve simultaneously (it is a
        # product across pairs), not just e1.
        mirror_map = {
            "L1": "R1",
            "R1": "L1",
            "L2": "R2",
            "R2": "L2",
            "X": "Y",
            "Y": "X",
            "P": "Q",
            "Q": "P",
            "M": "N",
            "N": "M",
        }
        buckets = {
            "e1_plus": ["L1", "L2"],
            "e1_minus": ["R1", "R2"],
            "e2_plus": ["X"],
            "e2_minus": ["Y"],
            "e3_plus": ["P"],
            "e3_minus": ["Q"],
            "e4_plus": ["M"],
            "e4_minus": ["N"],
        }

        # Unrestricted: full 2x2 cartesian product of e1_plus x e1_minus.
        unrestricted = count_multipolar_combinations(buckets, all_combinations=False)
        assert unrestricted == 4

        # Symmetric: only mirrored (L1,R1) and (L2,R2) pairs survive -- linear
        # in |e1+|, not |e1+| x |e1-|.
        symmetric = count_multipolar_combinations(
            buckets, all_combinations=False, symmetry_mirror_map=mirror_map
        )
        assert symmetric == len(buckets["e1_plus"])
        assert symmetric == 2


# ---------------------------------------------------------------------------
# tit.opt.ex.buckets -- build_electrode_mirror_map
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildElectrodeMirrorMap:
    def test_maps_left_right_pairs_and_midline_to_self(self, tmp_path):
        from tit.opt.ex.buckets import build_electrode_mirror_map

        csv_path = tmp_path / "net.csv"
        csv_path.write_text(
            "L1,-30,10,5\n"
            "R1,30,10,5\n"
            "L2,-20,-40,5\n"
            "R2,20,-40,5\n"
            "CZ,0,50,80\n"
        )

        mirror = build_electrode_mirror_map(csv_path)

        assert mirror["L1"] == "R1"
        assert mirror["R1"] == "L1"
        assert mirror["L2"] == "R2"
        assert mirror["R2"] == "L2"
        assert mirror["CZ"] == "CZ"


# ---------------------------------------------------------------------------
# tit.opt.ex.buckets -- generalized bucket loader (4-key default, 8-key mex)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGeneralizedBucketLoader:
    def test_default_4key_json_roundtrip_unchanged(self, tmp_path):
        from tit.opt.ex.buckets import BUCKET_KEYS, load_bucket_file, save_bucket_file

        buckets = {
            "e1_plus": ["A", "B"],
            "e1_minus": ["C"],
            "e2_plus": ["D"],
            "e2_minus": ["E", "F"],
        }
        path = tmp_path / "buckets.json"
        save_bucket_file(path, buckets)
        loaded = load_bucket_file(path)

        assert loaded == buckets
        assert set(loaded) == set(BUCKET_KEYS)

    def test_default_4key_aliases_still_curated(self):
        """Existing 4-key callers keep the hand-curated BUCKET_ALIASES table."""
        from tit.opt.ex.buckets import normalize_buckets

        result = normalize_buckets({"e1_": ["Z"], "E2+": ["Y"]})
        assert result["e1_minus"] == ["Z"]
        assert result["e2_plus"] == ["Y"]

    def test_8key_json_roundtrip(self, tmp_path):
        from tit.opt.ex.buckets import load_bucket_file, save_bucket_file
        from tit.opt.mex.logic import MEX_BUCKET_KEYS

        buckets = {key: [f"{key}_A", f"{key}_B"] for key in MEX_BUCKET_KEYS}
        path = tmp_path / "mex_buckets.json"
        save_bucket_file(path, buckets, keys=MEX_BUCKET_KEYS)
        loaded = load_bucket_file(path, keys=MEX_BUCKET_KEYS)

        assert loaded == buckets
        assert set(loaded) == set(MEX_BUCKET_KEYS)

    def test_8key_csv_roundtrip_with_symbol_aliases(self, tmp_path):
        from tit.opt.ex.buckets import load_bucket_file
        from tit.opt.mex.logic import MEX_BUCKET_KEYS

        csv_path = tmp_path / "mex_buckets.csv"
        csv_path.write_text(
            "e1+,A,B\n"
            "e1-,C\n"
            "e2+,D\n"
            "e2-,E\n"
            "e3+,F\n"
            "e3-,G\n"
            "e4+,H\n"
            "e4-,I\n"
        )
        loaded = load_bucket_file(csv_path, keys=MEX_BUCKET_KEYS)

        assert loaded["e1_plus"] == ["A", "B"]
        assert loaded["e3_minus"] == ["G"]
        assert loaded["e4_plus"] == ["H"]


# ---------------------------------------------------------------------------
# MExConfig -- construction and config_io round trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMExConfigValidation:
    def _bucket_electrodes(self, **overrides):
        defaults = dict(
            e1_plus=["A"],
            e1_minus=["B"],
            e2_plus=["C"],
            e2_minus=["D"],
            e3_plus=["E"],
            e3_minus=["F"],
            e4_plus=["G"],
            e4_minus=["H"],
        )
        defaults.update(overrides)
        from tit.opt.config import MExConfig

        return MExConfig.BucketElectrodes(**defaults)

    def test_defaults(self):
        from tit.opt.config import MExConfig

        config = MExConfig(
            subject_id="001",
            leadfield_hdf="lf.hdf5",
            roi_name="target",
            electrodes=self._bucket_electrodes(),
        )
        assert config.roi_name == "target.csv"
        assert config.current_mA == 2.0
        assert config.symmetric_bucket is False

    def test_rejects_non_positive_current(self):
        from tit.opt.config import MExConfig

        with pytest.raises(ValueError):
            MExConfig(
                subject_id="001",
                leadfield_hdf="lf.hdf5",
                roi_name="target",
                electrodes=self._bucket_electrodes(),
                current_mA=0.0,
            )

    def test_pair_currents_default_to_current_mA(self):
        from tit.opt.config import MExConfig

        config = MExConfig(
            subject_id="001",
            leadfield_hdf="lf.hdf5",
            roi_name="target",
            electrodes=self._bucket_electrodes(),
            current_mA=1.5,
        )
        assert config.currents_mA is None
        assert config.pair_currents_mA == (1.5, 1.5, 1.5, 1.5)

    def test_per_pair_currents_override_current_mA(self):
        from tit.opt.config import MExConfig

        config = MExConfig(
            subject_id="001",
            leadfield_hdf="lf.hdf5",
            roi_name="target",
            electrodes=self._bucket_electrodes(),
            currents_mA=[2, 2, 1, 3],
        )
        assert config.pair_currents_mA == (2.0, 2.0, 1.0, 3.0)

    @pytest.mark.parametrize(
        "currents",
        [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [2.0, 0.0, 2.0, 2.0],
            [2.0, -1.0, 2.0, 2.0],
            [2.0, math.nan, 2.0, 2.0],
            [2.0, math.inf, 2.0, 2.0],
        ],
    )
    def test_rejects_malformed_per_pair_currents(self, currents):
        from tit.opt.config import MExConfig

        with pytest.raises(ValueError, match="currents_mA"):
            MExConfig(
                subject_id="001",
                leadfield_hdf="lf.hdf5",
                roi_name="target",
                electrodes=self._bucket_electrodes(),
                currents_mA=currents,
            )

    def test_unequal_currents_reject_pool_electrodes(self):
        # Pool mode keeps one montage per class of slot permutations, which is
        # only a symmetry while every slot carries the same current.
        from tit.opt.config import MExConfig

        pool = MExConfig.PoolElectrodes(electrodes=[f"E{i}" for i in range(1, 9)])
        with pytest.raises(ValueError, match="bucket electrodes"):
            MExConfig(
                subject_id="001",
                leadfield_hdf="lf.hdf5",
                roi_name="target",
                electrodes=pool,
                currents_mA=[2, 2, 1, 3],
            )
        # Equal per-pair currents keep the slots interchangeable.
        MExConfig(
            subject_id="001",
            leadfield_hdf="lf.hdf5",
            roi_name="target",
            electrodes=pool,
            currents_mA=[1.5, 1.5, 1.5, 1.5],
        )

    def test_symmetric_bucket_rejects_pool_electrodes(self):
        from tit.opt.config import MExConfig

        with pytest.raises(ValueError):
            MExConfig(
                subject_id="001",
                leadfield_hdf="lf.hdf5",
                roi_name="target",
                electrodes=MExConfig.PoolElectrodes(electrodes=["E1"] * 8),
                symmetric_bucket=True,
            )

    def test_rejects_unknown_symmetry_pairing(self):
        from tit.opt.config import MExConfig

        with pytest.raises(ValueError):
            MExConfig(
                subject_id="001",
                leadfield_hdf="lf.hdf5",
                roi_name="target",
                electrodes=self._bucket_electrodes(),
                symmetry_pairing="diagonal",
            )


@pytest.mark.unit
class TestMExConfigConfigIO:
    def test_bucket_electrodes_roundtrip(self, tmp_path):
        from tit.config_io import read_config_json, write_config_json
        from tit.opt.config import MExConfig

        config = MExConfig(
            subject_id="001",
            leadfield_hdf="net_leadfield.hdf5",
            roi_name="target",
            electrodes=MExConfig.BucketElectrodes(
                e1_plus=["A"],
                e1_minus=["B"],
                e2_plus=["C"],
                e2_minus=["D"],
                e3_plus=["E"],
                e3_minus=["F"],
                e4_plus=["G"],
                e4_minus=["H"],
            ),
            current_mA=1.5,
        )
        path = write_config_json(config, prefix="mex_test")
        try:
            data = read_config_json(path)
            assert data["electrodes"]["_type"] == "BucketElectrodes"
            assert data["electrodes"]["e3_minus"] == ["F"]
            assert data["roi_name"] == "target.csv"
            assert data["current_mA"] == 1.5
        finally:
            os.unlink(path)

    def test_pool_electrodes_survive_main_rebuild(self):
        from tit.config_io import deserialize_config, serialize_config
        from tit.opt.config import MExConfig

        config = MExConfig(
            subject_id="001",
            leadfield_hdf="net_leadfield.hdf5",
            roi_name="target",
            electrodes=MExConfig.PoolElectrodes(
                electrodes=[f"E{i}" for i in range(1, 9)]
            ),
        )
        data = json.loads(json.dumps(serialize_config(config)))
        data.pop("project_dir", None)

        # tit.opt.mex.__main__ rebuilds the whole config through
        # tit.config_io.deserialize_config rather than hand-rolling electrodes.
        rebuilt = deserialize_config(MExConfig, data)

        assert isinstance(rebuilt.electrodes, MExConfig.PoolElectrodes)
        assert rebuilt.electrodes.electrodes == [f"E{i}" for i in range(1, 9)]


# ---------------------------------------------------------------------------
# tit.opt.ex.results.save_run_config -- ExConfig/MExConfig polymorphism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveRunConfigPolymorphism:
    """save_run_config (unchanged elsewhere) was generalized to accept either
    ExConfig's total_current/current_step/channel_limit or MExConfig's flat
    current_mA, and to build electrode_info from dataclass fields instead of
    a hard-coded 4-key dict.
    """

    def test_ex_config_bucket_mode_unaffected(self, tmp_path):
        from tit.opt.config import ExConfig
        from tit.opt.ex.results import save_run_config

        config = ExConfig(
            subject_id="001",
            leadfield_hdf="lf.hdf5",
            roi_name="roi",
            electrodes=ExConfig.BucketElectrodes(
                e1_plus=["A"], e1_minus=["B"], e2_plus=["C"], e2_minus=["D"]
            ),
            total_current=2.0,
            current_step=0.5,
        )
        path = save_run_config(config, 4, str(tmp_path), MagicMock())
        data = json.loads(Path(path).read_text())

        assert data["electrodes"] == {
            "e1_plus": ["A"],
            "e1_minus": ["B"],
            "e2_plus": ["C"],
            "e2_minus": ["D"],
        }
        assert data["total_current_mA"] == 2.0
        assert "current_mA" not in data

    def test_mex_config_bucket_mode_has_eight_keys_and_current_mA(self, tmp_path):
        from tit.opt.config import MExConfig
        from tit.opt.ex.results import save_run_config

        config = MExConfig(
            subject_id="001",
            leadfield_hdf="lf.hdf5",
            roi_name="roi",
            electrodes=MExConfig.BucketElectrodes(
                e1_plus=["A"],
                e1_minus=["B"],
                e2_plus=["C"],
                e2_minus=["D"],
                e3_plus=["E"],
                e3_minus=["F"],
                e4_plus=["G"],
                e4_minus=["H"],
            ),
            current_mA=2.0,
        )
        path = save_run_config(config, 8, str(tmp_path), MagicMock())
        data = json.loads(Path(path).read_text())

        assert set(data["electrodes"]) == {
            "e1_plus",
            "e1_minus",
            "e2_plus",
            "e2_minus",
            "e3_plus",
            "e3_minus",
            "e4_plus",
            "e4_minus",
        }
        assert data["current_mA"] == 2.0
        assert "total_current_mA" not in data
        assert "currents_mA" not in data

    def test_mex_config_records_per_pair_currents(self, tmp_path):
        from tit.opt.config import MExConfig
        from tit.opt.ex.results import save_run_config

        config = MExConfig(
            subject_id="001",
            leadfield_hdf="lf.hdf5",
            roi_name="roi",
            electrodes=MExConfig.BucketElectrodes(
                **{
                    f"e{p}_{s}": [f"{p}{s}"]
                    for p in range(1, 5)
                    for s in ("plus", "minus")
                }
            ),
            currents_mA=[2, 2, 1, 3],
        )
        path = save_run_config(config, 1, str(tmp_path), MagicMock())
        data = json.loads(Path(path).read_text())
        assert data["currents_mA"] == [2.0, 2.0, 1.0, 3.0]
        assert "current_mA" not in data


@pytest.mark.unit
class TestPerPairCurrentResults:
    """Unequal pair currents reach the run name and the results CSV."""

    def test_current_tag_keeps_the_single_value_form_for_equal_currents(self):
        from tit.opt.mex.engine import current_tag

        assert current_tag((2.0, 2.0, 2.0, 2.0)) == "_I-2.0mA"
        assert current_tag((2.0, 2.0, 1.0, 3.0)) == "_I-2.0-2.0-1.0-3.0mA"

    @pytest.mark.parametrize("suffix", ["_I-2.0mA", "_I-2.0-2.0-1.0-3.0mA"])
    def test_montage_string_strips_either_suffix(self, suffix):
        from tit.opt.ex.results import parse_montage_string

        montage = f"A_B <> C_D <> E_F <> G_H{suffix}"
        assert parse_montage_string(montage) == list("ABCDEFGH")

    def test_mex_csv_carries_four_current_columns_and_ex_keeps_two(self):
        from tit.opt.ex.results import build_csv_rows

        metrics = {
            "R_TImax_ROI": 1.0,
            "R_TImean_ROI": 0.5,
            "R_TImean_GM": 0.25,
            "R_Focality": 2.0,
        }
        mex = {
            "TI_field_A_B_and_C_D_and_E_F_and_G_H_I-2.0-2.0-1.0-3.0mA.msh": {
                **metrics,
                **{f"current_ch{ch}_mA": c for ch, c in zip(range(1, 5), (2, 2, 1, 3))},
            }
        }
        rows, *_ = build_csv_rows(mex, "R")
        header, row = rows
        assert header[1:5] == [f"Current_Ch{ch}_mA" for ch in range(1, 5)]
        assert row[1:5] == ["2.0", "2.0", "1.0", "3.0"]

        ex = {
            "TI_field_A_B_and_C_D_I1-1.0mA_I2-1.0mA.msh": {
                **metrics,
                "current_ch1_mA": 1.0,
                "current_ch2_mA": 1.0,
            }
        }
        header, _ = build_csv_rows(ex, "R")[0]
        assert header[1:3] == ["Current_Ch1_mA", "Current_Ch2_mA"]
        assert header[3] == "TImax_ROI"


# ---------------------------------------------------------------------------
# No recursive-envelope dispatch was reintroduced (critical correctness)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoRecursiveDispatch:
    def test_no_recursive_mti_dispatch_reintroduced(self):
        """MExSearchEngine must not depend on compute_mti_metric_field/MTIMetric."""
        import tit.calc as calc_mod
        import tit.opt.config as config_mod
        import tit.opt.mex.engine as engine_mod

        assert not hasattr(calc_mod, "compute_mti_metric_field")
        assert not hasattr(config_mod, "MTIMetric")
        assert not hasattr(config_mod.MExConfig, "MTIMetric")
        assert not hasattr(engine_mod, "compute_mti_metric_field")


# ---------------------------------------------------------------------------
# MExSearchEngine.compute_mti_field -- ROI metric keys
# ---------------------------------------------------------------------------


def _make_mex_engine(logger=None):
    if logger is None:
        logger = MagicMock()
    from tit.opt.mex.engine import MExSearchEngine

    return MExSearchEngine(
        leadfield_hdf="/fake/leadfield.hdf5",
        roi_file="/fake/roi.csv",
        roi_name="TestROI",
        logger=logger,
    )


def _setup_engine_fields(engine):
    engine.leadfield = MagicMock()
    engine.idx_lf = MagicMock()
    engine.mesh = MagicMock()
    engine.roi_indices = np.array([0, 1, 2])
    engine.roi_volumes = np.array([1.0, 1.0, 1.0])
    engine.gm_indices = np.array([0, 1, 2, 3, 4])
    engine.gm_volumes = np.array([1.0, 1.0, 1.0, 1.0, 1.0])


@pytest.mark.unit
class TestMExSearchEngineInit:
    def test_init_stores_roi_name(self):
        engine = _make_mex_engine()
        assert engine.roi_name == "TestROI"


@pytest.mark.unit
class TestComputeMtiField:
    def test_computes_expected_roi_metric_keys(self):
        import tit.opt.mex.engine as engine_mod

        engine = _make_mex_engine()
        _setup_engine_fields(engine)

        engine_mod.TI.get_field = MagicMock(
            side_effect=[np.zeros((5, 3)) for _ in range(4)]
        )
        metric_vectors = np.array(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ]
        )
        engine_mod.get_TI_vectors = MagicMock(return_value=metric_vectors)

        electrodes = ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2")
        result = engine.compute_mti_field(electrodes, currents_mA=(2.0,) * 4)

        expected_keys = {
            "TestROI_TImax_ROI",
            "TestROI_TImean_ROI",
            "TestROI_TImean_GM",
            "TestROI_Focality",
            "TestROI_n_elements",
            "current_ch1_mA",
            "current_ch2_mA",
            "current_ch3_mA",
            "current_ch4_mA",
        }
        assert set(result) == expected_keys
        # roi_indices=[0,1,2] -> field_roi norms=[0.1,0.2,0.3]
        assert result["TestROI_TImax_ROI"] == pytest.approx(0.3)
        assert result["TestROI_TImean_ROI"] == pytest.approx(0.2)
        assert result["TestROI_n_elements"] == 3
        for i in range(1, 5):
            assert result[f"current_ch{i}_mA"] == 2.0

        assert engine_mod.TI.get_field.call_count == 4
        engine_mod.TI.get_field.assert_any_call(
            ["A1", "A2", 0.002], engine.leadfield, engine.idx_lf
        )
        engine_mod.TI.get_field.assert_any_call(
            ["D1", "D2", 0.002], engine.leadfield, engine.idx_lf
        )

        call = engine_mod.get_TI_vectors.call_args
        assert len(call.args[0]) == 4  # four channel fields, paired positionally

    def test_each_pair_is_driven_at_its_own_current(self):
        import tit.opt.mex.engine as engine_mod

        engine = _make_mex_engine()
        _setup_engine_fields(engine)

        # A linear stand-in for the leadfield: the field is the current times a
        # per-pair unit field, so each pair's scaling is visible downstream.
        units = {
            name: np.full((5, 3), float(k + 1))
            for k, name in enumerate(("A1", "B1", "C1", "D1"))
        }
        engine_mod.TI.get_field = MagicMock(
            side_effect=lambda spec, *_: spec[2] * units[spec[0]]
        )
        engine_mod.get_TI_vectors = MagicMock(return_value=np.zeros((5, 3)))

        electrodes = ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2")
        result = engine.compute_mti_field(electrodes, currents_mA=(2.0, 2.0, 1.0, 3.0))

        fields = engine_mod.get_TI_vectors.call_args.args[0]
        for pair, (name, current) in enumerate(
            zip(("A1", "B1", "C1", "D1"), (2.0, 2.0, 1.0, 3.0))
        ):
            np.testing.assert_allclose(fields[pair], current / 1000.0 * units[name])
        assert [result[f"current_ch{ch}_mA"] for ch in range(1, 5)] == [
            2.0,
            2.0,
            1.0,
            3.0,
        ]

    def test_zero_roi_elements_returns_zeros(self):
        import tit.opt.mex.engine as engine_mod

        engine = _make_mex_engine()
        _setup_engine_fields(engine)
        engine.roi_indices = np.array([], dtype=int)
        engine.roi_volumes = np.array([])

        engine_mod.TI.get_field = MagicMock(
            side_effect=[np.zeros((5, 3)) for _ in range(4)]
        )
        engine_mod.get_TI_vectors = MagicMock(return_value=np.zeros((5, 3)))

        electrodes = ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2")
        result = engine.compute_mti_field(electrodes, currents_mA=(1.0,) * 4)

        assert result["TestROI_TImax_ROI"] == 0.0
        assert result["TestROI_TImean_ROI"] == 0.0
        assert result["TestROI_n_elements"] == 0


# ---------------------------------------------------------------------------
# run_m_ex_search -- volumetric atlas ROI + MNI coordinate space
# ---------------------------------------------------------------------------


def _mex_pool_config(**overrides):
    from tit.opt.config import MExConfig

    defaults = dict(
        subject_id="001",
        leadfield_hdf="net_leadfield.hdf5",
        roi_name="motor.csv",
        electrodes=MExConfig.PoolElectrodes(electrodes=[f"E{i}" for i in range(1, 9)]),
    )
    defaults.update(overrides)
    return MExConfig(**defaults)


@pytest.mark.unit
class TestRunMExSearchAtlasAndMni:
    def _pm(self, tmp_path):
        (tmp_path / "aseg.mgz").touch()
        (tmp_path / "mask.nii.gz").touch()
        pm = MagicMock()
        pm.logs.return_value = str(tmp_path / "logs")
        pm.m_ex_search_run.return_value = str(tmp_path / "output")
        pm.rois.return_value = str(tmp_path / "rois")
        pm.m2m.return_value = str(tmp_path / "m2m_001")
        return pm

    @patch("tit.opt.mex.mex.process_and_save")
    @patch("tit.opt.mex.mex.MExSearchEngine")
    @patch("tit.opt.mex.mex.add_file_handler")
    @patch("tit.opt.mex.mex.get_path_manager")
    def test_atlas_targets_build_a_roi_list(
        self, mock_gpm, mock_afh, mock_engine_cls, mock_save, tmp_path
    ):
        pm = self._pm(tmp_path)
        mock_gpm.return_value = pm

        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine.run.return_value = {"m1": {}}
        mock_save.return_value = {"config_json_path": "/j", "csv_path": "/c"}

        from tit.opt.mex.mex import run_m_ex_search

        config = _mex_pool_config(
            roi_atlas=[{"atlas_path": str(tmp_path / "aseg.mgz"), "label": 53}]
        )
        run_m_ex_search(config)

        roi_arg = mock_engine_cls.call_args[0][1]
        assert roi_arg == [
            os.path.join(str(tmp_path / "rois"), "motor.csv"),
            (str(tmp_path / "aseg.mgz"), 53),
        ]

    @patch("tit.opt.mex.mex.process_and_save")
    @patch("tit.opt.mex.mex.MExSearchEngine")
    @patch("tit.opt.mex.mex.add_file_handler")
    @patch("tit.opt.mex.mex.get_path_manager")
    def test_atlas_only_target_reads_no_sphere_csv(
        self, mock_gpm, mock_afh, mock_engine_cls, mock_save, tmp_path
    ):
        """roi_names=[] must drop the sphere CSV, leaving an atlas-only target.

        This is what lets the GUI offer Atlas ROI in mTI mode. Previously mex.py
        built the sphere path from roi_name unconditionally, so a multipolar run
        always needed a spherical ROI; roi_name is now only a naming label when
        roi_names is empty, and no CSV is opened.
        """
        pm = self._pm(tmp_path)
        mock_gpm.return_value = pm

        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine.run.return_value = {"m1": {}}
        mock_save.return_value = {"config_json_path": "/j", "csv_path": "/c"}

        from tit.opt.mex.mex import run_m_ex_search

        config = _mex_pool_config(
            roi_name="Left-Hippocampus",
            roi_names=[],
            roi_atlas=[{"atlas_path": str(tmp_path / "aseg.mgz"), "label": 17}],
        )
        run_m_ex_search(config)

        roi_arg = mock_engine_cls.call_args[0][1]
        assert roi_arg == [(str(tmp_path / "aseg.mgz"), 17)]
        # the label still names the run, it just is not a file to read
        assert mock_engine_cls.call_args[0][2] == "Left-Hippocampus.csv"

    @patch("tit.opt.mex.mex.process_and_save")
    @patch("tit.opt.mex.mex.MExSearchEngine")
    @patch("tit.opt.mex.mex.add_file_handler")
    @patch("tit.opt.mex.mex.get_path_manager")
    def test_atlas_only_target_skips_the_mni_transform(
        self, mock_gpm, mock_afh, mock_engine_cls, mock_save, tmp_path
    ):
        """With no sphere centers there is nothing to transform.

        Guards the ``if roi_names and ...`` condition: dropping the roi_names
        half would send an empty list through simnibs.mni2subject_coords.
        """
        pm = self._pm(tmp_path)
        mock_gpm.return_value = pm

        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine.run.return_value = {"m1": {}}
        mock_save.return_value = {"config_json_path": "/j", "csv_path": "/c"}

        from tit.opt.mex.mex import run_m_ex_search

        config = _mex_pool_config(
            roi_names=[],
            roi_coordinate_space="mni",
            roi_atlas=[{"atlas_path": str(tmp_path / "aseg.mgz"), "label": 17}],
        )
        with patch("simnibs.mni2subject_coords") as mock_transform:
            run_m_ex_search(config)

        mock_transform.assert_not_called()
        assert mock_engine_cls.call_args[0][1] == [(str(tmp_path / "aseg.mgz"), 17)]

    @patch("tit.opt.mex.mex.process_and_save")
    @patch("tit.opt.mex.mex.MExSearchEngine")
    @patch("tit.opt.mex.mex.add_file_handler")
    @patch("tit.opt.mex.mex.get_path_manager")
    def test_no_roi_atlas_passes_bare_roi_file_string(
        self, mock_gpm, mock_afh, mock_engine_cls, mock_save, tmp_path
    ):
        """A default (spherical) config passes exactly one ROI target.

        mex.py used to hand the engine a bare string here and a list only when
        atlas entries existed. It now always passes a list, mirroring
        tit/opt/ex/ex.py -- ``ExSearchEngine`` (which ``MExSearchEngine``
        extends) accepts either, and one shape means one code path.
        """
        pm = self._pm(tmp_path)
        mock_gpm.return_value = pm

        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine.run.return_value = {}
        mock_save.return_value = {"config_json_path": "/j", "csv_path": "/c"}

        from tit.opt.mex.mex import run_m_ex_search

        run_m_ex_search(_mex_pool_config())

        roi_arg = mock_engine_cls.call_args[0][1]
        assert roi_arg == [os.path.join(str(tmp_path / "rois"), "motor.csv")]

    @patch("tit.opt.mex.mex.process_and_save")
    @patch("tit.opt.mex.mex.MExSearchEngine")
    @patch("tit.opt.mex.mex.add_file_handler")
    @patch("tit.opt.mex.mex.get_path_manager")
    def test_mni_space_transforms_roi_center(
        self, mock_gpm, mock_afh, mock_engine_cls, mock_save, tmp_path
    ):
        rois_dir = tmp_path / "rois"
        rois_dir.mkdir()
        with open(rois_dir / "motor.csv", "w", newline="") as f:
            csv.writer(f).writerow([5.0, 6.0, 7.0])

        pm = self._pm(tmp_path)
        mock_gpm.return_value = pm

        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine.run.return_value = {}
        mock_save.return_value = {"config_json_path": "/j", "csv_path": "/c"}

        from tit.opt.mex.mex import run_m_ex_search

        config = _mex_pool_config(roi_coordinate_space="mni")

        with patch(
            "simnibs.mni2subject_coords",
            return_value=np.array([[9.0, 8.0, 7.0]]),
        ) as mock_transform:
            run_m_ex_search(config)

        mock_transform.assert_called_once()
        assert mock_transform.call_args[0][1] == str(tmp_path / "m2m_001")

        [roi_arg] = mock_engine_cls.call_args[0][1]
        assert roi_arg != str(rois_dir / "motor.csv")
        with open(roi_arg) as f:
            row = next(csv.reader(f))
        assert [float(v) for v in row] == pytest.approx([9.0, 8.0, 7.0])

    @patch("tit.opt.mex.mex.process_and_save")
    @patch("tit.opt.mex.mex.MExSearchEngine")
    @patch("tit.opt.mex.mex.add_file_handler")
    @patch("tit.opt.mex.mex.get_path_manager")
    def test_subject_space_does_not_call_transform(
        self, mock_gpm, mock_afh, mock_engine_cls, mock_save, tmp_path
    ):
        pm = self._pm(tmp_path)
        mock_gpm.return_value = pm

        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine.run.return_value = {}
        mock_save.return_value = {"config_json_path": "/j", "csv_path": "/c"}

        from tit.opt.mex.mex import run_m_ex_search

        with patch("simnibs.mni2subject_coords") as mock_transform:
            run_m_ex_search(_mex_pool_config())

        mock_transform.assert_not_called()


class TestInferSymmetryEegCsv:
    """Leadfields are named ``{subject}_leadfield_{net}.hdf5``."""

    def test_net_name_is_taken_after_the_leadfield_marker(self, tmp_path, monkeypatch):
        from tit.opt.ex import symmetry

        eeg_dir = tmp_path / "eeg_positions"
        eeg_dir.mkdir()
        csv = eeg_dir / "EEG10-10_UI_Jurak_2007.csv"
        csv.write_text("Electrode,0,0,0,Cz\n")
        monkeypatch.setattr(
            symmetry, "canonical_template_coord_path", lambda name: None
        )

        assert (
            symmetry.infer_symmetry_eeg_csv(
                "/lf/ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5", eeg_dir
            )
            == csv
        )

    def test_unknown_naming_returns_none(self, tmp_path, monkeypatch):
        from tit.opt.ex import symmetry

        monkeypatch.setattr(
            symmetry, "canonical_template_coord_path", lambda name: None
        )
        assert symmetry.infer_symmetry_eeg_csv("/lf/weird.hdf5", tmp_path) is None

    def test_explicit_symmetry_eeg_csv_wins(self, tmp_path):
        from tit.opt.config import MExConfig
        from tit.opt.ex.symmetry import resolve_eeg_positions_csv

        csv = tmp_path / "custom.csv"
        csv.write_text("Electrode,0,0,0,Cz\n")
        cfg = MExConfig(
            subject_id="ernie",
            leadfield_hdf="/lf/ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5",
            roi_name="roi.csv",
            electrodes=MExConfig.PoolElectrodes(electrodes=[f"E{i}" for i in range(8)]),
            symmetry_eeg_csv=str(csv),
        )

        class PM:
            def eeg_positions(self, sid):
                return str(tmp_path / "missing")

        assert resolve_eeg_positions_csv(cfg, PM()) == csv


class TestMExCarrierDrive:
    """`frequencies_hz` on MExConfig, and which searches accept them.

    The guard is deliberately narrow, and this pins *why*. Pool mode enumerates one montage per
    symmetry *class*, and those classes come from the envelope's invariances under the positional,
    phase-aligned layout (``tit/opt/mex/logic.py``) -- a different wiring makes genuinely distinct
    montages look equivalent, so candidates would be silently skipped. Bucket mode, with or without
    ``symmetric_bucket``, only *restricts* which electrode combinations are generated; every
    surviving candidate is still scored with the real physics, so no wiring can invalidate it.
    """

    BUCKETS = {
        f"e{i}_{pole}": [f"E{i}{pole[0]}"]
        for i in range(1, 5)
        for pole in ("plus", "minus")
    }

    def _cfg(self, **kwargs):
        from tit.opt.config import MExConfig

        kwargs.setdefault("electrodes", MExConfig.BucketElectrodes(**self.BUCKETS))
        return MExConfig(
            subject_id="ernie",
            leadfield_hdf="/lf/ernie_leadfield_GSN-HydroCel-185.hdf5",
            roi_name="roi.csv",
            **kwargs,
        )

    def test_no_frequencies_keeps_the_positional_reading(self):
        assert self._cfg().carrier_wiring is None

    def test_positional_frequencies_resolve_to_two_carriers(self):
        wiring = self._cfg(
            frequencies_hz=[2000.0, 2010.0, 4000.0, 4010.0]
        ).carrier_wiring
        assert len(wiring.beat_pairs) == 2
        assert wiring.is_positional()
        assert not wiring.has_coherent_sum

    def test_shared_carrier_is_accepted_in_bucket_mode(self):
        # Two pairs on each carrier -- the wiring the positional model could not express.
        wiring = self._cfg(
            frequencies_hz=[2000.0, 2000.0, 2010.0, 2010.0]
        ).carrier_wiring
        assert len(wiring.groups) == 2
        assert len(wiring.beat_pairs) == 1
        assert wiring.has_coherent_sum
        assert not wiring.is_positional()

    def test_symmetric_bucket_search_accepts_a_shared_carrier(self):
        # Bucket symmetry narrows the candidate set geometrically; it collapses no
        # equivalence classes, so it stays valid under any wiring.
        cfg = self._cfg(
            symmetric_bucket=True,
            frequencies_hz=[2000.0, 2000.0, 2010.0, 2010.0],
        )
        assert cfg.carrier_wiring.has_coherent_sum

    def test_pool_mode_refuses_a_non_positional_wiring(self):
        from tit.opt.config import MExConfig

        with pytest.raises(ValueError, match="pool-mode search collapses montages"):
            self._cfg(
                electrodes=MExConfig.PoolElectrodes(
                    electrodes=[f"E{i}" for i in range(8)]
                ),
                frequencies_hz=[2000.0, 2000.0, 2010.0, 2010.0],
            )

    def test_pool_mode_accepts_the_positional_wiring(self):
        from tit.opt.config import MExConfig

        cfg = self._cfg(
            electrodes=MExConfig.PoolElectrodes(electrodes=[f"E{i}" for i in range(8)]),
            frequencies_hz=[2000.0, 2010.0, 4000.0, 4010.0],
        )
        assert cfg.carrier_wiring.is_positional()

    def test_no_difference_frequency_is_refused(self):
        # Every candidate would score exactly zero, so the ranking would be meaningless.
        with pytest.raises(ValueError, match="no difference frequency"):
            self._cfg(frequencies_hz=[2000.0] * 4)

    def test_wrong_arity_is_refused(self):
        with pytest.raises(ValueError, match="drives 4 electrode pairs"):
            self._cfg(frequencies_hz=[2000.0, 2010.0])

    def test_phase_is_not_a_config_field(self):
        # Polarity is electrode order; there is no separate phase input.
        with pytest.raises(TypeError):
            self._cfg(phases_deg=[0.0, 180.0, 0.0, 0.0])

    def test_drive_round_trips_through_config_io(self):
        from tit.config_io import deserialize_config, serialize_config
        from tit.opt.config import MExConfig

        cfg = self._cfg(frequencies_hz=[2000.0, 2000.0, 2010.0, 2010.0])
        data = serialize_config(cfg)
        assert data["frequencies_hz"] == [2000.0, 2000.0, 2010.0, 2010.0]
        assert "phases_deg" not in data
        back = deserialize_config(MExConfig, data)
        assert back.frequencies_hz == cfg.frequencies_hz


@pytest.mark.unit
class TestComputeMtiFieldWithWiring:
    """The wired mEx scoring path runs end to end (it crashed with frequencies set)."""

    def _run(self, monkeypatch, units, frequencies):
        import tit.opt.mex.engine as engine_mod
        from tit.calc import get_TI_vectors
        from tit.wiring import derive_wiring

        engine = _make_mex_engine()
        _setup_engine_fields(engine)
        engine.wiring = derive_wiring(frequencies)
        monkeypatch.setattr(
            engine_mod.TI,
            "get_field",
            MagicMock(side_effect=[np.tile(u, (5, 1)) for u in units]),
        )
        monkeypatch.setattr(engine_mod, "get_TI_vectors", get_TI_vectors)
        electrodes = ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2")
        return engine.compute_mti_field(electrodes, currents_mA=(1.0,) * 4)

    def test_shared_carriers_are_summed_before_they_beat(self, monkeypatch):
        # Pairs A,B at one frequency and C,D at the other: 1.0 along x beats
        # against 1.0 along x, a depth of 2 * 1.0. (get_field is mocked, so the
        # current it would apply is not; the fields are used as given.)
        units = ([0.6, 0, 0], [0.4, 0, 0], [0.5, 0, 0], [0.5, 0, 0])
        result = self._run(monkeypatch, units, [2000.0, 2000.0, 2010.0, 2010.0])
        assert result["TestROI_TImax_ROI"] == pytest.approx(2.0)

    def test_reversing_a_shared_pair_changes_the_score(self, monkeypatch):
        # Electrode order is the only polarity control: pair B reversed (its
        # field negated) cancels against A before the beat, 0.2 against 1.0.
        units = ([0.6, 0, 0], [-0.4, 0, 0], [0.5, 0, 0], [0.5, 0, 0])
        result = self._run(monkeypatch, units, [2000.0, 2000.0, 2010.0, 2010.0])
        assert result["TestROI_TImax_ROI"] == pytest.approx(2 * 0.2)
