"""Structural tests for tit.wiring -- grouping, beat matching and refusal.

These assert the *structure* the wiring derives. That the structure describes
the real field is a separate claim, checked against an independent time-domain
simulation in ``tests/numerical/test_sci10_shared_frequency.py``.
"""

import logging

import numpy as np
import pytest

from tit.wiring import (
    INCOHERENCE_MARGIN,
    MIN_BEAT_HZ,
    WiringError,
    coherent_amplitudes,
    derive_wiring,
    envelope_inputs,
)

pytestmark = pytest.mark.unit


def _fields(*vectors):
    """One single-element (1, 3) field per channel."""
    return [np.array([v], dtype=float) for v in vectors]


class TestGrouping:
    """Channels at one frequency form one coherent group."""

    def test_distinct_frequencies_are_separate_groups(self):
        w = derive_wiring([2000.0, 2010.0])
        assert len(w.groups) == 2
        assert [g.members for g in w.groups] == [(0,), (1,)]
        assert not w.has_coherent_sum

    def test_identical_frequencies_share_a_group(self):
        w = derive_wiring([2000.0, 2000.0])
        assert len(w.groups) == 1
        assert w.groups[0].members == (0, 1)
        assert w.has_coherent_sum

    def test_groups_are_ordered_by_frequency_not_input_order(self):
        w = derive_wiring([4000.0, 2000.0, 4010.0, 2010.0])
        assert [g.frequency_hz for g in w.groups] == [2000.0, 2010.0, 4000.0, 4010.0]
        assert [g.members for g in w.groups] == [(1,), (3,), (0,), (2,)]

    def test_phase_is_not_part_of_the_drive(self):
        # Polarity is electrode order; derive_wiring takes frequencies only.
        with pytest.raises(TypeError):
            derive_wiring([2000.0, 2000.0], [0.0, 180.0])


class TestBeatMatching:
    """Which carriers beat, and at what one difference frequency."""

    def test_two_carriers_beat(self):
        w = derive_wiring([2000.0, 2010.0])
        assert w.beat_pairs == ((0, 1),)
        assert w.unpaired == ()
        assert w.delta_f_hz == pytest.approx(10.0)
        assert w.has_modulation

    def test_one_frequency_has_no_beat_at_all(self):
        w = derive_wiring([2000.0, 2000.0])
        assert w.beat_pairs == ()
        assert w.unpaired == (0,)
        assert w.delta_f_hz is None
        assert not w.has_modulation

    def test_mti_pairs_each_carrier_with_its_own_partner(self):
        w = derive_wiring([2000.0, 2010.0, 4000.0, 4010.0])
        assert w.beat_pairs == ((0, 1), (2, 3))
        assert w.unpaired == ()
        assert w.delta_f_hz == pytest.approx(10.0)

    def test_a_lone_far_carrier_is_unpaired_not_an_error(self):
        w = derive_wiring([2000.0, 2010.0, 9000.0])
        assert w.beat_pairs == ((0, 1),)
        assert w.unpaired == (2,)

    def test_shared_carrier_plus_a_beat_partner(self):
        # Two channels on one frequency, two on another: the classic case the
        # positional model could not express.
        w = derive_wiring([2000.0, 2000.0, 2010.0, 2010.0])
        assert len(w.groups) == 2
        assert w.beat_pairs == ((0, 1),)
        assert w.has_coherent_sum and w.has_modulation

    def test_the_positional_layout_is_recognised(self):
        assert derive_wiring([2000.0, 2010.0, 4000.0, 4010.0]).is_positional()
        assert not derive_wiring([2000.0, 2000.0, 2010.0, 2010.0]).is_positional()


class TestRefusal:
    """Assignments the (P, Q) envelope cannot represent are rejected, not guessed."""

    def test_evenly_spaced_triple_is_ambiguous(self):
        # 2000/2010/2020 admits two pairings and really does beat at both 10 and
        # 20 Hz; a single delta f cannot describe it.
        with pytest.raises(WiringError, match="Ambiguous carrier pairing"):
            derive_wiring([2000.0, 2010.0, 2020.0])

    def test_a_second_difference_frequency_is_rejected(self):
        with pytest.raises(WiringError, match="different difference"):
            derive_wiring([2000.0, 2010.0, 4000.0, 4020.0])

    def test_non_positive_frequency(self):
        with pytest.raises(WiringError, match="finite and positive"):
            derive_wiring([2000.0, 0.0])

    def test_non_finite_frequency(self):
        with pytest.raises(WiringError, match="finite and positive"):
            derive_wiring([2000.0, float("nan")])

    def test_empty(self):
        with pytest.raises(WiringError, match="At least one"):
            derive_wiring([])


@pytest.fixture
def wiring_logs(caplog):
    """Capture ``tit.wiring`` warnings.

    ``setup_logging`` sets ``propagate = False`` on the ``tit`` logger
    (``tit/logger.py``, "never bubble to root/terminal"), so caplog's root
    handler never sees these records. The handler has to go on the module
    logger itself.
    """
    logger = logging.getLogger("tit.wiring")
    previous_level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.WARNING)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)


class TestWarnings:
    """Cases the dichotomy fits poorly are flagged, not refused."""

    def test_sub_hertz_beat_warns(self, wiring_logs):
        derive_wiring([2000.0, 2000.0 + MIN_BEAT_HZ / 10.0])
        assert "below" in wiring_logs.text
        assert "stimulation epoch" in wiring_logs.text

    def test_carriers_too_close_to_be_incoherent_warn(self, wiring_logs):
        delta = 10.0
        near = 2000.0 + (INCOHERENCE_MARGIN / 2.0) * delta
        derive_wiring([2000.0, 2000.0 + delta, near, near + delta])
        assert "mutually incoherent" in wiring_logs.text

    def test_well_separated_carriers_do_not_warn(self, wiring_logs):
        derive_wiring([2000.0, 2010.0, 4000.0, 4010.0])
        assert wiring_logs.text == ""


class TestCoherentAmplitudes:
    """The vector pre-sum -- the only thing that makes electrode order matter."""

    def test_same_frequency_same_polarity_adds(self):
        w = derive_wiring([2000.0, 2000.0])
        amps = coherent_amplitudes(_fields([1, 0, 0], [0.8, 0, 0]), w)
        assert amps[0].field[0] == pytest.approx([1.8, 0.0, 0.0])

    def test_reversing_one_pair_cancels(self):
        # Swapping a pair's electrodes negates its FEM field -- the only way
        # polarity enters -- and at one frequency that cancels, not adds.
        w = derive_wiring([2000.0, 2000.0])
        amps = coherent_amplitudes(_fields([1, 0, 0], [-0.8, 0, 0]), w)
        assert amps[0].field[0] == pytest.approx([0.2, 0.0, 0.0])
        assert amps[0].mean_square[0] == pytest.approx(0.04)

    def test_distinct_frequencies_are_not_summed(self):
        w = derive_wiring([2000.0, 2010.0])
        amps = coherent_amplitudes(_fields([1, 0, 0], [0.8, 0, 0]), w)
        assert len(amps) == 2
        assert amps[0].field[0] == pytest.approx([1.0, 0.0, 0.0])
        assert amps[1].field[0] == pytest.approx([0.8, 0.0, 0.0])

    def test_field_count_must_match_the_wiring(self):
        w = derive_wiring([2000.0, 2010.0])
        with pytest.raises(WiringError, match="describes 2 channels"):
            coherent_amplitudes(_fields([1, 0, 0]), w)

    def test_shapes_must_agree(self):
        w = derive_wiring([2000.0, 2010.0])
        bad = [np.zeros((1, 3)), np.zeros((2, 3))]
        with pytest.raises(WiringError, match="identical shape"):
            coherent_amplitudes(bad, w)


class TestEnvelopeInputs:
    """The packing handed to tit.calc.

    The contract is deliberately the plain field list ``get_TI_vectors``
    already takes: a carrier with no beat partner is emitted
    paired with a zero field, which contributes its power to ``P`` and nothing
    to ``Q``. That is what lets ``tit.calc`` stay completely unmodified.
    """

    def test_two_beating_carriers_are_one_consecutive_pair(self):
        w = derive_wiring([2000.0, 2010.0])
        packed = envelope_inputs(_fields([1, 0, 0], [0, 1, 0]), w)
        assert len(packed) == 2
        assert packed[0][0] == pytest.approx([1.0, 0.0, 0.0])
        assert packed[1][0] == pytest.approx([0.0, 1.0, 0.0])

    def test_no_beat_emits_the_coherent_sum_against_a_zero_partner(self):
        w = derive_wiring([2000.0, 2000.0])
        packed = envelope_inputs(_fields([1, 0, 0], [0.8, 0, 0]), w)
        assert len(packed) == 2
        # The two channels summed as vectors first ...
        assert packed[0][0] == pytest.approx([1.8, 0.0, 0.0])
        # ... and partnered with zero, so there is nothing to beat against.
        assert packed[1][0] == pytest.approx([0.0, 0.0, 0.0])

    def test_non_beating_carrier_rides_alongside_a_beat(self):
        w = derive_wiring([2000.0, 2010.0, 9000.0])
        packed = envelope_inputs(_fields([1, 0, 0], [0, 1, 0], [0, 0, 1]), w)
        # One real carrier pair, then the lone carrier against zero.
        assert len(packed) == 4
        assert packed[2][0] == pytest.approx([0.0, 0.0, 1.0])
        assert packed[3][0] == pytest.approx([0.0, 0.0, 0.0])

    def test_packing_is_accepted_by_an_unmodified_tit_calc(self):
        """The whole point: no change to tit.calc is required."""
        from tit.calc import get_TI_vectors

        beat = derive_wiring([2000.0, 2010.0])
        none = derive_wiring([2000.0, 2000.0])
        fields = _fields([1, 0, 0], [0.8, 0, 0])

        packed = envelope_inputs(fields, beat)
        assert float(
            np.linalg.norm(get_TI_vectors(packed), axis=1)[0]
        ) == pytest.approx(1.6)

        packed = envelope_inputs(fields, none)
        assert float(
            np.linalg.norm(get_TI_vectors(packed), axis=1)[0]
        ) == pytest.approx(0.0)


class TestBeatUpperBound:
    """A separation too wide to be an envelope is not treated as one."""

    def test_widely_separated_carriers_do_not_beat(self):
        w = derive_wiring([2000.0, 9000.0])
        assert w.beat_pairs == ()
        assert w.unpaired == (0, 1)
        assert w.delta_f_hz is None
        assert not w.has_modulation

    def test_two_far_apart_carrier_pairs_still_beat_within_themselves(self):
        w = derive_wiring([2000.0, 2010.0, 9000.0, 9010.0])
        assert w.beat_pairs == ((0, 1), (2, 3))
        assert w.delta_f_hz == pytest.approx(10.0)
