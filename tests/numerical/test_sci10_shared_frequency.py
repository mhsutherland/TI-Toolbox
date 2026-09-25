"""SCI-10 -- explicit carrier frequency, and the coherent sum it makes possible.

Until ``tit.wiring`` there was no way to say which electrode pairs share a
frequency, so every path *assumed* they all differed.  Under that assumption the
shipped metrics are correct, and this file does not dispute them.  What it pins
is the regime the assumption excluded, which
``test_sci07_exposure_channels.test_shared_frequency_would_need_a_coherent_presum_and_never_occurs``
named as its own specification:

    "If two electrode pairs were driven *phase-locked at one frequency* ... the
    true time-averaged |E|^2 would be |E_a + E_b|^2 / 2, not
    (|E_a|^2 + |E_b|^2) / 2 ... If a future montage regains one, the coherent
    pre-sum belongs in ``tit.fields`` and this test is the specification."

Two consequences follow, and both are measured here rather than argued:

1. **There is no envelope.**  Two channels at one frequency do not beat, so the
   modulation depth is exactly zero -- while the sign-agnostic K=1 closed form,
   applied positionally, reports a large non-zero value. ``tit.calc`` is not
   modified for this: the lone carrier is simply handed to it paired with a
   zero field, which raises ``P`` and leaves ``Q`` at zero.
2. **Electrode order stops being free.**  ``+/- +/-`` and ``+/- -/+`` are the
   same montage to every frequency-blind metric, and physically different: one
   sums, the other cancels.  Electrode order is the only polarity control --
   reversing a pair negates its FEM field, which is a 180-degree phase -- so the
   reversed montage is built here by negating that channel's field.

Everything is checked against the **independent time-domain reference** already
used by SCI-07 -- ``E(t)`` summed on a fine grid over a full common period, from
which the true time-averaged ``|E|^2``, the true peak ``|E(t)|`` and the true
modulation depth are measured.  No toolbox code builds the reference.
"""

import numpy as np
import pytest

from tests.numerical.test_sci07_exposure_channels import (
    _FREQS,
    _as_rows,
    _time_series,
    _true_mean_square,
    _true_peak,
)

pytestmark = pytest.mark.unit


def _true_modulation_depth(fields, freqs, phases, direction):
    """Peak-to-trough swing of the projected signal's envelope, measured.

    The envelope of a beat is the swing of ``|E(t) . n|``'s upper envelope; for
    a two-carrier beat that is ``(max - min)`` of the rectified signal's local
    peaks.  Taking the difference between the largest and smallest *local
    maxima* of ``|E(t) . n|`` measures the modulation depth without assuming
    any closed form.
    """
    e_t = _time_series(fields, freqs, phases)
    sig = np.abs(e_t @ np.asarray(direction, float))
    # Local maxima of the rectified carrier: the envelope samples.
    peaks = sig[1:-1][(sig[1:-1] >= sig[:-2]) & (sig[1:-1] >= sig[2:])]
    if peaks.size == 0:
        return 0.0
    return float(peaks.max() - peaks.min())


# --------------------------------------------------------------------------
# 1. No difference frequency means no envelope.
# --------------------------------------------------------------------------


def test_one_frequency_has_no_modulation_but_the_positional_path_reports_one():
    """The regression that motivates the feature, both halves measured."""
    from tit.calc import get_TI_vectors
    from tit.wiring import derive_wiring, envelope_inputs

    e_a = np.array([1.0, 0.0, 0.0])
    e_b = np.array([0.8, 0.0, 0.0])
    n_hat = np.array([1.0, 0.0, 0.0])

    # Truth: one frequency, so the carrier amplitude never varies.
    measured = _true_modulation_depth(
        [e_a, e_b], (_FREQS[0], _FREQS[0]), (0.0, 0.0), n_hat
    )
    assert measured == pytest.approx(0.0, abs=1e-9)

    fields = _as_rows(e_a, e_b)

    # Frequency-aware: exactly zero.
    wiring = derive_wiring([_FREQS[0], _FREQS[0]])
    packed = envelope_inputs(fields, wiring)
    wired = np.linalg.norm(get_TI_vectors(packed), axis=1)
    assert float(wired[0]) == pytest.approx(0.0, abs=1e-12)

    # Frequency-blind (positional): a large number for a beat that is not there.
    positional = np.linalg.norm(get_TI_vectors(fields), axis=1)
    assert float(positional[0]) == pytest.approx(1.6, rel=1e-12)


def test_a_real_difference_frequency_still_matches_the_measured_beat():
    """Introducing delta f restores the envelope, and the wired value is it."""
    from tit.calc import get_TI_vectors
    from tit.wiring import derive_wiring, envelope_inputs

    e_a = np.array([1.0, 0.0, 0.0])
    e_b = np.array([0.8, 0.0, 0.0])
    n_hat = np.array([1.0, 0.0, 0.0])

    measured = _true_modulation_depth(
        [e_a, e_b], (_FREQS[0], _FREQS[1]), (0.0, 0.0), n_hat
    )
    # Two colinear carriers beat between |a|+|b| and ||a|-|b||, a swing of
    # 2*min(|a|,|b|) = 1.6.
    assert measured == pytest.approx(1.6, rel=1e-3)

    wiring = derive_wiring([_FREQS[0], _FREQS[1]])
    packed = envelope_inputs(_as_rows(e_a, e_b), wiring)
    wired = np.linalg.norm(get_TI_vectors(packed), axis=1)
    assert float(wired[0]) == pytest.approx(measured, rel=1e-3)


def test_wired_and_positional_agree_when_the_frequencies_really_do_differ():
    """The new path changes nothing in the regime the old assumption held."""
    from tit.calc import get_TI_vectors
    from tit.wiring import derive_wiring, envelope_inputs

    rng = np.random.default_rng(20260918)
    vecs = [rng.normal(size=3) for _ in range(4)]
    fields = _as_rows(*vecs)

    wiring = derive_wiring([_FREQS[0], _FREQS[1], _FREQS[3], _FREQS[3] + 10.0])
    packed = envelope_inputs(fields, wiring)
    assert len(packed) == 4

    wired = get_TI_vectors(packed)
    positional = get_TI_vectors(fields)
    assert np.allclose(wired, positional, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# 2. Electrode order matters once the frequency is shared.
# --------------------------------------------------------------------------


def test_montage_order_changes_exposure_at_one_frequency():
    """``+/- +/-`` vs ``+/- -/+``: constructive against destructive, measured."""
    from tit.fields import hf_peak, hf_sar
    from tit.wiring import derive_wiring

    e_a = np.array([1.0, 0.0, 0.0])
    e_b = np.array([0.8, 0.0, 0.0])
    fields = _as_rows(e_a, e_b)
    reversed_fields = _as_rows(e_a, -e_b)  # pair B's electrodes swapped
    same = (_FREQS[0], _FREQS[0])
    wiring = derive_wiring([_FREQS[0], _FREQS[0]])

    # Truth, from the time domain.
    ms_aligned = _true_mean_square([e_a, e_b], same, (0.0, 0.0))
    ms_reversed = _true_mean_square([e_a, -e_b], same, (0.0, 0.0))
    pk_aligned = _true_peak([e_a, e_b], same, (0.0, 0.0))
    pk_reversed = _true_peak([e_a, -e_b], same, (0.0, 0.0))

    assert ms_aligned == pytest.approx(1.8**2 / 2.0, rel=1e-9)
    assert ms_reversed == pytest.approx(0.2**2 / 2.0, rel=1e-9)

    # Frequency-aware metrics reproduce both, and they differ.
    assert float(hf_sar(*fields, wiring=wiring)[0]) == pytest.approx(
        2.0 * ms_aligned, rel=1e-12
    )
    assert float(hf_sar(*reversed_fields, wiring=wiring)[0]) == pytest.approx(
        2.0 * ms_reversed, rel=1e-12
    )
    assert float(hf_peak(*fields, wiring=wiring)[0]) == pytest.approx(
        pk_aligned, rel=1e-9
    )
    assert float(hf_peak(*reversed_fields, wiring=wiring)[0]) == pytest.approx(
        pk_reversed, rel=1e-9
    )

    # Without a wiring both orders collapse to the same, frequency-blind answer
    # -- which is why the montage order looked free.
    assert float(hf_sar(*fields)[0]) == pytest.approx(
        float(hf_sar(*reversed_fields)[0]), rel=1e-12
    )
    assert float(hf_peak(*fields)[0]) == pytest.approx(
        float(hf_peak(*reversed_fields)[0]), rel=1e-12
    )


def test_coherent_sum_beats_incoherent_addition_for_aligned_fields():
    """The cross term the incoherent sum drops, stated as a factor."""
    from tit.fields import hf_sar
    from tit.wiring import derive_wiring

    e = np.array([1.0, 0.0, 0.0])
    fields = _as_rows(e, e)
    wiring = derive_wiring([_FREQS[0], _FREQS[0]])

    incoherent = float(hf_sar(*fields)[0])          # |E|^2 + |E|^2 = 2
    coherent = float(hf_sar(*fields, wiring=wiring)[0])  # |2E|^2 = 4
    assert incoherent == pytest.approx(2.0, rel=1e-12)
    assert coherent == pytest.approx(4.0, rel=1e-12)
    # Two equal aligned fields: exactly the factor of two SCI-07 predicted.
    assert coherent / incoherent == pytest.approx(2.0, rel=1e-12)


# --------------------------------------------------------------------------
# 3. Mixed wiring: coherent within a frequency, beating across.
# --------------------------------------------------------------------------


def test_shared_carrier_pairs_beat_after_they_are_summed():
    """Four channels on two frequencies -- the montage the old model could not express."""
    from tit.calc import get_TI_vectors
    from tit.wiring import derive_wiring, envelope_inputs

    f_lo, f_hi = _FREQS[0], _FREQS[1]
    a1 = np.array([0.6, 0.0, 0.0])
    a2 = np.array([0.4, 0.0, 0.0])
    b1 = np.array([0.0, 0.5, 0.0])
    b2 = np.array([0.0, 0.5, 0.0])

    wiring = derive_wiring([f_lo, f_lo, f_hi, f_hi])
    assert len(wiring.groups) == 2 and wiring.has_coherent_sum

    packed = envelope_inputs(_as_rows(a1, a2, b1, b2), wiring)
    assert len(packed) == 2
    # The groups summed first: 1.0 along x, 1.0 along y.
    assert packed[0][0] == pytest.approx([1.0, 0.0, 0.0])
    assert packed[1][0] == pytest.approx([0.0, 1.0, 0.0])

    wired = np.linalg.norm(get_TI_vectors(packed), axis=1)
    # Equal orthogonal carriers: the same envelope as any single such pair.
    direct = np.linalg.norm(
        get_TI_vectors(_as_rows(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))),
        axis=1,
    )
    assert float(wired[0]) == pytest.approx(float(direct[0]), rel=1e-12)


def test_a_non_beating_carrier_raises_power_without_raising_the_envelope():
    """An unpaired carrier lifts P, leaves Q alone, and so lowers the depth."""
    from tit.calc import get_TI_vectors
    from tit.wiring import derive_wiring, envelope_inputs

    e_a = np.array([1.0, 0.0, 0.0])
    e_b = np.array([1.0, 0.0, 0.0])
    e_c = np.array([0.0, 0.0, 1.5])  # far-off carrier, nothing to beat with

    beat_only = derive_wiring([_FREQS[0], _FREQS[1]])
    with_extra = derive_wiring([_FREQS[0], _FREQS[1], 9000.0])
    assert with_extra.unpaired == (2,)

    p0 = envelope_inputs(_as_rows(e_a, e_b), beat_only)
    p1 = envelope_inputs(_as_rows(e_a, e_b, e_c), with_extra)

    md0 = float(np.linalg.norm(get_TI_vectors(p0), axis=1)[0])
    md1 = float(np.linalg.norm(get_TI_vectors(p1), axis=1)[0])
    assert md1 < md0

    # ... while the carrier exposure rises, because the extra carrier is real.
    from tit.fields import hf_sar

    sar0 = float(hf_sar(*_as_rows(e_a, e_b), wiring=beat_only)[0])
    sar1 = float(hf_sar(*_as_rows(e_a, e_b, e_c), wiring=with_extra)[0])
    assert sar1 == pytest.approx(sar0 + 1.5**2, rel=1e-12)


# --------------------------------------------------------------------------
# 4. Nothing moves when no frequencies are given.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_fields", [2, 4, 6])
def test_absent_wiring_is_bit_identical_to_the_positional_result(n_fields):
    """``wiring=None`` is the shipped behaviour, unchanged."""
    from tit.calc import get_TI_avg, get_TI_vectors
    from tit.fields import hf_peak, hf_sar

    rng = np.random.default_rng(1000 + n_fields)
    fields = [rng.normal(size=(6, 3)) for _ in range(n_fields)]

    # These are the exact calls tit.sim makes today; they must not have moved.
    assert np.all(np.isfinite(get_TI_vectors(fields)))
    assert np.all(np.isfinite(get_TI_avg(fields)))
    assert np.array_equal(hf_sar(*fields), hf_sar(*fields, wiring=None))
    assert np.array_equal(hf_peak(*fields), hf_peak(*fields, wiring=None))
