"""SCI-11 -- the phase-locked peak carrier field, checked against the time domain.

``hf_peak`` is the maximum instantaneous carrier field over *every* relative
carrier phase: an upper bound that assumes the carriers drift into any
alignment. A phase-locked drive -- one clock, exact frequencies, every source
starting at phase 0 -- visits only the phases its frequencies allow, so its
actual peak can be lower. ``hf_peak_phaselocked`` is that actual peak.

Every claim here is measured against an **independent reference**: ``E(t)``
built from the fields on a dense time grid over one full common period, and its
largest magnitude read off directly. No toolbox code builds the reference.

What is pinned:

1. it matches the reference (to the reference's own sampling error) for two,
   three, four and six carriers, with and without shared frequencies;
2. it never exceeds ``hf_peak``;
3. it equals ``hf_peak`` for two commensurate carriers and ``|E1 +/- E2|`` for
   one shared frequency;
4. reversing one pair of a four-carrier drive lowers it where ``hf_peak`` does
   not move -- the polarity test that motivated it;
5. the common period is found correctly, and sets with no usable period are
   refused.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit

#: Reference samples per cycle of the fastest carrier. Its own under-estimate of
#: the peak is below (pi/400)^2/2 ~ 3e-5 relative.
_REF_SAMPLES_PER_CYCLE = 400


def _reference_peak(fields, freqs):
    """max_t |sum_i E_i cos(2 pi f_i t)| on a dense grid over one full common period."""
    from tit.fields import phase_locked_period

    period = phase_locked_period(freqs)
    n = int(period * max(freqs) * _REF_SAMPLES_PER_CYCLE) + 1
    t = np.linspace(0.0, period, n)
    cos_t = np.cos(2.0 * np.pi * np.asarray(freqs, float)[:, None] * t[None, :])
    out = np.empty(fields[0].shape[0])
    for e in range(fields[0].shape[0]):
        amps = np.stack([f[e] for f in fields])  # (K, 3)
        et = cos_t.T @ amps  # (T, 3)
        out[e] = np.linalg.norm(et, axis=1).max()
    return out


def _random_fields(k, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n, 3)) for _ in range(k)]


# --------------------------------------------------------------------------
# 1-2. Agreement with the time domain; never above hf_peak.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freqs",
    [
        (2000.0, 2010.0),
        (2000.0, 2010.0, 4000.0),
        (2000.0, 2010.0, 3000.0, 3010.0),  # the mTI polarity-test drive
        (2000.0, 2000.0, 2010.0),  # a shared carrier plus a beat partner
        (2000.0, 2010.0, 3000.0, 3010.0, 2500.0, 2510.0),
    ],
)
def test_matches_the_time_domain_peak_and_never_exceeds_hf_peak(freqs):
    from tit.fields import hf_peak, hf_peak_phaselocked

    fields = _random_fields(len(freqs), seed=len(freqs))
    got = hf_peak_phaselocked(*fields, frequencies_hz=freqs)
    ref = _reference_peak(fields, freqs)
    # The reference is itself a sampled maximum, so it can only sit slightly below the truth.
    assert np.all(got >= ref * (1 - 1e-9)), "phase-locked peak below the sampled time-domain peak"
    assert np.all(np.abs(got - ref) <= 1e-4 * ref)
    assert np.all(got <= hf_peak(*fields) * (1 + 1e-9))


# --------------------------------------------------------------------------
# 3. Where it must coincide with the closed forms.
# --------------------------------------------------------------------------


def test_two_commensurate_carriers_reach_hf_peak():
    """Both sign patterns occur within one common period, so nothing is lost to locking."""
    from tit.fields import hf_peak, hf_peak_phaselocked

    fields = _random_fields(2, seed=7)
    got = hf_peak_phaselocked(*fields, frequencies_hz=(2000.0, 2010.0))
    assert np.allclose(got, hf_peak(*fields), rtol=1e-9)


def test_one_shared_frequency_is_the_coherent_vector_sum():
    """At one frequency the fields add at every instant: |E1 + E2|, and |E1 - E2| reversed."""
    from tit.fields import hf_peak_phaselocked

    e1, e2 = _random_fields(2, seed=11)
    same = hf_peak_phaselocked(e1, e2, frequencies_hz=(10000.0, 10000.0))
    reversed_ = hf_peak_phaselocked(e1, -e2, frequencies_hz=(10000.0, 10000.0))
    assert np.allclose(same, np.linalg.norm(e1 + e2, axis=1), rtol=1e-12)
    assert np.allclose(reversed_, np.linalg.norm(e1 - e2, axis=1), rtol=1e-12)


# --------------------------------------------------------------------------
# 4. The polarity test: hf_peak cannot see it, the phase-locked peak can.
# --------------------------------------------------------------------------


def test_reversing_one_pair_of_a_four_carrier_drive_lowers_the_locked_peak():
    """Nearly parallel channels, pair 4 reversed: carrier 2's envelope trough meets carrier 1's peak."""
    from tit.fields import hf_peak, hf_peak_phaselocked

    freqs = (2000.0, 2010.0, 3000.0, 3010.0)
    rng = np.random.default_rng(3)
    base = rng.normal(size=(30, 3))
    fields = [base + 0.1 * rng.normal(size=base.shape) for _ in range(4)]
    flipped = fields[:3] + [-fields[3]]

    in_phase = hf_peak_phaselocked(*fields, frequencies_hz=freqs)
    anti = hf_peak_phaselocked(*flipped, frequencies_hz=freqs)
    # hf_peak is blind to the reversal by construction ...
    assert np.allclose(hf_peak(*fields), hf_peak(*flipped), rtol=1e-12)
    # ... the in-phase drive reaches it (every source peaks together at t = 0) ...
    assert np.allclose(in_phase, hf_peak(*fields), rtol=1e-9)
    # ... and the reversed one does not, as the time-domain reference confirms.
    ref_anti = _reference_peak(flipped, freqs)
    assert np.all(np.abs(anti - ref_anti) <= 1e-4 * ref_anti)
    assert np.all(anti < 0.95 * in_phase)


# --------------------------------------------------------------------------
# 5. The common period.
# --------------------------------------------------------------------------


def test_common_period_of_the_polarity_test_drive_is_a_tenth_of_a_second():
    from tit.fields import phase_locked_period

    assert phase_locked_period((2000.0, 2010.0, 3000.0, 3010.0)) == pytest.approx(0.1)
    assert phase_locked_period((10000.0, 10000.0)) == pytest.approx(1e-4)
    assert phase_locked_period((2000.5, 2010.0)) == pytest.approx(2.0)


@pytest.mark.parametrize("freqs", [(2000.0, 2000.001), (2000.0, 2000.0004)])
def test_frequencies_with_no_usable_common_period_are_refused(freqs):
    from tit.fields import phase_locked_period

    with pytest.raises(ValueError):
        phase_locked_period(freqs)


def test_field_and_frequency_counts_must_agree():
    from tit.fields import hf_peak_phaselocked

    with pytest.raises(ValueError, match="one per pair"):
        hf_peak_phaselocked(*_random_fields(3), frequencies_hz=(2000.0, 2010.0))
