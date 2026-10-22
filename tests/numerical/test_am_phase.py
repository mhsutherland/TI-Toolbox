"""AM-phase metrics (``tit.tools.am_phase``) against a time-domain reference.

Two carriers drive ``E(t) = E1 cos(theta) + E2 cos(theta + psi)``.  The
reference below builds that signal on a fine ``(psi, theta)`` grid and reads
the ``|E|`` envelope off it directly -- its maximum over ``theta`` at each
``psi`` -- without reusing toolbox code.  The claims checked:

* the envelope peaks at ``psi = 0`` when ``E1 . E2 > 0`` and at ``psi = pi``
  when ``E1 . E2 < 0`` (the label);
* the envelope maximum, minimum and depth match the closed form and the
  golden-section search;
* the depth does **not** vanish for perpendicular fields: it is
  ``(sqrt(2) - 1) a`` for equal magnitudes ``a``, with equal peaks at 0 and pi;
* reversing one pair swaps the label and leaves the depth unchanged.
"""

import numpy as np
import pytest

from tit.tools.am_phase import (
    LABEL_PSI0,
    LABEL_PSIPI,
    am_phase_metrics,
    brute_force_check,
)

pytestmark = pytest.mark.unit

_STEP_DEG = 0.25
_TH = np.deg2rad(np.arange(0.0, 360.0, _STEP_DEG))
_PS = np.deg2rad(np.arange(0.0, 360.0, _STEP_DEG))


def _envelope(e1, e2):
    """``max_theta |E(theta, psi)|`` for every ``psi`` on the grid."""
    ct = np.cos(_TH)[None, :]
    ctp = np.cos(_TH[None, :] + _PS[:, None])
    sq = sum((e1[i] * ct + e2[i] * ctp) ** 2 for i in range(3))
    return np.sqrt(sq).max(axis=1)


def _random_fields(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.normal(size=(n, 3))


def test_label_is_where_the_envelope_peaks():
    E1, E2 = _random_fields(200, seed=1)
    met = am_phase_metrics(E1, E2)
    for k in range(len(E1)):
        env = _envelope(E1[k], E2[k])
        e0, epi = env[0], env[len(_PS) // 2]
        if abs(e0 - epi) < 1e-3 * env.max():
            continue  # near 90 degrees: two equal peaks, either label is right
        peak = 0 if e0 > epi else 1
        expected = LABEL_PSI0 if peak == 0 else LABEL_PSIPI
        assert met["label"][k] == expected
        # the global peak is one of the two candidates, never in between
        assert env.max() == pytest.approx(max(e0, epi), rel=1e-6)


def test_envelope_extremes_and_depth_match_the_time_domain():
    E1, E2 = _random_fields(200, seed=2)
    met = am_phase_metrics(E1, E2)
    for k in range(len(E1)):
        env = _envelope(E1[k], E2[k])
        scale = env.max()
        assert met["env_max"][k] == pytest.approx(env.max(), rel=1e-6)
        # grid minimum is an upper bound on the true minimum, within grid error
        assert met["env_min"][k] <= env.min() + 1e-9
        assert met["env_min"][k] == pytest.approx(env.min(), abs=1e-4 * scale)
        assert met["depth"][k] == pytest.approx(env.max() - env.min(), abs=1e-4 * scale)


def test_depth_does_not_vanish_for_perpendicular_fields():
    a = 0.3
    E1 = np.array([[a, 0.0, 0.0]])
    E2 = np.array([[0.0, a, 0.0]])
    met = am_phase_metrics(E1, E2)
    env = _envelope(E1[0], E2[0])
    assert met["depth"][0] == pytest.approx((np.sqrt(2) - 1) * a, rel=1e-6)
    assert met["cos"][0] == pytest.approx(0.0, abs=1e-12)
    # two equal peaks per beat
    assert env[0] == pytest.approx(env[len(_PS) // 2], rel=1e-9)


@pytest.mark.parametrize("sign, label", [(1.0, LABEL_PSI0), (-1.0, LABEL_PSIPI)])
def test_parallel_fields_modulate_fully(sign, label):
    E1 = np.array([[0.2, 0.0, 0.0]])
    E2 = np.array([[sign * 0.5, 0.0, 0.0]])
    met = am_phase_metrics(E1, E2)
    assert met["label"][0] == label
    assert met["env_max"][0] == pytest.approx(0.7)
    assert met["env_min"][0] == pytest.approx(0.3, abs=1e-6)
    assert met["depth"][0] == pytest.approx(0.4, abs=1e-6)


def test_reversing_one_pair_swaps_the_label_and_keeps_the_depth():
    E1, E2 = _random_fields(500, seed=3)
    met = am_phase_metrics(E1, E2)
    flipped = am_phase_metrics(E1, -E2)
    nonzero = np.abs(np.einsum("ij,ij->i", E1, E2)) > 1e-12
    swapped = np.where(met["label"] == LABEL_PSI0, LABEL_PSIPI, LABEL_PSI0)
    np.testing.assert_array_equal(flipped["label"][nonzero], swapped[nonzero])
    np.testing.assert_allclose(flipped["depth"], met["depth"], rtol=1e-9, atol=1e-12)


def test_brute_force_check_agrees_with_the_metrics():
    E1, E2 = _random_fields(300, seed=4)
    met = am_phase_metrics(E1, E2)
    bf = brute_force_check(E1, E2, met, n=100, seed=0)
    assert bf["elements_checked"] == 100
    assert bf["peak_not_at_0_or_180_matching_label"] == 0
    assert bf["max_rel_depth_error_vs_grid"] < 1e-3
