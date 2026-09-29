"""Carrier-derived high-frequency field metrics — single source of truth.

Safety metrics computed from per-pair carrier E-field vector arrays (shape
``(..., 3)`` each), following Cassarà et al. 2025, *Recommendations for the
Safe Application of Temporal Interference Stimulation in the Human Brain*
Part I (Bioelectromagnetics 46(2), e22542) and Part II (46(1), e22536).

The governing rule is Part II, p. 8: *"In the presence of multiple currents
(e.g., TIS channels), coherent field superposition was used for identical
frequencies, and incoherent superposition (i.e., SAR addition) was used when
the frequencies differed."*  Fields at one frequency are summed **as vectors
first**; only then do distinct carriers combine — in power for time-averaged
quantities, and by worst-case relative phase for the peak.

Which fields share a frequency is a property of the montage's drive, and both
functions accept it as an optional ``wiring`` (:mod:`tit.wiring`, built from the
per-channel frequencies).  Given one, the fields at each frequency
are summed **as vectors first**, and only the resulting per-frequency
amplitudes combine incoherently — the coherent pre-sum the governing rule calls
for.

Without a ``wiring`` the historical assumption stands: every field is taken to
be its own carrier at its own distinct frequency (the positional model, see
``docs/dev/DECISIONS.md``), and the fields combine incoherently exactly as they
always have.  That path is untouched, so no existing result moves.

The distinction is not academic.  Two pairs driven at one frequency add or
cancel according to their polarity, so ``+/- +/-`` and ``+/- -/+`` are
physically different montages; without a wiring both return the same number.
Polarity comes from electrode order alone: a reversed pair's field is negated
before the pre-sum, which is exactly a 180-degree phase reversal.

``hf_peak(*fields, wiring=None)`` — peak carrier field.  Distinct carriers
run at mutually incommensurate frequencies, so every relative phase
combination occurs over time; the worst-case instantaneous magnitude is the max
over sign choices, ``max_s |sum_c s_c * E_c|`` over *carriers* ``c``.  At two
carriers this is exactly ``max(|E1+E2|, |E1-E2|)`` (Cassarà Part I, Eq. 3,
p. 11: the worst case is "in-phase, spatially aligned fields").  With a wiring
the sign choice runs over the coherently summed frequencies rather than the raw
fields: within one frequency the relative polarity is *known*, not worst-cased.

``hf_sar(*fields, wiring=None)`` — heating driver, proportional to SAR.
Distinct carriers are incoherent, so power adds rather than amplitude:
``sum_c |E_c|^2`` over carriers — one field per carrier without a wiring, one
coherently summed frequency per carrier with one
(Cassarà Part I, p. 11: "the SAR distributions from the two channels, rather
than the E-fields themselves, must be summed"; Part II, p. 16: total power
deposition "is equal to the summed combination from all channels (incoherent
field superposition)").

``hf_peak_phaselocked(*fields, frequencies_hz)`` — peak carrier field for a
*phase-locked* drive.  ``hf_peak`` assumes every relative carrier phase
eventually occurs; a drive whose channels share one clock, run at exactly the
stated frequencies and all start at phase 0 visits only the phases its
frequencies allow.  This returns the true maximum over time of
``|sum_i E_i cos(2 pi f_i t)|``, polarity again coming from electrode order.
It never exceeds ``hf_peak``; it equals ``hf_peak`` for two commensurate
carriers and ``|sum E|`` for one shared frequency, and can be lower from three
carriers up (e.g. a four-pair montage with one pair reversed).  The modulation
envelope in :mod:`tit.calc` makes the same fixed-phase assumption, so the two
are consistent; ``hf_peak`` stays the assumption-free upper bound.

All three are distinct from the stimulation-relevant modulation envelope
(``TI_max`` / ``TI_normal``), computed in :mod:`tit.calc`.

See Also
--------
tit.sim.TI : Writes hf_peak / hf_sar as volume fields on the TI mesh.
tit.sim.mTI : Writes hf_peak / hf_sar as volume fields on the mTI mesh.
tit.source.fsaverage : Projects hf_peak / hf_sar onto fsaverage.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from functools import reduce

import numpy as np

#: Fields at or below this count use exact sign enumeration (2**(N-1)
#: combinations) in `hf_peak`; above it, a sign-refined direction sweep is
#: used instead (see `_hf_peak_sweep`). Measured cost of exact enumeration at
#: 200k elements: N=8 (128 combos) ~2.0s, N=12 (2048 combos) ~44.6s -- the
#: combinatorial blowup past N=8 is what the sweep fallback avoids.
EXACT_SIGN_ENUM_MAX_FIELDS = 8

# Spatial elements processed per chunk, bounding peak memory independent of
# input size for both the sign-enumeration and direction-sweep paths.
_CHUNK_SIZE = 20_000

# Direction count for the N > EXACT_SIGN_ENUM_MAX_FIELDS sweep fallback.
# The sweep is still a lower bound on the true max (only sampled directions'
# implied sign patterns are tried), so this is chosen generously to keep the
# sampling gap small.
_SWEEP_N_DIRECTIONS = 4000

# Directions processed per inner batch during the sweep, bounding memory
# alongside _CHUNK_SIZE (peak use ~ _CHUNK_SIZE * _SWEEP_DIR_BATCH, independent
# of N since the sweep accumulates one field at a time).
_SWEEP_DIR_BATCH = 200

#: Longest common period `hf_peak_phaselocked` will search.  A phase-locked
#: drive repeats every 1/gcd(f_i) seconds; frequencies with no common period
#: this short (e.g. 2000 and 2000.37 Hz) never settle into a pattern worth
#: calling phase-locked, and the search cost grows with the period.
PHASELOCKED_MAX_PERIOD_S = 2.0

# Frequencies are rationalised on this grid (Hz) to find their common period.
_PHASELOCKED_FREQ_RESOLUTION_HZ = 1e-3

# Time samples per cycle of the fastest carrier.  The sampled maximum is then
# refined (parabola + Newton step), so this sets how close a candidate must be
# to the true peak to be found, not the final accuracy.
_PHASELOCKED_SAMPLES_PER_CYCLE = 32

# Local sample maxima per element carried into refinement.
_PHASELOCKED_CANDIDATES = 8

# Elements per chunk; memory ~ chunk * n_samples * 3 floats.
_PHASELOCKED_CHUNK = 256


def _stack_fields(fields: tuple) -> tuple[np.ndarray, tuple]:
    """Validate and stack N carrier fields into a flat ``(N, M, 3)`` array.

    Returns the stack plus the common original shape (for reshaping the
    result back). Raises ``ValueError`` if fewer than 2 fields, shapes
    differ, or the last axis isn't 3.
    """
    if len(fields) < 2:
        raise ValueError(f"hf_peak/hf_sar require at least 2 fields, got {len(fields)}")
    arrays = [np.asarray(f, dtype=float) for f in fields]
    shape = arrays[0].shape
    if shape[-1] != 3:
        raise ValueError(f"fields must have shape (..., 3), got last axis {shape[-1]}")
    for i, a in enumerate(arrays[1:], start=1):
        if a.shape != shape:
            raise ValueError(
                f"all fields must share the same shape: field 0 has {shape}, "
                f"field {i} has {a.shape}"
            )
    stack = np.stack(arrays, axis=0).reshape(len(arrays), -1, 3)
    return stack, shape


def _sign_matrix(n: int) -> np.ndarray:
    """The ``2**(n-1)`` sign vectors needed for the max-over-signs peak.

    The first sign is fixed to +1: flipping every sign leaves ``|sum|``
    unchanged, so enumerating the rest covers all distinct sums.
    """
    tail = np.array(list(itertools.product((1.0, -1.0), repeat=n - 1)))
    head = np.ones((tail.shape[0], 1))
    return np.hstack([head, tail])


def _hf_peak_exact(stack: np.ndarray) -> np.ndarray:
    """Exact ``max_s |sum_i s_i * E_i|`` via sign enumeration, chunked over rows."""
    n, m, _ = stack.shape
    signs = _sign_matrix(n)
    out = np.empty(m, dtype=float)
    for lo in range(0, m, _CHUNK_SIZE):
        hi = min(lo + _CHUNK_SIZE, m)
        block = stack[:, lo:hi, :]
        best = np.zeros(hi - lo, dtype=float)
        for row in signs:
            total = block[0] * row[0]
            for i in range(1, n):
                total = total + block[i] * row[i]
            np.maximum(best, np.linalg.norm(total, axis=-1), out=best)
        out[lo:hi] = best
    return out


def _fibonacci_directions(n_points: int) -> np.ndarray:
    """``n_points`` roughly-uniform unit directions on the sphere (golden-angle spiral)."""
    i = np.arange(n_points) + 0.5
    phi = np.arccos(1 - 2 * i / n_points)
    theta = np.pi * (3 - np.sqrt(5)) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=-1,
    )


def _hf_peak_sweep(stack: np.ndarray) -> np.ndarray:
    """Sign-refined direction sweep for N > `EXACT_SIGN_ENUM_MAX_FIELDS`.

    Samples directions and finds each row's best support direction ``n*``
    (``max_n sum_i |E_i . n|``), then evaluates the *exact*, realizable
    vector sum for the sign pattern ``n*`` implies: ``s_i = sign(E_i . n*)``,
    result ``|sum_i s_i E_i|``. By Cauchy-Schwarz this is >= the raw support
    value (a mere projection), so it is both tighter and an actually
    achievable field state. Still a lower bound on the true max over all
    ``2**(N-1)`` sign combinations, since only the sampled directions'
    implied patterns are tried -- see `hf_peak`.
    """
    n, m, _ = stack.shape
    directions = _fibonacci_directions(_SWEEP_N_DIRECTIONS)
    out = np.empty(m, dtype=float)
    for lo in range(0, m, _CHUNK_SIZE):
        hi = min(lo + _CHUNK_SIZE, m)
        c = hi - lo
        best_support = np.full(c, -np.inf, dtype=float)
        best_dir = np.zeros(c, dtype=np.int64)
        for d0 in range(0, _SWEEP_N_DIRECTIONS, _SWEEP_DIR_BATCH):
            d1 = min(d0 + _SWEEP_DIR_BATCH, _SWEEP_N_DIRECTIONS)
            dirs_batch = directions[d0:d1]
            support = np.zeros((c, d1 - d0), dtype=float)
            for i in range(n):
                support += np.abs(stack[i, lo:hi, :] @ dirs_batch.T)
            batch_argmax = support.argmax(axis=1)
            batch_best = support[np.arange(c), batch_argmax]
            improved = batch_best > best_support
            best_support = np.where(improved, batch_best, best_support)
            best_dir = np.where(improved, d0 + batch_argmax, best_dir)

        n_star = directions[best_dir]  # (c, 3), each row's best direction
        total = np.zeros((c, 3), dtype=float)
        for i in range(n):
            e_i = stack[i, lo:hi, :]
            dot = np.einsum("cx,cx->c", e_i, n_star)
            sign = np.where(dot < 0, -1.0, 1.0)  # tie-break dot==0 to +1
            total += sign[:, None] * e_i
        out[lo:hi] = np.linalg.norm(total, axis=-1)
    return out


def _grouped_amplitudes(fields, wiring):
    """Coherently pre-sum the channel fields into one amplitude per frequency.

    This is the step the governing rule puts before everything else: fields at
    one frequency are vectors at the same instant, so they add as vectors.

    Returns
    -------
    stack : np.ndarray, shape ``(G, M, 3)``
        Summed carrier amplitude per frequency group.
    shape : tuple
        The original field shape, for reshaping the result back.
    """
    from tit.wiring import coherent_amplitudes

    amps = coherent_amplitudes(fields, wiring)
    shape = np.asarray(fields[0], dtype=float).shape
    stack = np.stack([a.field.reshape(-1, 3) for a in amps], axis=0)
    return stack, shape


def hf_peak_is_exact(n_fields: int | None = None, wiring=None) -> bool:
    """Is :func:`hf_peak` exact for this many carriers, or a lower bound?

    ``True`` up to `EXACT_SIGN_ENUM_MAX_FIELDS` carriers, where every one of
    the ``2**(C-1)`` sign patterns is enumerated.  Above that the direction
    sweep tries only the sign patterns implied by sampled directions, so the
    result is a **lower bound** on the true worst-case peak and is therefore
    slightly non-conservative as a safety metric.  Callers that record or
    display ``hf_peak`` should carry this flag alongside the value.

    Without a wiring one field is one carrier, so *n_fields* is the carrier
    count.  With a ``wiring`` the carriers are its frequency *groups*.

    Parameters
    ----------
    n_fields : int, optional
        Carrier count, when no wiring is available.
    wiring : tit.wiring.Wiring, optional
        Takes precedence over *n_fields*.
    """
    if wiring is not None:
        return len(wiring.groups) <= EXACT_SIGN_ENUM_MAX_FIELDS
    if n_fields is None:
        raise ValueError("hf_peak_is_exact needs either n_fields or a wiring.")
    return int(n_fields) <= EXACT_SIGN_ENUM_MAX_FIELDS


def hf_peak(*fields, wiring=None) -> np.ndarray:
    """Peak carrier field: max over sign choices of the carrier vector sum.

    Each field is one carrier (positional wiring).  Carriers combine at their
    worst-case relative phase, which for incommensurate frequencies is
    realised over time:
    ``max_s |sum_c s_c E_c|``.  At two carriers this is Cassarà et al. 2025
    Part I, Eq. 3 (p. 11), ``max(|E1+E2|, |E1-E2|)``.

    Exact sign enumeration (``2**(C-1)`` combinations over ``C`` carriers) is
    used up to `EXACT_SIGN_ENUM_MAX_FIELDS`. Above that, a Fibonacci-sphere
    direction sweep picks the best-sampled direction and evaluates the exact,
    realizable vector sum for the sign pattern it implies -- tighter than a
    raw support-function value, but still a lower bound (hence slightly
    non-conservative) since only sampled directions' sign patterns are tried.
    Query :func:`hf_peak_is_exact` for which path a montage takes.

    Parameters
    ----------
    *fields : array-like, shape ``(..., 3)``
        Two or more channel E-field vectors (one per electrode pair), all
        the same shape.
    wiring : tit.wiring.Wiring, optional
        Which channels share a frequency.  Given one, the channels at each
        frequency are summed as vectors first and the sign choice runs over the
        resulting per-frequency amplitudes.  Omitted, every field is taken to
        be its own carrier -- the shipped behaviour, unchanged.

    Returns
    -------
    numpy.ndarray, shape ``(...,)``
        The worst-case peak carrier field magnitude.
    """
    if wiring is None:
        stack, shape = _stack_fields(fields)
        n = stack.shape[0]
        if n <= EXACT_SIGN_ENUM_MAX_FIELDS:
            flat = _hf_peak_exact(stack)
        else:
            flat = _hf_peak_sweep(stack)
        return flat.reshape(shape[:-1])

    _stack_fields(fields)  # same arity and shape contract as the plain path
    stack, shape = _grouped_amplitudes(fields, wiring)
    if stack.shape[0] <= EXACT_SIGN_ENUM_MAX_FIELDS:
        flat = _hf_peak_exact(stack)
    else:
        flat = _hf_peak_sweep(stack)
    return flat.reshape(shape[:-1])


def phase_locked_period(frequencies_hz) -> float:
    """Common period (s) of a phase-locked drive at these carrier frequencies.

    ``1 / gcd(f_i)``, with the frequencies rationalised on a 1 mHz grid.

    Parameters
    ----------
    frequencies_hz : sequence of float
        One carrier frequency per electrode pair (Hz).

    Returns
    -------
    float
        The period after which every carrier returns to its starting phase.

    Raises
    ------
    ValueError
        If a frequency is not finite and positive, is not a multiple of
        1 mHz, or the common period exceeds `PHASELOCKED_MAX_PERIOD_S`.
    """
    freqs = [float(f) for f in frequencies_hz]
    if not freqs:
        raise ValueError("hf_peak_phaselocked needs at least one carrier frequency.")
    ticks = []
    for f in freqs:
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError(
                f"Carrier frequency must be finite and positive, got {f!r}."
            )
        q = (
            Fraction(f).limit_denominator(10**6)
            / Fraction(_PHASELOCKED_FREQ_RESOLUTION_HZ).limit_denominator()
        )
        if abs(float(q) - round(float(q))) > 1e-6:
            raise ValueError(
                f"hf_peak_phaselocked: carrier frequency {f:g} Hz is not a multiple of "
                f"{_PHASELOCKED_FREQ_RESOLUTION_HZ:g} Hz, so the drive has no common period."
            )
        ticks.append(int(round(float(q))))
    gcd_hz = reduce(math.gcd, ticks) * _PHASELOCKED_FREQ_RESOLUTION_HZ
    period = 1.0 / gcd_hz
    if period > PHASELOCKED_MAX_PERIOD_S:
        named = ", ".join(f"{f:g}" for f in freqs)
        raise ValueError(
            f"hf_peak_phaselocked: frequencies {named} Hz repeat only every {period:g} s "
            f"(longer than {PHASELOCKED_MAX_PERIOD_S:g} s), so a phase-locked peak is not "
            "defined for them; choose frequencies whose greatest common divisor is at "
            f"least {1.0 / PHASELOCKED_MAX_PERIOD_S:g} Hz, or use hf_peak."
        )
    return period


def hf_peak_phaselocked(*fields, frequencies_hz) -> np.ndarray:
    """Peak carrier field for a phase-locked drive: ``max_t |sum_i E_i cos(2 pi f_i t)|``.

    Assumes the channels share one clock, run at exactly ``frequencies_hz``
    and all start at phase 0; polarity comes from electrode order (the sign of
    each ``E_i``).  Tissue is quasi-static, so the relative phase at every
    point equals the relative phase at the sources.  Channels at one frequency
    therefore add as vectors at every instant, and distinct frequencies visit
    only the relative phases their common period allows -- unlike
    :func:`hf_peak`, which assumes every relative phase occurs.

    The waveform is even and periodic in the common period ``T``
    (:func:`phase_locked_period`), so the search runs over ``[0, T/2]``,
    sampled at `_PHASELOCKED_SAMPLES_PER_CYCLE` points per cycle of the
    fastest carrier.  The largest local sample maxima of each element are then
    refined by a parabola vertex and a Newton step on ``|E(t)|^2``.  Every value
    returned is ``|E(t)|`` at an actual time, so the result never exceeds the
    true maximum (nor :func:`hf_peak`); the refinement makes it match the
    maximum to well under 1e-6 relative in the tests.

    Parameters
    ----------
    *fields : array-like, shape ``(..., 3)``
        Two or more channel E-field vectors (one per electrode pair), all the
        same shape, in pair order.
    frequencies_hz : sequence of float
        One carrier frequency per field, in the same order.

    Returns
    -------
    numpy.ndarray, shape ``(...,)``
        The phase-locked peak carrier field magnitude.

    Raises
    ------
    ValueError
        If the field and frequency counts differ, or the frequencies have no
        usable common period (see :func:`phase_locked_period`).
    """
    stack, shape = _stack_fields(fields)
    n, m, _ = stack.shape
    freqs = np.asarray([float(f) for f in frequencies_hz], dtype=float)
    if freqs.size != n:
        raise ValueError(
            f"hf_peak_phaselocked: {n} fields but {freqs.size} frequencies; one per pair is required."
        )
    period = phase_locked_period(freqs)
    omega = 2.0 * np.pi * freqs
    half = 0.5 * period
    n_samp = int(np.ceil(half * freqs.max() * _PHASELOCKED_SAMPLES_PER_CYCLE)) + 1
    t = np.linspace(0.0, half, n_samp)
    dt = t[1] - t[0]
    cos_t = np.cos(omega[:, None] * t[None, :])  # (n, T)
    # |E(t)|^2 = sum_ij (E_i . E_j) cos_i(t) cos_j(t): one small Gram matrix per
    # element times a fixed (pairs x samples) table -- a single BLAS product.
    iu, ju = np.triu_indices(n)
    weight = np.where(iu == ju, 1.0, 2.0)
    prod = cos_t[iu] * cos_t[ju]  # (P, T)
    # Windows of about half a cycle of the fastest carrier hold roughly one local
    # maximum each; the best sample of the best windows are the candidates.
    w = max(1, _PHASELOCKED_SAMPLES_PER_CYCLE // 2)
    n_win = -(-n_samp // w)
    pad = n_win * w - n_samp
    k = min(_PHASELOCKED_CANDIDATES, n_win)

    def eval_at(block, tc):
        """|E|^2 and its first two time derivatives at times tc (c, k)."""
        ph = omega[:, None, None] * tc[None, :, :]  # (n, c, k)
        c, s = np.cos(ph), np.sin(ph)
        e = np.einsum("icj,icd->cjd", c, block)  # (c, k, 3)
        de = np.einsum("icj,icd->cjd", -s * omega[:, None, None], block)
        d2e = np.einsum("icj,icd->cjd", -c * (omega**2)[:, None, None], block)
        f0 = np.sum(e * e, axis=-1)
        f1 = 2.0 * np.sum(e * de, axis=-1)
        f2 = 2.0 * (np.sum(de * de, axis=-1) + np.sum(e * d2e, axis=-1))
        return f0, f1, f2

    out = np.empty(m, dtype=float)
    for lo in range(0, m, _PHASELOCKED_CHUNK):
        hi = min(lo + _PHASELOCKED_CHUNK, m)
        block = stack[:, lo:hi, :]  # (n, c, 3)
        gram = np.einsum("icd,jcd->cij", block, block)[:, iu, ju] * weight  # (c, P)
        mag2 = gram @ prod  # (c, T)
        if pad:
            mag2 = np.pad(mag2, ((0, 0), (0, pad)), constant_values=-np.inf)
        win = mag2.reshape(hi - lo, n_win, w)
        in_win = win.argmax(axis=2)  # (c, n_win)
        win_max = np.take_along_axis(win, in_win[..., None], axis=2)[..., 0]
        top = np.argpartition(win_max, -k, axis=1)[:, -k:]  # (c, k)
        sidx = top * w + np.take_along_axis(in_win, top, axis=1)
        best = np.take_along_axis(mag2, sidx, axis=1).max(axis=1)
        t0 = t[sidx]
        tc = t0
        # Two Newton steps on d|E|^2/dt = 0 from each candidate sample, kept
        # within one sample of it; every evaluation is |E|^2 at a real time.
        for _ in range(2):
            f0, f1, f2 = eval_at(block, tc)
            best = np.maximum(best, f0.max(axis=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.where(f2 < 0, -f1 / f2, 0.0)
            tc = np.clip(tc + step, t0 - dt, t0 + dt)
        f0, _, _ = eval_at(block, tc)
        best = np.maximum(best, f0.max(axis=1))
        out[lo:hi] = np.sqrt(np.maximum(best, 0.0))
    return out.reshape(shape[:-1])


def hf_sar(*fields, wiring=None) -> np.ndarray:
    """Incoherent carrier heating driver, proportional to SAR: ``sum_c |E_c|^2``.

    Each field is one carrier (positional wiring), and the carriers sit at
    different, incommensurate frequencies, so their SAR/power adds rather
    than their amplitudes — Cassarà et al. 2025 Part II, p. 8: *"coherent
    field superposition was used for identical frequencies, and incoherent
    superposition (i.e., SAR addition) was used when the frequencies
    differed."*

    This is a field-domain proxy in ``(V/m)^2``, **not** calibrated SAR.
    With ``E_c`` the sinusoidal *amplitude* (peak, as the paper's thresholds
    are stated), the time-averaged SAR of Part II Eq. 1 is
    ``(sigma / 2 rho) * hf_sar`` — the ``1/2`` being the sinusoid's
    time-average, applied once here and nowhere else in the toolbox.
    Equivalently the RMS-squared carrier field is ``hf_sar / 2``.

    Parameters
    ----------
    *fields : array-like, shape ``(..., 3)``
        Two or more channel E-field vectors (one per electrode pair), all
        the same shape.
    wiring : tit.wiring.Wiring, optional
        Which channels share a frequency.  Given one, the channels at each
        frequency are summed as vectors before their powers add, so two pairs
        on one frequency contribute ``|E_a + E_b|^2`` rather than
        ``|E_a|^2 + |E_b|^2``.  Omitted, every field is taken to be its own
        carrier -- the shipped behaviour, unchanged.

    Returns
    -------
    numpy.ndarray, shape ``(...,)``
        ``sum_c |E_c|^2`` in ``(V/m)^2`` — proportional to tissue heating.
    """
    if wiring is None:
        stack, shape = _stack_fields(fields)
        flat = np.sum(np.linalg.norm(stack, axis=-1) ** 2, axis=0)
        return flat.reshape(shape[:-1])

    _stack_fields(fields)  # same arity and shape contract as the plain path
    stack, shape = _grouped_amplitudes(fields, wiring)
    flat = np.sum(np.sum(stack * stack, axis=-1), axis=0)
    return flat.reshape(shape[:-1])
