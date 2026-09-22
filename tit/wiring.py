"""Carrier wiring -- which channels are coherent, which beat, and at what phase.

Every field quantity the toolbox reports depends on a fact the FEM cannot supply:
which electrode pairs are driven at the *same* frequency and which at different
ones.  Fields at one frequency superpose as vectors at every instant; fields at
different frequencies do not, and only they produce a beat envelope.  This is
the governing rule of Cassara et al. 2025 Part II, p. 8 -- *"coherent field
superposition was used for identical frequencies, and incoherent superposition
(i.e., SAR addition) was used when the frequencies differed"* -- and it is
quoted in full in :mod:`tit.fields`.

Before this module the rule was applied by *assumption*: every electrode pair
was taken to sit at its own distinct frequency and consecutive pairs were taken
to beat (the positional model, ``docs/dev/DECISIONS.md`` 2026-09-07).  That
assumption is silent and, when a montage drives two pairs from one frequency,
wrong in a way that hides rather than shows:

* the beat envelope does not exist at all, yet the sign-agnostic K=1 closed form
  (:func:`tit.calc._get_TI_vectors_k1`) still returns a non-zero ``TI_max``; and
* ``+/- +/-`` and ``+/- -/+`` differ physically -- one sums, the other cancels --
  yet every shipped metric returned the same number for both.

:func:`derive_wiring` turns explicit per-channel frequencies and phases into the
structure the physics needs, so coherence is *derived* rather than assumed.
Passing no frequencies leaves the positional model in force, unchanged.

The three consumers
-------------------
``envelope_inputs``
    Packs the groups into the ``(fields, psi, unpaired)`` triple
    :func:`tit.calc.get_TI_vectors` already accepts.  A group with no beat
    partner goes to ``unpaired``, where it raises the carrier power ``P`` and
    contributes nothing to the coherent term ``Q`` -- so a montage with no
    difference frequency yields a modulation depth of exactly zero, which is the
    physically correct answer.
``coherent_amplitudes``
    The per-frequency vector pre-sum that :mod:`tit.fields` applies before
    combining carriers incoherently.
``Wiring.describe``
    One line of prose for a job log or a GUI readout.  A zero envelope map must
    never look like a failure.

Conventions
-----------
Frequencies are Hz, phases are degrees, and both are per *channel* (one
electrode pair, one current source), positionally aligned with
``Montage.electrode_pairs`` exactly as ``SimulationConfig.intensities`` is.
Phase is the drive phase of the source: the quasi-static field phasor is the
real FEM field times that phase, with no propagation delay anywhere in the head.

See Also
--------
tit.calc : The envelope math this module feeds.
tit.fields : The carrier-exposure metrics this module feeds.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CoherentGroup",
    "GroupAmplitude",
    "Wiring",
    "WiringError",
    "coherent_amplitudes",
    "derive_wiring",
    "envelope_inputs",
]

logger = logging.getLogger(__name__)


#: Frequencies within this many Hz of each other are the *same* carrier and sum
#: coherently.  Essentially exact: it guards float round-off, not hardware
#: tolerance.  A deliberate offset a user actually typed is a beat, and is
#: handled as one -- see `MIN_BEAT_HZ`.
FREQ_IDENTICAL_TOL_HZ = 1e-6

#: Phases within this many degrees of 0 or 180 count as in phase / anti-phase,
#: so the coherent sum stays a real vector (see `CoherentGroup.signs`).
PHASE_TOL_DEG = 1e-6

#: A non-zero difference frequency below this is neither a beat nor coherence:
#: the envelope period exceeds any plausible stimulation epoch, so neither model
#: describes it.  Warned about, not rejected -- the user may know better.
MIN_BEAT_HZ = 0.1

#: Frequency separations up to this are treated as envelope-producing beats.
#: Above it, two carriers are taken to be mutually incoherent.  The (P, Q)
#: envelope represents exactly one difference frequency, so a *second* distinct
#: separation below this bound is rejected rather than approximated.
MAX_BEAT_HZ = 1000.0

#: Distinct carriers are assumed mutually incoherent, which needs their
#: separation to be large against the beat.  Below this multiple of the beat
#: frequency the assumption is strained and a warning is logged.
INCOHERENCE_MARGIN = 10.0


class WiringError(ValueError):
    """A frequency/phase assignment the field math cannot represent."""


@dataclass(frozen=True)
class CoherentGroup:
    """The channels driven at one frequency, which superpose as vectors.

    Attributes
    ----------
    frequency_hz : float
        The group's carrier frequency.
    phase_deg : float
        The phase shared by every member, in ``[0, 180)``.  Anti-phase members
        carry their 180 degrees in :attr:`signs` instead, so that the group's
        amplitude stays a real vector.
    members : tuple of int
        Channel indices, ascending.
    member_phases_deg : tuple of float
        Each member's own drive phase, as given.  Redundant with
        :attr:`phase_deg` and :attr:`signs` for a linearly polarised group, but
        it is the only description left when :attr:`signs` is ``None``.
    signs : tuple of float or None
        ``+1`` or ``-1`` per member, or ``None`` when the members' phases are
        not all 0 or 180 apart.  ``None`` means the group sums to an
        elliptically polarised carrier: its exposure is still exact, but the
        (P, Q) envelope formulation does not cover it.
    """

    frequency_hz: float
    phase_deg: float
    members: tuple[int, ...]
    member_phases_deg: tuple[float, ...]
    signs: tuple[float, ...] | None

    @property
    def is_real(self) -> bool:
        """Whether the coherent sum is a real vector (not elliptically polarised)."""
        return self.signs is not None


@dataclass(frozen=True)
class Wiring:
    """The derived carrier structure of a montage.

    Attributes
    ----------
    groups : tuple of CoherentGroup
        One per distinct frequency, ascending by frequency.
    beat_pairs : tuple of tuple of int
        Index pairs into :attr:`groups`, each ``(low, high)`` separated by
        :attr:`delta_f_hz`.  These are the carriers that beat.
    unpaired : tuple of int
        Indices into :attr:`groups` with no beat partner.  They raise the total
        carrier power without contributing to the envelope.
    delta_f_hz : float or None
        The one difference frequency, or ``None`` when nothing beats.
    psi_rad : tuple of float
        Envelope phase per entry of :attr:`beat_pairs`, in radians: the phase of
        the high carrier minus that of the low one.  Feeds ``tit.calc``'s
        ``psi``; envelopes offset in phase weaken the aggregate and cancel at
        180 degrees (Botzanowski et al. 2025, Supplementary Fig. 1).
    n_channels : int
        How many channels this wiring was derived from.
    """

    groups: tuple[CoherentGroup, ...]
    beat_pairs: tuple[tuple[int, int], ...]
    unpaired: tuple[int, ...]
    delta_f_hz: float | None
    psi_rad: tuple[float, ...]
    n_channels: int

    @property
    def has_modulation(self) -> bool:
        """Whether any beat exists, i.e. whether a modulation envelope is defined."""
        return bool(self.beat_pairs)

    @property
    def has_coherent_sum(self) -> bool:
        """Whether any frequency carries more than one channel.

        ``True`` is exactly the condition under which electrode ordering changes
        the answer, because two channels then add or cancel as vectors.
        """
        return any(len(g.members) > 1 for g in self.groups)

    @property
    def is_real(self) -> bool:
        """Whether every group sums to a real vector, so the envelope is defined."""
        return all(g.is_real for g in self.groups)

    @property
    def channel_beat_pairs(self) -> tuple[frozenset, ...]:
        """The beating carriers as sets of *channel* indices.

        ``beat_pairs`` indexes :attr:`groups`; this resolves those to the
        electrode pairs that actually beat, which is what callers reasoning
        about montage symmetry need.
        """
        return tuple(
            frozenset(self.groups[lo].members) | frozenset(self.groups[hi].members)
            for lo, hi in self.beat_pairs
        )

    def is_positional(self) -> bool:
        """Whether this wiring is exactly what the positional model assumes.

        True when every frequency carries one channel, consecutive channels
        beat, and every envelope is phase-aligned -- i.e. the wiring adds no
        information the positional reading did not already encode.  Callers
        that rely on positional symmetry can keep doing so only in that case.
        """
        if any(len(g.members) != 1 for g in self.groups):
            return False
        if any(abs(p) > 1e-12 for p in self.psi_rad):
            return False
        if self.unpaired:
            return False
        expected = {frozenset((i, i + 1)) for i in range(0, self.n_channels, 2)}
        return set(self.channel_beat_pairs) == expected

    def describe(self) -> str:
        """One line of prose for a job log or a GUI readout.

        Returns
        -------
        str
            For example ``"2 channels - 1 carrier pair - delta f = 10 Hz"`` or
            ``"2 channels - 1 frequency - no modulation (TI_max = 0)"``.
        """

        def n_of(count, noun, plural=None):
            return f"{count} {noun if count == 1 else (plural or noun + 's')}"

        parts = [
            n_of(self.n_channels, "channel"),
            n_of(len(self.groups), "frequency", "frequencies"),
        ]
        if self.has_modulation:
            parts.append(n_of(len(self.beat_pairs), "carrier pair"))
            parts.append(f"delta f = {self.delta_f_hz:g} Hz")
            if self.unpaired:
                parts.append(f"{len(self.unpaired)} non-beating")
        else:
            parts.append("no modulation (TI_max = 0)")
        if self.has_coherent_sum:
            parts.append("coherent sum (electrode order matters)")
        return " - ".join(parts)


def _validate_drive(frequencies_hz, phases_deg):
    """Coerce and check the raw per-channel drive arrays."""
    freqs = [float(f) for f in frequencies_hz]
    if not freqs:
        raise WiringError("At least one channel frequency is required.")
    for i, f in enumerate(freqs):
        if not math.isfinite(f) or f <= 0.0:
            raise WiringError(
                f"Channel {i + 1} frequency must be finite and positive, got {f!r}."
            )
    if phases_deg is None:
        phases = [0.0] * len(freqs)
    else:
        phases = [float(p) for p in phases_deg]
        if len(phases) != len(freqs):
            raise WiringError(
                f"Got {len(freqs)} frequencies but {len(phases)} phases; "
                "one of each is required per electrode pair."
            )
        for i, p in enumerate(phases):
            if not math.isfinite(p):
                raise WiringError(f"Channel {i + 1} phase must be finite, got {p!r}.")
    return freqs, phases


def _group_phase_and_signs(member_phases):
    """Resolve a group's common phase and the +/-1 each member folds into.

    The group's amplitude is written ``exp(i*theta) * sum_i s_i E_i`` so that the
    summed part stays a real vector.  That is possible exactly when every
    member's phase is 0 or 180 degrees from the others.

    Returns
    -------
    theta_deg : float
        The common phase, in ``[0, 180)``.
    signs : tuple of float or None
        ``None`` when the members are not all in phase or anti-phase, i.e. the
        group is elliptically polarised.
    """
    theta = member_phases[0] % 180.0
    signs = []
    for p in member_phases:
        delta = (p - theta) % 360.0
        if delta <= PHASE_TOL_DEG or delta >= 360.0 - PHASE_TOL_DEG:
            signs.append(1.0)
        elif abs(delta - 180.0) <= PHASE_TOL_DEG:
            signs.append(-1.0)
        else:
            return theta, None
    return theta, tuple(signs)


def _build_groups(freqs, phases):
    """Cluster channels by frequency, ascending, and resolve each group's phase.

    Clustering compares against the *group's first* frequency rather than the
    previous one, so a run of near-identical values cannot chain into an
    arbitrarily wide group.
    """
    order = sorted(range(len(freqs)), key=lambda i: (freqs[i], i))
    clusters: list[list[int]] = []
    for i in order:
        if clusters and abs(freqs[i] - freqs[clusters[-1][0]]) <= FREQ_IDENTICAL_TOL_HZ:
            clusters[-1].append(i)
        else:
            clusters.append([i])

    groups = []
    for members in clusters:
        theta, signs = _group_phase_and_signs([phases[i] for i in members])
        groups.append(
            CoherentGroup(
                frequency_hz=freqs[members[0]],
                phase_deg=theta,
                members=tuple(members),
                member_phases_deg=tuple(phases[i] for i in members),
                signs=signs,
            )
        )
    return tuple(groups)


def _match_beats(groups):
    """Pair the distinct frequencies into beating carriers.

    The beat frequency is the smallest positive separation; partners are the
    groups exactly that far apart.  Anything left over has no partner and is
    reported as non-beating.  A smallest separation wider than `MAX_BEAT_HZ` is
    not a beat at all -- nothing then beats and every group is non-beating.

    Raises
    ------
    WiringError
        If a group has two possible partners at the beat frequency (the pairing
        would be arbitrary and the montage really does produce two beats), or if
        a leftover group sits close enough to another to beat at a *different*
        difference frequency.  The (P, Q) envelope represents exactly one
        difference frequency, so both are rejected rather than approximated.
    """
    n = len(groups)
    if n < 2:
        return (), tuple(range(n)), None

    freqs = [g.frequency_hz for g in groups]
    seps = sorted(
        {
            round(freqs[j] - freqs[i], 9)
            for i in range(n)
            for j in range(i + 1, n)
            if freqs[j] - freqs[i] > FREQ_IDENTICAL_TOL_HZ
        }
    )
    if not seps or seps[0] > MAX_BEAT_HZ:
        # Every separation is wider than any envelope a neuron could follow, so
        # these are mutually incoherent carriers, not a beating pair.
        return (), tuple(range(n)), None
    delta_f = seps[0]

    def beats(i, j):
        return abs(abs(freqs[j] - freqs[i]) - delta_f) <= FREQ_IDENTICAL_TOL_HZ

    for i in range(n):
        partners = [j for j in range(n) if j != i and beats(i, j)]
        if len(partners) > 1:
            named = ", ".join(f"{freqs[j]:g} Hz" for j in [i] + partners)
            raise WiringError(
                f"Ambiguous carrier pairing: {named} are each {delta_f:g} Hz apart, "
                "so more than one pairing is possible and the montage produces "
                "more than one difference frequency. The envelope is defined for "
                "a single difference frequency; separate these carriers or drive "
                "them at frequencies with one unambiguous beat."
            )

    matched: set[int] = set()
    beat_pairs = []
    for i in range(n):
        if i in matched:
            continue
        for j in range(i + 1, n):
            if j not in matched and beats(i, j):
                beat_pairs.append((i, j))
                matched.update((i, j))
                break

    unpaired = tuple(i for i in range(n) if i not in matched)
    for i in unpaired:
        for j in range(n):
            if j == i:
                continue
            sep = abs(freqs[j] - freqs[i])
            if FREQ_IDENTICAL_TOL_HZ < sep <= MAX_BEAT_HZ and not beats(i, j):
                raise WiringError(
                    f"Carriers at {freqs[i]:g} Hz and {freqs[j]:g} Hz are "
                    f"{sep:g} Hz apart, which beats at a different difference "
                    f"frequency from the montage's {delta_f:g} Hz. The envelope "
                    "is defined for a single difference frequency; treating this "
                    "carrier as non-beating would discard a real envelope."
                )

    return tuple(beat_pairs), unpaired, delta_f


def _warn_about_marginal_separations(groups, beat_pairs, delta_f):
    """Log the cases where the coherent/incoherent dichotomy is a poor fit."""
    if delta_f is not None and delta_f < MIN_BEAT_HZ:
        logger.warning(
            "Difference frequency is %g Hz, below %g Hz: the envelope period "
            "(%.3g s) exceeds any plausible stimulation epoch, so these carriers "
            "are neither usefully coherent nor usefully beating. The envelope is "
            "computed as asked, but treat it with care.",
            delta_f,
            MIN_BEAT_HZ,
            1.0 / delta_f,
        )
    if delta_f is None or len(beat_pairs) < 2:
        return
    freqs = [g.frequency_hz for g in groups]
    for a, (i, _) in enumerate(beat_pairs):
        for j, _ in beat_pairs[a + 1 :]:
            sep = abs(freqs[j] - freqs[i])
            if sep < INCOHERENCE_MARGIN * delta_f:
                logger.warning(
                    "Carriers at %g Hz and %g Hz are only %g Hz apart, under %gx "
                    "the %g Hz beat. Distinct carriers are assumed mutually "
                    "incoherent, which that separation barely supports; the "
                    "reported envelope and exposure may be optimistic.",
                    freqs[i],
                    freqs[j],
                    sep,
                    INCOHERENCE_MARGIN,
                    delta_f,
                )


def derive_wiring(
    frequencies_hz: Sequence[float],
    phases_deg: Sequence[float] | None = None,
) -> Wiring:
    """Derive the carrier structure from per-channel frequencies and phases.

    Parameters
    ----------
    frequencies_hz : sequence of float
        One carrier frequency per electrode pair, positionally aligned with
        ``Montage.electrode_pairs``.  Must be finite and positive.
    phases_deg : sequence of float, optional
        One drive phase per electrode pair, in degrees.  Defaults to all zero.
        A 180-degree phase is exactly a polarity reversal, so ``+/- -/+`` is
        expressible either by swapping the pair's electrodes or by setting this.

    Returns
    -------
    Wiring
        The coherent groups, which of them beat, and at what envelope phase.

    Raises
    ------
    WiringError
        If the drive arrays are malformed, if the carrier pairing is ambiguous,
        or if the montage produces more than one difference frequency.

    Notes
    -----
    Channels at the same frequency form one group and superpose as vectors --
    this is the only thing that makes ``+/- +/-`` differ from ``+/- -/+``.
    Groups separated by the montage's one difference frequency beat; any group
    without such a partner contributes carrier power but no envelope, so a
    montage driven at a single frequency has a modulation depth of exactly zero.

    Examples
    --------
    >>> derive_wiring([2000.0, 2010.0]).has_modulation
    True
    >>> derive_wiring([2000.0, 2000.0]).has_modulation
    False
    """
    freqs, phases = _validate_drive(frequencies_hz, phases_deg)
    groups = _build_groups(freqs, phases)
    beat_pairs, unpaired, delta_f = _match_beats(groups)
    _warn_about_marginal_separations(groups, beat_pairs, delta_f)
    psi = tuple(
        math.radians(groups[hi].phase_deg - groups[lo].phase_deg)
        for lo, hi in beat_pairs
    )
    return Wiring(
        groups=groups,
        beat_pairs=beat_pairs,
        unpaired=unpaired,
        delta_f_hz=delta_f,
        psi_rad=psi,
        n_channels=len(freqs),
    )


@dataclass(frozen=True)
class GroupAmplitude:
    """One frequency group's coherently summed field.

    Attributes
    ----------
    group : CoherentGroup
        The group this amplitude belongs to.
    real : np.ndarray
        ``(N, 3)`` in-phase amplitude.  For a linearly polarised group this is
        the whole field, with the group's common phase factored out -- i.e. the
        vector sum ``sum_i s_i E_i``, which is what the envelope math consumes.
    imag : np.ndarray or None
        ``(N, 3)`` quadrature amplitude, or ``None`` when the group is linearly
        polarised.  When present the carrier traces an ellipse,
        ``E(t) = real * cos(wt) - imag * sin(wt)``.
    """

    group: CoherentGroup
    real: np.ndarray
    imag: np.ndarray | None

    @property
    def mean_square(self) -> np.ndarray:
        """Time-averaged ``|E(t)|^2`` times two, i.e. the squared amplitude.

        The sinusoid's factor of one half is applied once downstream, in the SAR
        calibration, and nowhere in the field-domain quantity -- the convention
        :mod:`tit.fields` already uses.
        """
        out = np.sum(self.real * self.real, axis=-1)
        if self.imag is not None:
            out = out + np.sum(self.imag * self.imag, axis=-1)
        return out


def coherent_amplitudes(
    fields: Sequence[ArrayLike], wiring: Wiring
) -> list[GroupAmplitude]:
    """Sum each frequency group's channel fields as vectors.

    This is the coherent pre-sum the governing rule calls for: fields at one
    frequency add as vectors at every instant, so they must be combined *before*
    distinct carriers are combined incoherently.

    Parameters
    ----------
    fields : sequence of np.ndarray
        One ``(N, 3)`` field per channel, positionally aligned with the
        frequencies the wiring was derived from.
    wiring : Wiring
        From :func:`derive_wiring`.

    Returns
    -------
    list of GroupAmplitude
        One per group, in the wiring's group order.

    Raises
    ------
    WiringError
        If the field count does not match the wiring, or shapes disagree.
    """
    arrs = [np.asarray(f, dtype=np.float64) for f in fields]
    if len(arrs) != wiring.n_channels:
        raise WiringError(
            f"Wiring describes {wiring.n_channels} channels but {len(arrs)} "
            "fields were given."
        )
    ref = arrs[0].shape
    if not ref or ref[-1] != 3:
        raise WiringError(f"Fields must have shape (..., 3), got {ref}.")
    for i, arr in enumerate(arrs[1:], start=2):
        if arr.shape != ref:
            raise WiringError(
                f"All fields must have identical shape; field 1 has {ref}, "
                f"field {i} has {arr.shape}."
            )

    out = []
    for group in wiring.groups:
        if group.is_real:
            real = np.zeros(ref, dtype=np.float64)
            for sign, idx in zip(group.signs, group.members):
                if sign > 0:
                    real += arrs[idx]
                else:
                    real -= arrs[idx]
            out.append(GroupAmplitude(group=group, real=real, imag=None))
        else:
            # No common phase to factor out -- which is precisely why the
            # group is elliptically polarised -- so each member keeps its own:
            # A = sum_i E_i exp(i phi_i).
            real = np.zeros(ref, dtype=np.float64)
            imag = np.zeros(ref, dtype=np.float64)
            for idx, phase in zip(group.members, group.member_phases_deg):
                phi = math.radians(phase)
                real += math.cos(phi) * arrs[idx]
                imag += math.sin(phi) * arrs[idx]
            out.append(GroupAmplitude(group=group, real=real, imag=imag))
    return out


def envelope_inputs(
    fields: Sequence[ArrayLike], wiring: Wiring
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Pack channel fields into the ``(fields, psi)`` pair :mod:`tit.calc` consumes.

    Parameters
    ----------
    fields : sequence of array-like
        One ``(N, 3)`` field per channel, aligned with the wiring.
    wiring : Wiring
        From :func:`derive_wiring`.

    Returns
    -------
    fields : list of np.ndarray
        Two fields per carrier, consecutive, exactly as
        :func:`tit.calc.get_TI_vectors` requires.
    psi : np.ndarray, shape (K,)
        One envelope phase in radians per carrier, aligned with ``fields``.

    Raises
    ------
    WiringError
        If any group is elliptically polarised.  The (P, Q) envelope is derived
        for carriers with a real amplitude vector; an elliptical carrier is a
        different quantity, and returning a number from a formula whose premise
        does not hold would be worse than refusing.

    Notes
    -----
    A carrier with **no beat partner** is emitted as ``[A_g, 0]`` -- paired with
    a zero field.  That is not a trick: a lone carrier contributes its power to
    ``P`` and nothing to the coherent term ``Q``, and pairing it with zero
    reproduces exactly that, since ``M_Q`` gains ``sym(A_g . 0^T) = 0``.  It is
    also what keeps :mod:`tit.calc` unmodified -- the zero-partnered pair is an
    ordinary two-field carrier as far as it is concerned.

    So a montage driven at a single frequency yields ``[A, 0]``: ``Q`` is
    identically zero and the modulation depth is exactly zero, which is the
    correct answer and the reason the carrier metrics are the only meaningful
    output in that case.
    """
    elliptical = [g.frequency_hz for g in wiring.groups if not g.is_real]
    if elliptical:
        named = ", ".join(f"{f:g} Hz" for f in elliptical)
        raise WiringError(
            f"The carriers at {named} sum to an elliptically polarised field, "
            "because their channels' phases are neither equal nor 180 degrees "
            "apart. The modulation envelope is defined for carriers with a real "
            "amplitude vector, so it is not computed here. Exposure metrics "
            "(hf_peak, hf_sar) remain exact for this montage."
        )

    amps = coherent_amplitudes(fields, wiring)
    packed: list[NDArray[np.float64]] = []
    psi: list[float] = []
    for lo, hi in wiring.beat_pairs:
        packed.append(amps[lo].real)
        packed.append(amps[hi].real)
    psi.extend(wiring.psi_rad)
    zero = np.zeros_like(amps[0].real) if amps else None
    for idx in wiring.unpaired:
        packed.append(amps[idx].real)
        packed.append(zero)
        # The zero-partnered product vanishes whatever phase is declared.
        psi.append(0.0)
    return packed, np.asarray(psi, dtype=np.float64)
