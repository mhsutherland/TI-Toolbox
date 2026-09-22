/**
 * Carrier wiring, for the run pages' readouts.
 *
 * Shared by the Simulator's job rows and the Optimizer's mEx Carriers section,
 * so both describe the same frequencies in the same words.
 *
 * `tit/wiring.py` is the authority: the server re-derives the wiring from the
 * same frequencies and rejects anything it cannot represent, so nothing here
 * decides what runs. What this does decide is what the user is told *before*
 * they submit — and the case that matters is a montage with no difference
 * frequency, whose `TI_max` map is legitimately, physically zero. Without a
 * readout saying so, a blank result looks like a bug.
 *
 * Kept deliberately small, and in step with `derive_wiring`: group channels by
 * frequency, take the smallest positive separation as the beat, pair groups
 * that far apart. Where the Python would raise, this returns `"unresolved"`
 * rather than guessing — the server's error is the one the user should see.
 */

/** Frequencies within this many Hz are the same carrier (mirrors `FREQ_IDENTICAL_TOL_HZ`). */
const FREQ_IDENTICAL_TOL_HZ = 1e-6;

/** Separations wider than this are not an envelope (mirrors `MAX_BEAT_HZ`). */
const MAX_BEAT_HZ = 1000;

export type WiringKind =
  /** No frequencies given — the positional reading, exactly as before. */
  | "positional"
  /** Frequencies resolve to at least one beating carrier. */
  | "modulated"
  /** No difference frequency anywhere: the envelope is identically zero. */
  | "no-modulation"
  /** The server will decide; this readout will not guess. */
  | "unresolved";

export interface WiringHint {
  kind: WiringKind;
  /** One line for the job row. */
  summary: string;
  /** True when two or more channels share a frequency, so electrode order matters. */
  coherentSum: boolean;
}

/** Parse a comma-separated drive string; `null` when it is blank or malformed. */
export function parseDriveList(raw: string | undefined): number[] | null {
  if (raw == null) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const parts = trimmed.split(",").map((p) => p.trim());
  if (parts.some((p) => p === "")) return null;
  const values = parts.map(Number);
  return values.some((v) => !Number.isFinite(v)) ? null : values;
}

function pluralize(count: number, noun: string, plural?: string): string {
  return `${count} ${count === 1 ? noun : (plural ?? `${noun}s`)}`;
}

/**
 * Describe what the given per-channel frequencies imply.
 *
 * @param frequencies One per electrode pair, or `null` for the positional reading.
 * @param nChannels   How many electrode pairs the row actually has.
 */
export function describeCarrierWiring(
  frequencies: number[] | null,
  nChannels: number,
): WiringHint {
  if (frequencies == null || frequencies.length === 0) {
    return {
      kind: "positional",
      summary: "No frequencies set — each pair treated as its own carrier",
      coherentSum: false,
    };
  }
  if (frequencies.length !== nChannels || frequencies.some((f) => !(f > 0))) {
    return { kind: "unresolved", summary: "Frequencies incomplete", coherentSum: false };
  }

  // Group by frequency, ascending. Iterated as entries throughout, because
  // `noUncheckedIndexedAccess` makes raw indexing `number | undefined`.
  const groups: { freq: number; members: number[] }[] = [];
  for (const entry of frequencies
    .map((f, i) => ({ f, i }))
    .sort((a, b) => a.f - b.f || a.i - b.i)) {
    const last = groups[groups.length - 1];
    if (last && Math.abs(entry.f - last.freq) <= FREQ_IDENTICAL_TOL_HZ) {
      last.members.push(entry.i);
    } else {
      groups.push({ freq: entry.f, members: [entry.i] });
    }
  }

  const coherentSum = groups.some((g) => g.members.length > 1);
  const freqs = groups.map((g) => g.freq);
  const head = [
    pluralize(nChannels, "channel"),
    pluralize(groups.length, "frequency", "frequencies"),
  ];
  const tail = coherentSum ? " \u00b7 coherent sum (electrode order matters)" : "";

  const seps = new Set<number>();
  for (const [i, fi] of freqs.entries()) {
    for (const [j, fj] of freqs.entries()) {
      if (j > i && fj - fi > FREQ_IDENTICAL_TOL_HZ) seps.add(fj - fi);
    }
  }
  const deltaF = seps.size ? Math.min(...seps) : null;

  if (deltaF == null || deltaF > MAX_BEAT_HZ) {
    return {
      kind: "no-modulation",
      summary: `${head.join(" \u00b7 ")} \u00b7 no modulation (TI_max = 0)${tail}`,
      coherentSum,
    };
  }

  const beats = (a: number, b: number) =>
    Math.abs(Math.abs(b - a) - deltaF) <= FREQ_IDENTICAL_TOL_HZ;

  // More than one possible partner means more than one difference frequency;
  // `derive_wiring` raises there, so this defers rather than guessing.
  for (const [i, fi] of freqs.entries()) {
    if (freqs.filter((fj, j) => j !== i && beats(fi, fj)).length > 1) {
      return {
        kind: "unresolved",
        summary: `${head.join(" \u00b7 ")} \u00b7 ambiguous pairing`,
        coherentSum,
      };
    }
  }

  const matched = new Set<number>();
  let nPairs = 0;
  for (const [i, fi] of freqs.entries()) {
    if (matched.has(i)) continue;
    for (const [j, fj] of freqs.entries()) {
      if (j > i && !matched.has(j) && beats(fi, fj)) {
        matched.add(i);
        matched.add(j);
        nPairs += 1;
        break;
      }
    }
  }

  // A leftover close enough to beat at some *other* separation is rejected
  // server-side, so this defers too.
  for (const [i, fi] of freqs.entries()) {
    if (matched.has(i)) continue;
    for (const [j, fj] of freqs.entries()) {
      if (j === i) continue;
      const sep = Math.abs(fj - fi);
      if (sep > FREQ_IDENTICAL_TOL_HZ && sep <= MAX_BEAT_HZ && !beats(fi, fj)) {
        return {
          kind: "unresolved",
          summary: `${head.join(" \u00b7 ")} \u00b7 more than one difference frequency`,
          coherentSum,
        };
      }
    }
  }

  const unpaired = freqs.length - matched.size;
  const parts = [...head, pluralize(nPairs, "carrier pair"), `\u0394f = ${deltaF} Hz`];
  if (unpaired > 0) parts.push(`${unpaired} non-beating`);
  return {
    kind: "modulated",
    summary: `${parts.join(" \u00b7 ")}${tail}`,
    coherentSum,
  };
}
