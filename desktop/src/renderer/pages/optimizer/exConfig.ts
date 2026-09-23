/**
 * Ex / mEx form state → `ExConfig` / `MExConfig` wire shapes, plus the bucket vocabulary and the
 * ROI→target expansion. Merged from `optimizer-ex/{ExForm,MExForm,lib}.tsx` — see
 * `pages/optimizer/PARITY.md` for the control-by-control map of where each field went.
 *
 * The one behavioural change from the predecessors: the target's ROI now comes from the ONE
 * shared `RoiValue` (`pages/_shared/roi`), so an atlas target carries the atlas's real
 * `atlas_path` (resolved through the catalog) instead of the atlas *id* the old local picker put
 * in that field.
 */
import type { RoiValue, AtlasLookup } from "../_shared/roi";
import type { ExConfigBody, MExConfigBody } from "./api";

export const EX_BUCKET_KEYS = ["e1_plus", "e1_minus", "e2_plus", "e2_minus"] as const;
export const MEX_BUCKET_KEYS = [
  "e1_plus",
  "e1_minus",
  "e2_plus",
  "e2_minus",
  "e3_plus",
  "e3_minus",
  "e4_plus",
  "e4_minus",
] as const;

export const BUCKET_LABELS: Record<string, string> = {
  e1_plus: "E1+",
  e1_minus: "E1-",
  e2_plus: "E2+",
  e2_minus: "E2-",
  e3_plus: "E3+",
  e3_minus: "E3-",
  e4_plus: "E4+",
  e4_minus: "E4-",
};

export const BUCKET_TOOLTIPS: Record<string, string> = {
  e1_plus: "Electrodes for channel 1 positive pole (anodes)",
  e1_minus: "Electrodes for channel 1 negative pole (cathodes)",
  e2_plus: "Electrodes for channel 2 positive pole (anodes)",
  e2_minus: "Electrodes for channel 2 negative pole (cathodes)",
  e3_plus: "Electrodes for pair 3 positive pole (anodes)",
  e3_minus: "Electrodes for pair 3 negative pole (cathodes)",
  e4_plus: "Electrodes for pair 4 positive pole (anodes)",
  e4_minus: "Electrodes for pair 4 negative pole (cathodes)",
};

export const MTI_SYMMETRY_HELP =
  "When checked, only left/right mirrored electrode-pair candidates are evaluated (mirroring " +
  "derived from the selected leadfield's EEG net), instead of the full combination of all eight " +
  "buckets.\n\n" +
  "Within each pair: each pair's own +/- electrodes mirror across the midline.\n" +
  "Cross pairs: additionally requires pair 1<->3 and pair 2<->4 to mirror each other.";

export const EX_CARRIER_HELP =
  "Ex-search takes the two channels to sit at two distinct carrier frequencies — the standard TI " +
  "arrangement, e.g. 2.000 and 2.010 kHz.\n\n" +
  "There is nothing to set here, because the ranking does not depend on what those frequencies " +
  "are. A two-channel envelope is phase-invariant: its modulation depth follows from the two " +
  "field amplitudes and the angle between them, not from the carrier values.\n\n" +
  "Two channels on the *same* frequency do not beat at all, so there would be no envelope to " +
  "maximise. Examine that case in the Simulator, where the carrier field (hf_peak / hf_sar) is " +
  "reported. Per-channel frequencies are configurable for mTI searches, which have two carriers.";

export const MEX_CARRIER_HELP =
  "One carrier frequency per bipolar pair. Leave them blank to keep the positional wiring: each " +
  "pair on its own carrier, pairs 1&2 beating and pairs 3&4 beating.\n\n" +
  "Setting them lets you express wirings the positional model cannot — notably two pairs sharing " +
  "each carrier (2000, 2000, 2010, 2010), where the two fields on one frequency add as vectors " +
  "and the electrode order within those pairs starts to matter.\n\n" +
  "Phase is the drive phase of each source; 180 degrees is exactly a polarity reversal. Unlike " +
  "Ex, this changes the result: with two carriers the envelope phase enters the modulation depth.\n\n" +
  "Each pair runs at its own current. The search ranks electrodes at the currents you set and " +
  "never varies them, so unbalanced pairs can rank montages differently from equal ones.";

export interface ExFormState {
  electrodeMode: "bucketed" | "all";
  buckets: Record<string, string[]>;
  pool: string[];
  totalCurrent: number;
  currentStep: number;
  channelLimit: number | null;
}

export interface MExFormState {
  buckets: Record<string, string[]>;
  symmetricBucket: boolean;
  symmetryPairing: "within_pairs" | "cross_pairs";
  /**
   * Per-channel carrier frequency (Hz), one per bipolar pair; `null` is *unset*, not zero.
   *
   * All four unset — the default — leaves the server on the positional wiring: each pair at its
   * own distinct frequency, consecutive pairs beating. That is what every mEx run before this
   * field meant, so an untouched form ranks candidates exactly as it always did.
   *
   * Setting them is the only way to express a wiring the positional model cannot, notably two
   * pairs sharing each carrier (`[2000, 2000, 2010, 2010]`). Unlike Ex, this genuinely changes the
   * objective: mEx is two carriers, where the envelope phase enters `Q` (`tit/calc.py`).
   */
  frequencies: (number | null)[];
  /** Per-channel drive phase (degrees); `null` means 0. 180 is a polarity reversal. */
  phases: (number | null)[];
  /**
   * Per-pair current (mA), one per bipolar pair — the only current control mEx has.
   *
   * A fixed input, like frequency: the search ranks electrodes at these currents and never varies
   * them. The default, 2.0 mA on every pair, is the single shared current every mEx run before
   * this field used.
   */
  currents: number[];
}

export function defaultExFormState(): ExFormState {
  return {
    electrodeMode: "bucketed",
    buckets: { e1_plus: [], e1_minus: [], e2_plus: [], e2_minus: [] },
    pool: [],
    totalCurrent: 2.0,
    currentStep: 0.2,
    channelLimit: 1.6,
  };
}

export function defaultMExFormState(): MExFormState {
  return {
    buckets: Object.fromEntries(MEX_BUCKET_KEYS.map((k) => [k, []])),
    symmetricBucket: false,
    symmetryPairing: "within_pairs",
    frequencies: [null, null, null, null],
    phases: [null, null, null, null],
    currents: [2.0, 2.0, 2.0, 2.0],
  };
}
/** How many bipolar pairs an mEx search drives — four, one per `e{n}_plus`/`e{n}_minus` bucket. */
export const MEX_CHANNELS = 4;

/**
 * The drive to send for this form, or nulls.
 *
 * A *partial* drive is treated as unset, the same rule the Simulator applies: a frequency on some
 * channels and not others has no meaning the server could act on. Phases are zero-filled rather
 * than sent alone, because the server rejects `phases_deg` without `frequencies_hz`.
 */
export function mexCarrierDrive(form: MExFormState): {
  frequencies_hz: number[] | null;
  phases_deg: number[] | null;
} {
  const complete =
    form.frequencies.length === MEX_CHANNELS &&
    form.frequencies.every((f): f is number => f != null && Number.isFinite(f) && f > 0);
  if (!complete) return { frequencies_hz: null, phases_deg: null };
  return {
    frequencies_hz: form.frequencies as number[],
    phases_deg: Array.from({ length: MEX_CHANNELS }, (_, i) => form.phases[i] ?? 0),
  };
}

/** The current each pair runs at (mA), in pair order. */
export function mexPairCurrents(form: MExFormState): number[] {
  return Array.from({ length: MEX_CHANNELS }, (_, i) => form.currents[i] ?? 2.0);
}

/**
 * `MExConfig.current_mA` / `currents_mA` for this form.
 *
 * Equal currents go as the single `current_mA` they always were, so run names and saved configs
 * of an equal-current run are unchanged. Unequal ones go as `currents_mA`, which the server uses
 * in place of `current_mA`; `current_mA` then only has to be a valid positive value.
 */
export function mexCurrentsPayload(form: MExFormState): { current_mA: number; currents_mA: number[] | null } {
  const currents = mexPairCurrents(form);
  const first = currents[0] ?? 2.0;
  return { current_mA: first, currents_mA: currents.every((c) => c === first) ? null : currents };
}

/** One resolved ex/mEx run target: what `roi_name` / `roi_names` / `roi_atlas` will carry. */
export interface ExTarget {
  /** `ExConfig.roi_name` — also the run's output label. */
  roiName: string;
  /** `ExConfig.roi_names` — the saved CSV names unioned into this target, or `[]` for atlas-only. */
  roiNames: string[] | null;
  /** `ExConfig.roi_atlas` — volumetric regions unioned into the target, or `null`. */
  roiAtlas: { atlas_path: string; label: number | null; atlas_space?: "subject" | "mni" }[] | null;
  radius: number;
  space: "subject" | "mni";
}

/**
 * Expand the shared ROI value into the ex/mEx target list.
 *
 * - `saved` + combine → one target named by joining the ROI names with "+".
 * - `saved` without combine → one target per selected ROI (one run each).
 * - `subcortical` → a single atlas-only target (`roi_names: []`, the schema's "no spherical
 *   centres" signal), named for the atlas and its region count.
 * - anything else, or nothing selected → no targets, which is what disables Run.
 *
 * `allowCombine === false` (mEx) forces the per-ROI expansion regardless of the flag, matching the
 * mTI run path, which has no combined mode.
 */
export function exTargets(
  roi: RoiValue,
  atlasLookup: (atlas: string) => AtlasLookup | undefined,
  allowCombine = true,
): ExTarget[] {
  if (roi.mode === "mask") {
    const path = roi.path.trim();
    if (!/\.nii(\.gz)?$/i.test(path)) return [];
    return [{ roiName: path.split("/").pop()!.replace(/\.nii(\.gz)?$/i, ""), roiNames: [],
      roiAtlas: [{ atlas_path: path, label: null, atlas_space: roi.space }], radius: 3, space: roi.space }];
  }
  if (roi.mode === "saved") {
    if (roi.selected.length === 0) return [];
    if (roi.combine && allowCombine) {
      return [{ roiName: roi.selected.join("+"), roiNames: [...roi.selected], roiAtlas: null, radius: roi.radius, space: roi.space }];
    }
    return roi.selected.map((name) => ({ roiName: name, roiNames: null, roiAtlas: null, radius: roi.radius, space: roi.space }));
  }
  if (roi.mode === "subcortical") {
    const atlas = roi.atlas ? atlasLookup(roi.atlas) : undefined;
    if (!atlas || roi.regions.length === 0) return [];
    const n = roi.regions.length;
    return [
      {
        roiName: `${roi.atlas}_${n}region${n === 1 ? "" : "s"}`,
        roiNames: [],
        roiAtlas: roi.regions.map((r) => ({ atlas_path: atlas.path, label: r.id, atlas_space: roi.space })),
        radius: 3.0,
        space: roi.space,
      },
    ];
  }
  return [];
}

export function buildExConfig(
  subjectId: string,
  leadfieldHdf: string,
  form: ExFormState,
  target: ExTarget,
  runName: string,
): ExConfigBody {
  const electrodes: ExConfigBody["electrodes"] =
    form.electrodeMode === "bucketed"
      ? {
          e1_plus: form.buckets.e1_plus ?? [],
          e1_minus: form.buckets.e1_minus ?? [],
          e2_plus: form.buckets.e2_plus ?? [],
          e2_minus: form.buckets.e2_minus ?? [],
          _type: "BucketElectrodes",
        }
      : { electrodes: form.pool, _type: "PoolElectrodes" };
  return {
    subject_id: subjectId,
    leadfield_hdf: leadfieldHdf,
    roi_name: target.roiName,
    electrodes,
    total_current: form.totalCurrent,
    current_step: form.currentStep,
    channel_limit: form.channelLimit,
    roi_radius: target.radius,
    roi_names: target.roiNames,
    roi_atlas: target.roiAtlas?.map((roi) => ({ ...roi, atlas_space: roi.atlas_space ?? "subject" })) ?? null,
    roi_coordinate_space: target.space,
    run_name: runName.trim() || null,
    n_jobs: -1,
    // Carrier drive. The Optimizer takes frequency as a fixed input, and the
    // page has no control for it on the two-channel path, because a
    // single-carrier envelope is phase-invariant: no value here could reorder
    // the ranking. Null keeps the positional reading.
    frequencies_hz: null,
    phases_deg: null,
    // Ex-search symmetric buckets (added on main in v2.5.0) have no control on
    // the Optimizer page yet; these are the server-side defaults. See
    // docs/dev/RELEASE.md §B.
    symmetric_bucket: false,
    symmetry_eeg_csv: null,
    symmetry_pairing: "within_pairs",
  };
}

export function buildMExConfig(
  subjectId: string,
  leadfieldHdf: string,
  form: MExFormState,
  target: ExTarget,
  runName: string,
): MExConfigBody {
  return {
    subject_id: subjectId,
    leadfield_hdf: leadfieldHdf,
    roi_name: target.roiName,
    electrodes: {
      e1_plus: form.buckets.e1_plus ?? [],
      e1_minus: form.buckets.e1_minus ?? [],
      e2_plus: form.buckets.e2_plus ?? [],
      e2_minus: form.buckets.e2_minus ?? [],
      e3_plus: form.buckets.e3_plus ?? [],
      e3_minus: form.buckets.e3_minus ?? [],
      e4_plus: form.buckets.e4_plus ?? [],
      e4_minus: form.buckets.e4_minus ?? [],
      _type: "BucketElectrodes",
    },
    ...mexCurrentsPayload(form),
    roi_radius: target.radius,
    roi_names: target.roiNames,
    roi_atlas: target.roiAtlas?.map((roi) => ({ ...roi, atlas_space: roi.atlas_space ?? "subject" })) ?? null,
    roi_coordinate_space: target.space,
    run_name: runName.trim() || null,
    symmetric_bucket: form.symmetricBucket,
    symmetry_eeg_csv: null,
    symmetry_pairing: form.symmetryPairing,
    n_jobs: -1,
    // Carrier drive. Null keeps mEx's positional wiring — each pair its own
    // carrier, consecutive pairs beating — which is what an untouched form
    // means and the only wiring its pool-mode symmetry classes are valid for.
    ...mexCarrierDrive(form),
  };
}

export function formatBytes(n: number): string {
  if (n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log2(n) / 10));
  return `${(n / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/** `exTargets` for a saved-ROI selection, without constructing a `RoiValue` — the shape unit
 *  tests and callers that already have names want. */
export function savedTargets(names: string[], combine: boolean, radius = 3.0, space: "subject" | "mni" = "subject"): ExTarget[] {
  return exTargets({ mode: "saved", selected: names, combine, radius, space }, () => undefined, true);
}

/** `exTargets` for one subcortical atlas selection, given the atlas's resolved path. */
export function atlasTarget(atlasId: string, atlasPath: string, labels: number[], space: "subject" | "mni" = "subject"): ExTarget {
  const value: RoiValue = {
    mode: "subcortical",
    space,
    atlas: atlasId,
    regions: labels.map((id) => ({ id, name: String(id) })),
    tissues: "GM",
  };
  return exTargets(value, () => ({ path: atlasPath }), true)[0] as ExTarget;
}
