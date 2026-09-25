/**
 * The Simulator's carrier-drive readout and the payload it produces.
 *
 * `tit/wiring.py` is the authority on what a set of frequencies means; these pin that the page
 * (a) tells the user when their montage has no beat, before they submit, and (b) sends nothing
 * at all when they did not set a frequency, so an untouched row runs exactly as it always did.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createAjvResolver } from "../../src/renderer/forms/ajvResolver";
import { resetSchemaCache, type JSONSchema } from "../../src/renderer/forms/schema";
import { buildSimulationConfig, type GlobalParams } from "../../src/renderer/pages/simulator/buildConfig";
import {
  describeCarrierWiring,
  parseDriveList,
} from "../../src/renderer/pages/_shared/carriers";
import {
  DEFAULT_JOB_SETTINGS,
  emptyRow,
  rowHasCarrierDrive,
  type SelectedRow,
} from "../../src/renderer/pages/simulator/types";

const schemaPath = join(__dirname, "..", "..", "..", "contracts", "generated", "config.schema.json");
const schemaDoc = JSON.parse(readFileSync(schemaPath, "utf8")) as JSONSchema;

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(schemaDoc), { status: 200, headers: { "content-type": "application/json" } })),
  );
  resetSchemaCache();
});

async function validate(defName: string, values: Record<string, unknown>) {
  const resolver = createAjvResolver<Record<string, unknown>>(defName);
  const result = await resolver(values, undefined, { shouldUseNativeValidation: false } as never);
  return { valid: Object.keys(result.errors).length === 0, errors: result.errors };
}

const DEFAULTS: GlobalParams = { ...DEFAULT_JOB_SETTINGS, customConductivities: {} };

function row(overrides: Partial<SelectedRow> = {}): SelectedRow {
  return {
    ...emptyRow("sub-01"),
    name: "F3_F4",
    pairs: [["E24", "E124"], ["E36", "E104"]],
    kind: "uni_polar",
    currents: "1.0,1.0",
    ...overrides,
  };
}

describe("parseDriveList", () => {
  it("reads a complete list", () => {
    expect(parseDriveList("2000,2010")).toEqual([2000, 2010]);
  });

  it("treats blank as unset rather than zero", () => {
    expect(parseDriveList("")).toBeNull();
    expect(parseDriveList("   ")).toBeNull();
    expect(parseDriveList(undefined)).toBeNull();
  });

  it("treats a partially filled list as unset", () => {
    // A frequency on one channel and not the other has no meaning the server could act on.
    expect(parseDriveList("2000,")).toBeNull();
    expect(parseDriveList(",2010")).toBeNull();
  });

  it("rejects non-numeric entries", () => {
    expect(parseDriveList("2000,abc")).toBeNull();
  });
});

describe("describeCarrierWiring", () => {
  it("says nothing was set when nothing was set", () => {
    expect(describeCarrierWiring(null, 2).kind).toBe("positional");
  });

  it("reports the beat for two distinct frequencies", () => {
    const hint = describeCarrierWiring([2000, 2010], 2);
    expect(hint.kind).toBe("modulated");
    expect(hint.summary).toContain("Δf = 10 Hz");
    expect(hint.coherentSum).toBe(false);
  });

  it("warns that one shared frequency means no modulation at all", () => {
    // The case this readout exists for: TI_max is correctly zero, and a blank
    // map would otherwise look like a failed run.
    const hint = describeCarrierWiring([2000, 2000], 2);
    expect(hint.kind).toBe("no-modulation");
    expect(hint.summary).toContain("TI_max = 0");
    expect(hint.coherentSum).toBe(true);
  });

  it("flags that electrode order matters once a frequency is shared", () => {
    expect(describeCarrierWiring([2000, 2000, 2010, 2010], 4).coherentSum).toBe(true);
    expect(describeCarrierWiring([2000, 2000, 2010, 2010], 4).summary).toContain("electrode order matters");
  });

  it("pairs each mTI carrier with its own partner", () => {
    const hint = describeCarrierWiring([2000, 2010, 4000, 4010], 4);
    expect(hint.kind).toBe("modulated");
    expect(hint.summary).toContain("2 carrier pairs");
  });

  it("counts a far-off carrier as non-beating", () => {
    expect(describeCarrierWiring([2000, 2010, 9000], 3).summary).toContain("1 non-beating");
  });

  it("treats two widely separated carriers as no modulation", () => {
    // 7 kHz apart is not an envelope any neuron follows.
    expect(describeCarrierWiring([2000, 9000], 2).kind).toBe("no-modulation");
  });

  it("defers an ambiguous pairing to the server instead of guessing", () => {
    expect(describeCarrierWiring([2000, 2010, 2020], 3).kind).toBe("unresolved");
  });

  it("defers a second difference frequency to the server", () => {
    expect(describeCarrierWiring([2000, 2010, 4000, 4020], 4).kind).toBe("unresolved");
  });

  it("does not resolve a drive that does not cover every channel", () => {
    expect(describeCarrierWiring([2000, 2010], 4).kind).toBe("unresolved");
  });
});

/**
 * The four-channel cases the Optimizer's mEx Carriers section renders. Same helper, same wording —
 * which is why it lives in `pages/_shared/carriers` rather than on either page.
 */
describe("describeCarrierWiring for mEx (four channels)", () => {
  it("describes the positional default as two beating carriers", () => {
    const hint = describeCarrierWiring([2000, 2010, 4000, 4010], 4);
    expect(hint.kind).toBe("modulated");
    expect(hint.summary).toContain("2 carrier pairs");
    expect(hint.coherentSum).toBe(false);
  });

  it("describes a shared carrier as one beating pair over two frequencies", () => {
    // Two pairs on each carrier: inexpressible before tit.wiring, and the reason
    // the mEx section carries these controls at all.
    const hint = describeCarrierWiring([2000, 2000, 2010, 2010], 4);
    expect(hint.kind).toBe("modulated");
    expect(hint.summary).toContain("2 frequencies");
    expect(hint.summary).toContain("1 carrier pair");
    expect(hint.coherentSum).toBe(true);
    expect(hint.summary).toContain("electrode order matters");
  });

  it("flags four channels on one frequency as no modulation", () => {
    // The server refuses this for an optimizer run, because every candidate would tie.
    expect(describeCarrierWiring([2000, 2000, 2000, 2000], 4).kind).toBe("no-modulation");
  });

  it("describes a crossed wiring, where non-consecutive channels beat", () => {
    const hint = describeCarrierWiring([2000, 4000, 2010, 4010], 4);
    expect(hint.kind).toBe("modulated");
    expect(hint.summary).toContain("2 carrier pairs");
  });
});

describe("rowHasCarrierDrive", () => {
  it("is false for an untouched row", () => {
    expect(rowHasCarrierDrive(row(), 2)).toBe(false);
  });

  it("is false while only some channels have a frequency", () => {
    expect(rowHasCarrierDrive(row({ frequencies: "2000," }), 2)).toBe(false);
  });

  it("is true once every channel has one", () => {
    expect(rowHasCarrierDrive(row({ frequencies: "2000,2010" }), 2)).toBe(true);
  });
});

describe("buildSimulationConfig carrier drive", () => {
  it("sends no drive keys at all for an untouched row", async () => {
    const config = buildSimulationConfig(row(), DEFAULTS);
    expect(config).not.toHaveProperty("frequencies_hz");
    expect((await validate("SimulationConfig", config)).valid).toBe(true);
  });

  it("sends the frequencies, and no phase, once every channel has one", async () => {
    const config = buildSimulationConfig(row({ frequencies: "2000,2010" }), DEFAULTS);
    expect(config.frequencies_hz).toEqual([2000, 2010]);
    // Polarity is electrode order; there is no phase on the wire.
    expect(config).not.toHaveProperty("phases_deg");
    expect((await validate("SimulationConfig", config)).valid).toBe(true);
  });

  it("carries a shared carrier, where electrode order sets the polarity", async () => {
    const config = buildSimulationConfig(row({ frequencies: "2000,2000" }), DEFAULTS);
    expect(config.frequencies_hz).toEqual([2000, 2000]);
    expect((await validate("SimulationConfig", config)).valid).toBe(true);
  });

  it("rejects a payload that still carries phases_deg", async () => {
    // The field is gone from the contract; SimulationConfig forbids unknown keys.
    const config = { ...buildSimulationConfig(row({ frequencies: "2000,2000" }), DEFAULTS), phases_deg: [0, 180] };
    expect((await validate("SimulationConfig", config)).valid).toBe(false);
  });
});
