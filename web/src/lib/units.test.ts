import { describe, expect, it } from "vitest";
import type { FieldValue } from "../types";
import {
  fmtNumber,
  formatB0,
  formatFieldValue,
  gaussPerCmToTeslaPerMeter,
  milliTeslaPerMeterToTeslaPerMeter,
  numberInputError,
  parseNumberInput,
  roundSig,
  secondsToMinutes,
  teslaPerMeterToGaussPerCm,
} from "./units";

function fv(partial: Partial<FieldValue>): FieldValue {
  return {
    status: "EXTRACTED",
    value: null,
    unit: null,
    normalized_value: null,
    normalized_unit: null,
    comparator: null,
    evidence: [],
    review_reasons: [],
    ...partial,
  };
}

describe("unit conversion", () => {
  it("converts T/m and G/cm (1 T/m = 100 G/cm)", () => {
    expect(teslaPerMeterToGaussPerCm(30)).toBe(3000);
    expect(teslaPerMeterToGaussPerCm(7.2)).toBe(720);
    expect(gaussPerCmToTeslaPerMeter(720)).toBe(7.2);
    expect(gaussPerCmToTeslaPerMeter(4000)).toBe(40);
    expect(gaussPerCmToTeslaPerMeter(teslaPerMeterToGaussPerCm(19))).toBe(19);
  });

  it("converts mT/m and seconds", () => {
    expect(milliTeslaPerMeterToTeslaPerMeter(40)).toBe(0.04);
    expect(secondsToMinutes(900)).toBe(15);
  });

  it("removes floating point noise", () => {
    expect(roundSig(0.1 * 3)).toBe(0.3);
    expect(roundSig(1.2000000000000002)).toBe(1.2);
    expect(fmtNumber(3.0)).toBe("3");
    expect(fmtNumber(3000)).toBe("3,000");
  });

  it("parses user-entered numbers", () => {
    expect(parseNumberInput("1,5")).toBe(1.5);
    expect(parseNumberInput("3,000")).toBe(3000);
    expect(parseNumberInput(" 2.8 ")).toBe(2.8);
    expect(parseNumberInput("")).toBeNull();
    expect(parseNumberInput("abc")).toBeNull();
  });

  it("accepts only plain decimal numbers", () => {
    expect(parseNumberInput("0x10")).toBeNull();
    expect(parseNumberInput("1e3")).toBeNull();
    expect(parseNumberInput("Infinity")).toBeNull();
    expect(parseNumberInput("2.0 W/kg")).toBeNull();
    expect(parseNumberInput(".5")).toBe(0.5);
    expect(parseNumberInput("-1")).toBe(-1); // parsed, then rejected by numberInputError
  });

  it("never rounds user input onto a limit (G/cm -> T/m)", () => {
    expect(gaussPerCmToTeslaPerMeter(3000)).toBe(30);
    expect(gaussPerCmToTeslaPerMeter(70)).toBe(0.7);
    expect(gaussPerCmToTeslaPerMeter(3000.0004)).toBeGreaterThan(30);
  });

  it("reports why an entry cannot be used", () => {
    expect(numberInputError("")).toBeNull(); // not entered
    expect(numberInputError("", { required: true })).toBe("Required");
    expect(numberInputError("abc")).toBe("Not a number");
    expect(numberInputError("-0.1")).toBe("Must be at least 0");
    expect(numberInputError("0", { minExclusive: true })).toBe("Must be above 0");
    expect(numberInputError("1.5", { integer: true, min: 1 })).toBe("Must be a whole number");
    expect(numberInputError("2000", { max: 1440 })).toBe("Must be at most 1,440");
    expect(numberInputError("1,5")).toBeNull();
  });
});

describe("value display", () => {
  it("shows spatial field gradient in T/m and G/cm", () => {
    expect(
      formatFieldValue("max_sfg_t_per_m", fv({ normalized_value: 25, normalized_unit: "T/m", comparator: "le" })),
    ).toBe("≤ 25 T/m (2,500 G/cm)");
  });

  it("uses the micro sign for B1+rms", () => {
    expect(
      formatFieldValue("b1rms_max_uT", fv({ normalized_value: 1.7, normalized_unit: "uT", comparator: "le" })),
    ).toBe("≤ 1.7 µT");
  });

  it("shows lower limits and explicit no-limit statements", () => {
    expect(
      formatFieldValue("wait_time_min", fv({ normalized_value: 30, normalized_unit: "min", comparator: "ge" })),
    ).toBe("≥ 30 min");
    expect(formatFieldValue("max_active_scan_time_min", fv({ comparator: "any", value: "no limit" }))).toBe(
      "no limit (explicitly unrestricted)",
    );
  });

  it("formats field strengths", () => {
    expect(formatB0({ normalized_value: [1.5, 3], comparator: "in" })).toBe("1.5 T or 3 T");
    expect(formatB0({ normalized_value: [3], comparator: "le" })).toBe("≤ 3 T");
  });

  it("labels categorical tokens and not-stated fields", () => {
    expect(formatFieldValue("transmit_coil", fv({ normalized_value: ["body", "head"] }))).toBe("body coil, head coil");
    expect(formatFieldValue("operating_mode", fv({ normalized_value: "normal" }))).toBe("Normal Operating Mode");
    expect(formatFieldValue("wb_sar_max_w_per_kg", fv({ status: "NOT_STATED" }))).toBe("not stated");
  });
});
