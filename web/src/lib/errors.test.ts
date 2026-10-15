import { describe, expect, it } from "vitest";
import { formatErrorDetail } from "./errors";

describe("API error messages", () => {
  it("keeps plain messages", () => {
    expect(formatErrorDetail("unknown implant x", "Bad Request")).toBe("unknown implant x");
    expect(formatErrorDetail(undefined, "Bad Request")).toBe("Bad Request");
  });

  it("turns pydantic validation errors into readable text", () => {
    const detail = [
      { type: "greater_than_equal", loc: ["body", "protocol", "sequences", 0, "b1rms_uT"], msg: "Input should be greater than or equal to 0" },
      { type: "finite_number", loc: ["body", "scanner", "b0_tesla"], msg: "Input should be a finite number" },
    ];
    expect(formatErrorDetail(detail, "x")).toBe(
      "protocol.sequences.0.b1rms_uT: Input should be greater than or equal to 0; scanner.b0_tesla: Input should be a finite number",
    );
  });
});
