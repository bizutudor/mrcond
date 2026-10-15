"""Generic scanner presets.

These are deliberately *generic* closed-bore presets for demonstration. They do
not describe any vendor's system. Always enter the specifications of the actual
scanner (field strength, gradient slew-rate per axis, available transmit coils,
RF excitation modes) from its documentation.
"""

from __future__ import annotations

from .schema import ScannerProfile

PRESETS: dict[str, ScannerProfile] = {
    p.profile_id: p
    for p in [
        ScannerProfile(
            profile_id="generic-1.5T",
            name="Generic 1.5 T closed-bore (cylindrical, horizontal B0)",
            b0_tesla=1.5,
            scanner_type="cylindrical",
            b0_orientation="horizontal",
            max_slew_rate_t_per_m_per_s=200.0,
            transmit_coils=["body", "head"],
            rf_excitation_modes=["CP"],
            notes="Generic preset - replace with the actual scanner specifications.",
        ),
        ScannerProfile(
            profile_id="generic-3T",
            name="Generic 3 T closed-bore (cylindrical, horizontal B0)",
            b0_tesla=3.0,
            scanner_type="cylindrical",
            b0_orientation="horizontal",
            max_slew_rate_t_per_m_per_s=200.0,
            transmit_coils=["body", "head"],
            rf_excitation_modes=["CP", "MC-2"],
            notes="Generic preset - replace with the actual scanner specifications.",
        ),
    ]
}


def get_profile(profile_id: str) -> ScannerProfile:
    return PRESETS[profile_id].model_copy(deep=True)
