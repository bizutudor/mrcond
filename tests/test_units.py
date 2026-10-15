import pytest

from mrcond.units import (
    convert,
    find_quantities,
    find_word_durations,
    gauss_per_cm_to_tesla_per_m,
    normalize_unit,
    parse_number,
    round_sig,
    tesla_per_m_to_gauss_per_cm,
    to_canonical,
)


@pytest.mark.parametrize(
    "text,value", [("3,000", 3000.0), ("1,5", 1.5), ("2.8", 2.8), ("720", 720.0), ("12,345.5", 12345.5)]
)
def test_parse_number(text, value):
    assert parse_number(text) == value


def test_gradient_conversions():
    assert tesla_per_m_to_gauss_per_cm(1) == 100
    assert gauss_per_cm_to_tesla_per_m(720) == 7.2
    assert gauss_per_cm_to_tesla_per_m(4000) == 40
    assert convert(30, "T/m", "G/cm") == 3000
    assert convert(3000, "gauss/cm", "T/m") == 30
    assert to_canonical(25, "mT/m") == (0.025, "T/m")
    assert to_canonical(1900, "Gauss/cm") == (19, "T/m")


@pytest.mark.parametrize("unit", ["µT", "μT", "uT", "microtesla", "micro-Tesla", "mcT"])
def test_b1_spellings(unit):
    info = normalize_unit(unit)
    assert info is not None and info.dimension == "b1" and info.factor == 1.0


@pytest.mark.parametrize("unit", ["W/kg", "W/Kg", "watts per kilogram", "W / kg"])
def test_sar_spellings(unit):
    assert normalize_unit(unit).unit == "W/kg"


def test_time_and_length():
    assert to_canonical(900, "s") == (15, "min")
    assert to_canonical(90, "seconds") == (1.5, "min")
    assert to_canonical(1, "hour") == (60, "min")
    assert to_canonical(2, "cm") == (20, "mm")


def test_incompatible_units_raise():
    with pytest.raises(ValueError):
        convert(2, "W/kg", "T/m")
    with pytest.raises(ValueError):
        to_canonical(2, "furlong")


def test_round_sig_removes_float_noise():
    assert round_sig(0.1 * 3) == 0.3
    assert round_sig(1.2000000000000002) == 1.2


def test_find_quantities_shared_unit_and_pairs():
    qs = find_quantities("Static magnetic field of 1.5 and 3.0 Tesla only")
    assert [q.canonical_value for q in qs] == [1.5, 3.0]
    qs = find_quantities("Maximum Spatial Field Gradient 30 T/m (3,000 gauss /cm)")
    assert [(q.unit, q.canonical_value) for q in qs] == [("T/m", 30.0), ("G/cm", 30.0)]
    qs = find_quantities("2-W/kg for 15 minutes; 1.5T or 3.0T; 200 T/m/s per axis; B1+rms ≤ 2.8 µT; 5mm")
    assert [(q.dimension, q.canonical_value) for q in qs] == [
        ("sar", 2.0),
        ("time", 15.0),
        ("field", 1.5),
        ("field", 3.0),
        ("slew", 200.0),
        ("b1", 2.8),
        ("length", 5.0),
    ]


def test_rates_and_products_are_not_field_strengths():
    # dB/dt in T/s and the B0 x gradient product in T²/m must never be read as a static field
    assert find_quantities("dB/dt of 60 T/s or less")[0].dimension == "dbdt"
    assert find_quantities("product < 25 T²/m")[0].dimension == "product"
    assert not [q for q in find_quantities("B1+rms, 31P, B0") if q.dimension == "field"]


def test_word_durations():
    qs = find_word_durations("a maximum scan time of one hour, then thirty minutes")
    assert [q.canonical_value for q in qs] == [60.0, 30.0]
