from diskchef.engine import weights
import pytest


@pytest.mark.parametrize(
    "input, expected",
    (
            ("hello", 0),
            ("Hello", 0),
            ("FeSiM", 0),
            ('H', 1),
            ('D2O', 20),
            ("H2CO", 30),
            ("HCO+", 29),
            ("gH2", 2),
            ("gggH2", 0),
            ("gagH2", 0),
            ("aagH2", 0),
            ("aaH2", 0),
            ("aH3-", 3),
            ("H++", 0),
            ("H+-", 0),
            ("H--", 0),
    )
)
def test_weight(input, expected):
    assert int(weights.mol_weight(input).value) == expected
