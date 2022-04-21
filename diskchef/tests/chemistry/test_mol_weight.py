from diskchef.engine import weights
import pytest


@pytest.mark.parametrize(
    "input, expected",
    (
            ('H', 1),
            ('D2O', 20),
            ("H2CO", 30),
            ("HCO+", 29),
            ("gH2", 2),
            ("aH3-", 3),
    )
)
def test_weight_correct(input, expected):
    assert int(weights.mol_weight(input).value) == expected


@pytest.mark.parametrize(
    "inwrut",
    (
            ("hello"),
            ("Hello"),
            ("FeSiM"),
            ("gggH2"),
            ("gagH2"),
            ("aagH2"),
            ("aaH2"),
            ("H++"),
            ("H+-"),
            ("H--"),
    )
)
def test_weight_wrong(inwrut):
    with pytest.raises(ValueError):
        weights.mol_weight(inwrut)
