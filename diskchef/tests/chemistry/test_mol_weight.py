from diskchef.engine import weights
import pytest

def test_weight():
    assert int(weights.mol_weight('H2').value) == 2
    assert int(weights.mol_weight('H2CO').value) == 30
    assert int(weights.mol_weight('Hello').value) == 4 # hence we prove Hello weighs 4 u
    assert int(weights.mol_weight('hello').value) == 0 # and that capitalisation creats mass in words