import pytest

from diskchef.chemistry.williams_best_co import ChemistryWB2014
from diskchef.physics.williams_best import WilliamsBest2014

@pytest.fixture
def wb():
    physics = WilliamsBest2014(radial_bins=5, vertical_bins=5)
    chemistry = ChemistryWB2014(physics)
    return chemistry

def test_wb_chem(wb):
    wb.run_chemistry()