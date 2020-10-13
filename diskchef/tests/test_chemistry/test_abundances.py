import os
from diskchef.chemistry.abundances import Abundances

data_path = os.path.join(os.path.dirname(__file__), "../data")

def test_read():
    file = os.path.join(data_path, "3abunds.inp")
    abunds = Abundances.from_3abunds(file)
    assert len(abunds) == 15
    assert abunds["SI"] == 9.74e-9

def test_default():
    abunds = Abundances()
    assert abunds["H2"] == 0.5
    assert abunds["CO"] == 0.5e-4