from diskchef.physics import WilliamsBest2014
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterMathtext


def test_temperature_range():
    """Test that if the colorbar range is small, the different tick formatter is used"""
    phys = WilliamsBest2014(
        r_min=10 * u.au, r_max=100 * u.au,
        midplane_temperature_1au=100 * u.K, atmosphere_temperature_1au=120 * u.K,
    )
    plot = phys.plot_temperatures()
    # plt.show()
    assert isinstance(plot.cbar_formatter, LogFormatterMathtext)