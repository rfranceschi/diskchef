from collections import namedtuple
from typing import List, NamedTuple, Union
from dataclasses import dataclass

import numpy as np
from astropy import units as u

from diskchef.physics.base import PhysicsBase


@dataclass
class ParametrizedPhysics(PhysicsBase):
    """Base class for all parameter-based physics"""
    pass


_PowerLawPiece = namedtuple("PowerLawPiece", ["r_in", "r_out", "slope", "value_1au"])


@u.quantity_input
def powerlawpiece(
        r_in: u.au, r_out: u.au,
        slope: float, value_1au: u.Quantity
):
    """
        Return namedtuple with parameter for a piece power law with astropy units

        Args:
            r_in: inner radius (if r < r_in: return 0), in astropy.units length
            r_out: outer radius (if r > r_out: return 0), in astropy.units length
            slope: slope of the power law
            value_1au: value at r == 1 au
        """
    slope = float(slope)
    return _PowerLawPiece(r_in=r_in, r_out=r_out, slope=slope, value_1au=value_1au)


class PieceWisePowerLaw:
    def __init__(self, pieces: Union[List[_PowerLawPiece], List[List]]):
        """
        Callable class that returns a piecewise power law profile

        Args:
            pieces: list of PowerLawPiece(r_in, r_out, slope, value_1au)
        """
        try:
            self.unit = pieces[0][-1].unit
        except AttributeError:
            self.unit = u.dimensionless_unscaled
        self.pieces = [
            piece if isinstance(piece, _PowerLawPiece)
            else powerlawpiece(
                *piece[:-1],
                (piece[-1] * u.dimensionless_unscaled).to(self.unit))
            for piece in pieces
        ]

    @u.quantity_input
    def __call__(self, r: u.au):
        out = np.zeros(r.shape) * self.unit
        for piece in self.pieces:
            mask = (r >= piece.r_in) & (r < piece.r_out)
            out[mask] += piece.value_1au * np.power(
                (r[mask].to(u.au) / u.au).value,
                piece.slope)
        return out


@dataclass
class PowerLawPhysics(ParametrizedPhysics):
    """Class that outputs a physical model of the disk built as a collection of power laws"""
    column_density_profile: PieceWisePowerLaw = PieceWisePowerLaw(
        [[0.1 * u.au, 500 * u.au, -1.5, 100 * u.g / u.cm ** 2]]
    )
    midplane_temperature_profile: PieceWisePowerLaw = PieceWisePowerLaw(
        [[0.1 * u.au, 500 * u.au, -2, 100 * u.K]]
    )
    flaring_power: float = 1.26
    scale_hight_1au = 0.106 * u.au

    @u.quantity_input
    def density_scalehight(self, r: u.au) -> u.au:
        """Density scalehight of a flared disk from
        Pierre-Olivier Lagage1,*, Coralie Doucet1, Eric Pantin1, Emilie Habart2, Gaspard Duchêne3, François Ménard3, Christophe Pinte3, Sébastien Charnoz1, Jan-Willem Pel4,5

        Science  27 Oct 2006:
        Vol. 314, Issue 5799, pp. 621-623
        DOI: 10.1126/science.1131436

        default is 0.106 * r ** 1.26
        """
        return self.scale_hight_1au * (r / u.au) ** self.flaring_power

    @u.quantity_input(r=u.au, z=u.au)
    def gas_density(self, r, z) -> u.g / u.cm ** 3:
        """Calculates gas density at given r, z

        Assumes the PieceWisePowerLaw column density profile
        gaussian-scaled with the density scalehight
        """
        density_scalehight = self.density_scalehight(r)
        return (
                self.column_density_profile(r) /
                np.sqrt(2 * np.pi) / density_scalehight *
                np.exp(-z ** 2 / 2 / density_scalehight ** 2)
        )

    @u.quantity_input(r=u.au, z=u.au)
    def dust_temperature(self, r, z) -> u.K:
        """Calculates dust temperature at given r, z"""
        return self.midplane_temperature_profile(r)
