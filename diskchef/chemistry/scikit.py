"""Class to read the scikit-learn models"""
from typing import Union
from dataclasses import dataclass
import pickle
import os
import copy

import numpy as np
from astropy import units as u
from astropy import constants as c
import sklearn

from diskchef.engine.other import PathLike
from diskchef.chemistry.base import ChemistryBase
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.engine.exceptions import CHEFNotImplementedError, CHEFRuntimeError


@dataclass
class SciKitChemistry(ChemistryBase):
    """Predicts chemistry based on `sklearn` trained models

    Usage:

    >>> physics = WilliamsBest2014(radial_bins=3, vertical_bins=3)
    >>> chem = SciKitChemistry(physics)
    >>> chem._model  # doctest: +NORMALIZE_WHITESPACE
    TransformedTargetRegressor(regressor=Pipeline(steps=[('quantiletransformer',
                                                          QuantileTransformer(n_quantiles=100)),
                                                         ('kneighborsregressor',
                                                          KNeighborsRegressor(n_neighbors=62))]),
                               transformer=QuantileTransformer(n_quantiles=100,
                                                               output_distribution='normal'))
    >>> chem._model.X
    ('log(numberdens)', 'log(temperature)')
    >>> chem._model.y
    ('CO', 'CN', 'HCN', 'HNC', 'HCO+', 'H2CO', 'N2H+', 'CS', 'C2H', 'e-')
    >>> chem.run_chemistry()
    >>> chem.table  # doctest: +NORMALIZE_WHITESPACE
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature log(numberdens) log(temperature)      CO           CN          HCN          HNC          HCO+         H2CO         N2H+          CS          C2H           e-
         AU           AU                         g / cm3      g / cm3           K               K
    ------------ ------------ ---------------- ------------ ------------ --------------- ---------------- --------------- ---------------- ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------
    1.000000e-01 0.000000e+00     0.000000e+00 1.153290e-06 1.153290e-08    7.096268e+02     7.096268e+02    1.747433e+01     2.851030e+00 3.957313e-05 1.034711e-20 6.443242e-08 1.004589e-11 4.602171e-22 1.148471e-16 9.356989e-23 5.283663e-09 1.629806e-20 8.953019e-19
    1.000000e-01 3.500000e-02     3.500000e-01 5.024587e-25 5.024587e-27    3.548134e+03     3.548134e+03   -8.865046e-01     3.550000e+00 7.899202e-20 6.030650e-22 1.383617e-33 3.440258e-34 1.810430e-25 8.080346e-35 3.133545e-28 8.086328e-35 3.225259e-34 6.885677e-03
    1.000000e-01 7.000000e-02     7.000000e-01 5.921268e-63 5.921268e-65    3.548134e+03     3.548134e+03   -3.881519e+01     3.550000e+00 7.899202e-20 6.030650e-22 1.383617e-33 3.440258e-34 1.810430e-25 8.080346e-35 3.133545e-28 8.086328e-35 3.225259e-34 6.885677e-03
    7.071068e+00 0.000000e+00     0.000000e+00 2.285168e-10 2.285168e-12    6.820453e+01     6.820453e+01    1.377131e+01     1.833813e+00 6.110678e-05 5.306787e-21 2.500877e-15 2.334290e-15 8.215606e-15 6.813926e-17 5.370184e-19 1.000208e-13 7.848889e-21 2.960001e-15
    7.071068e+00 2.474874e+00     3.500000e-01 2.716386e-14 2.716386e-16    3.410227e+02     3.410227e+02    9.846386e+00     2.532783e+00 5.216075e-06 7.275340e-09 5.540219e-08 5.519892e-09 5.787552e-10 6.843302e-12 9.254907e-12 1.244593e-09 3.664016e-08 4.506056e-07
    7.071068e+00 4.949747e+00     7.000000e-01 7.189026e-20 7.189026e-22    3.410227e+02     3.410227e+02    4.269065e+00     2.532783e+00 1.505247e-19 4.245336e-21 1.280808e-33 2.352613e-34 2.966060e-25 5.397051e-35 1.103334e-27 5.950178e-35 5.497506e-34 5.614276e-03
    5.000000e+02 0.000000e+00     0.000000e+00 2.871710e-17 2.871710e-19    6.555359e+00     6.555359e+00    6.870536e+00     8.165965e-01 5.155117e-06 4.467439e-10 4.646804e-09 1.183951e-09 8.429080e-11 1.676795e-10 3.320050e-12 3.666043e-11 1.159168e-11 3.160602e-09
    5.000000e+02 1.750000e+02     3.500000e-01 6.929036e-19 6.929036e-21    2.625083e+01     2.625083e+01    5.253068e+00     1.419143e+00 1.114742e-09 4.579831e-11 5.682458e-12 3.789283e-12 3.302587e-12 3.493642e-13 1.479063e-16 2.512731e-13 1.698792e-10 7.239851e-05
    5.000000e+02 3.500000e+02     7.000000e-01 8.170464e-20 8.170464e-22    3.277680e+01     3.277680e+01    4.324642e+00     1.515566e+00 1.439732e-11 5.804472e-16 4.176212e-17 2.125057e-17 1.202085e-13 1.151401e-20 5.088273e-20 3.080947e-21 4.804163e-21 8.000221e-05
    """
    mean_molecular_mass: u.g / u.mol = 2.33 * u.g / u.mol
    model: Union[PathLike, sklearn.base.TransformerMixin] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scikit_estimators", "andes2_atomic_knearest_temp_dens.pkl"
    )

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.model, "predict"):
            self._model = self.model
            self.logger.info("SciKit Transformer object is recognized")
        else:
            with open(self.model, 'rb') as fff:
                self._model = pickle.load(fff)
            if hasattr(self._model, "predict"):
                self.logger.info("SciKit Transformer object is read and recognized")
            else:
                raise CHEFNotImplementedError

    def run_chemistry(self):
        all_arguments = copy.copy(self._model.X)
        existing_arguments_with_log = {f"log({colname})" for colname in self.table.colnames}
        for argument in all_arguments:
            if argument in self.table.colnames:
                pass
            elif argument in existing_arguments_with_log:
                self.table[f"log({argument})"] = np.log10(
                    (self.table[argument]).cgs.value
                )
                self.logger.info("log(%s) was taken in CGS system", argument)
            elif argument == "log(numberdens)":
                self.table["log(numberdens)"] = np.log10(
                    (self.table["Gas density"] / self.mean_molecular_mass * c.N_A).to(u.cm ** -3).value
                )
            elif argument == "log(temperature)":
                self.table["log(temperature)"] = np.log10(self.table["Gas temperature"].to(u.K).value)
            else:
                raise CHEFRuntimeError(f"{argument} or its precursor is not found in the original data")

        y = self._model.predict(np.array([self.table[col] for col in self._model.X]).T)
        for species, abunds in zip(self._model.y, y.T):
            self.table[species] = 10 ** abunds
