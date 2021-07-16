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

DRAINE_UV_FIELD = 2.6e-6 * u.W / u.m ** 2


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
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature   n(H+2H2)   log(numberdens) log(temperature)      CO           CN          HCN          HNC          HCO+         H2CO         N2H+          CS          C2H           e-
         AU           AU                         g / cm3      g / cm3           K               K           1 / cm3
    ------------ ------------ ---------------- ------------ ------------ --------------- ---------------- ------------ --------------- ---------------- ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------
    1.000000e-01 0.000000e+00     0.000000e+00 1.153290e-15 1.153290e-17    7.096268e+02     7.096268e+02 5.095483e+08    8.474334e+00     2.851030e+00 1.664660e-10 2.075371e-13 7.722633e-14 4.447148e-14 5.587394e-12 5.853986e-16 3.863739e-17 3.872823e-16 3.844251e-13 7.300494e-05
    1.000000e-01 3.500000e-02     3.500000e-01 5.024587e-34 5.024587e-36    3.548134e+03     3.548134e+03 2.219970e-10   -9.886505e+00     3.550000e+00 7.899202e-20 6.030650e-22 1.383617e-33 3.440258e-34 1.810430e-25 8.080346e-35 3.133545e-28 8.086328e-35 3.225259e-34 6.885677e-03
    1.000000e-01 7.000000e-02     7.000000e-01 5.921268e-72 5.921268e-74    3.548134e+03     3.548134e+03 2.616143e-48   -4.781519e+01     3.550000e+00 7.899202e-20 6.030650e-22 1.383617e-33 3.440258e-34 1.810430e-25 8.080346e-35 3.133545e-28 8.086328e-35 3.225259e-34 6.885677e-03
    7.071068e+00 0.000000e+00     0.000000e+00 2.285168e-13 2.285168e-15    6.820453e+01     6.820453e+01 1.009636e+11    1.077131e+01     1.833813e+00 6.431126e-05 2.138011e-18 3.018989e-14 2.846458e-14 4.012340e-12 4.361136e-15 3.347467e-15 9.598290e-12 3.676756e-18 3.737908e-11
    7.071068e+00 2.474874e+00     3.500000e-01 2.716386e-17 2.716386e-19    3.410227e+02     3.410227e+02 1.200157e+07    6.846386e+00     2.532783e+00 3.196870e-15 9.588096e-18 7.029195e-25 5.756870e-26 2.279574e-19 9.920758e-30 2.743343e-21 2.316110e-28 1.619243e-28 2.640124e-04
    7.071068e+00 4.949747e+00     7.000000e-01 7.189026e-23 7.189026e-25    3.410227e+02     3.410227e+02 3.176265e+01    1.269065e+00     2.532783e+00 7.899202e-20 6.030650e-22 1.383617e-33 3.440258e-34 1.810430e-25 8.080346e-35 3.133545e-28 8.086328e-35 3.225259e-34 6.885677e-03
    5.000000e+02 0.000000e+00     0.000000e+00 2.871710e-20 2.871710e-22    6.555359e+00     6.555359e+00 1.268783e+04    3.870536e+00     8.165965e-01 7.246437e-09 1.639502e-09 9.789595e-12 6.854840e-12 1.985437e-12 3.611351e-12 1.609762e-16 2.563294e-12 2.442295e-10 4.478516e-05
    5.000000e+02 1.750000e+02     3.500000e-01 6.929036e-22 6.929036e-24    2.625083e+01     2.625083e+01 3.061396e+02    2.253068e+00     1.419143e+00 9.592822e-12 3.264297e-14 2.652106e-14 1.248681e-14 8.773964e-14 4.373150e-18 6.765207e-18 1.123771e-21 1.185412e-18 7.771940e-05
    5.000000e+02 3.500000e+02     7.000000e-01 8.170464e-23 8.170464e-25    3.277680e+01     3.277680e+01 3.609885e+01    1.324642e+00     1.515566e+00 3.185463e-12 1.816816e-16 2.245415e-18 1.057603e-18 5.366938e-14 2.360676e-22 1.899195e-20 9.770146e-23 1.644348e-23 8.516644e-05
    """
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
                    (self.table["Gas density"] / self.molar_mass * c.N_A).to(u.cm ** -3).value
                )
            elif argument == "log(temperature)":
                self.table["log(temperature)"] = np.log10(self.table["Gas temperature"].to(u.K).value)
            elif argument == "log(uv)":
                self.table["log(uv)"] = np.log10(
                    (self.table["UV radiation strength"] / DRAINE_UV_FIELD).to_value(u.dimensionless_unscaled))
            elif argument == "log(ionization)":
                self.table["log(ionization)"] = np.log10(self.table["Ionization rate"].to(1 / u.s).value)
            else:
                raise CHEFRuntimeError(f"{argument} or its precursor is not found in the original data")

        X = np.nan_to_num(np.array([self.table[col] for col in self._model.X]).T)
        y = self._model.predict(X)
        for species, abunds in zip(self._model.y, y.T):
            self.table[species] = 10 ** abunds
