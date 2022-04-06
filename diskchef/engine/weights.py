import re

import astropy.units as u


def mol_weight(mol: str) -> u.u:
    """
    Python version of ANDES / ALCHEMIC `aweight` FORTRAN function

    Uses the same element weights as ANDES, see more:
    https://gitlab.com/ANDES-dev/ANDES2/-/blob/f549fdf0/src/ChemistryAdds.f90#L62

    Args:
        mol: str -- String containing ANDES-standard species name

    Returns:
        molecular weight, in `astropy.units.u`

    Usage:
        >>> mol_weight('H2CO') #doctest: +ELLIPSIS
        <Quantity 30... u>
    """
    elements = {
        'H': 1.00790 * u.u,
        'C': 1.20110E+01 * u.u,
        'N': 1.40067E+01 * u.u,
        'O': 1.59994E+01 * u.u,
        'S': 3.20660E+01 * u.u,
        'P': 3.09738E+01 * u.u,
        'F': 1.90000E+01 * u.u,
        'He': 4.00260E-00 * u.u,
        'Fe': 5.58470E+01 * u.u,
        'Si': 2.80855E+01 * u.u,
        'Na': 2.29898E+01 * u.u,
        'Mg': 2.43050E+01 * u.u,
        'Cl': 3.54527E+01 * u.u,
        '+': -9.10938356e-28 * u.g,
        '-': +9.10938356e-28 * u.g,
        '': 0.0 * u.u
    }
    elements['D'] = 2.0 * elements['H']

    mass = 0 * u.u
    regex = r'((g|a?)(He|Fe|Si|Na|Mg|Cl|H|C|N|O|S|P|D|F)(\d*)(-|\+?))'
    matches = re.findall(regex, mol)
    check_full = ''
    for match in matches:
        check_full = check_full + match[0]
        mass = mass + elements[match[2]] * match[3] + elements[match[4]]
    if check_full == mol:
        return mass
    else:
        print('Unsupported name of the molecule')
        return 0.0 * u.u
