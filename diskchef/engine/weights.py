import re

from astropy.table import QTable
import astropy.units as u

def mol_weight(mol): #Py-version of a aweight ANDES fortran function (ChemistryAdds.f90)
    #### Initialisation of masses of all elements (from ANDES) + using u quantity from astropy
    elements = QTable()
    elements['H'] = [1.00790 * u.u]
    elements['C'] = [1.20110E+01 * u.u]
    elements['N'] = [1.40067E+01 * u.u]
    elements['O'] = [1.59994E+01 * u.u]
    elements['S'] = [3.20660E+01 * u.u]
    elements['P'] = [3.09738E+01 * u.u]
    elements['D'] = 2.0 * elements['H']
    elements['F'] = [1.90000E+01 * u.u]
    elements['He'] = [4.00260E-00 * u.u]
    elements['Fe'] = [5.58470E+01 * u.u]
    elements['Si'] = [2.80855E+01 * u.u]
    elements['Na'] = [2.29898E+01 * u.u]
    elements['Mg'] = [2.43050E+01 * u.u]
    elements['Cl'] = [3.54527E+01 * u.u]
    ####
    mass = 0 * u.u
    regex = r'((He|Fe|Si|Na|Mg|Cl|H|C|N|O|S|P|D|F)(\d*))' # Regular expression for finding elements in molecule string
    # it finds substring of a type <element><number>
    matches = re.findall(regex, mol) # Finds matches in molecule string according to regex and returns an array of tuples of following type (<match>, <element>, <number of elements>)
    for match in matches: # goes through each match
        mass = mass + elements[match[1]][0] * match[2] # adds found element to the mass of the molecule
    return mass