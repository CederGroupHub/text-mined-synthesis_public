import json
import operator
import os
import warnings
from functools import reduce

__author__ = 'Haoyan Huo'
__maintainer__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'

__all__ = ['NON_VOLATILE_ELEMENTS', 'ELEMENTS', 'PT', 'PT_LIST']

NON_VOLATILE_ELEMENTS = {
    'Li', 'Be',
    'Na', 'Mg', 'Al', 'Si',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db',
    'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts'
}

ELEMENTS = set(
    'H|He|'
    'Li|Be|B|C|N|O|F|Ne|'
    'Na|Mg|Al|Si|P|S|Cl|Ar|'
    'K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|'
    'Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|'
    'Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|'
    'Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg'.split('|')
)


def _patch_pt(pt):
    """
    Fix value issues in the periodic table data downloaded from
    https://github.com/andrejewski/periodic-table/blob/master/data.json

    :param pt: List of element data
    :return: Updated, fixed periodic table data
    """
    keys = reduce(operator.iand, [x.keys() for x in pt])
    missing_keys = set()
    for key in keys:
        if any(key not in y for y in pt):
            warnings.warn('Dropping key: %s because not all element has this key.' % key)
            missing_keys.add(key)

    numeric_keys = [
        "electronegativity", "atomicRadius", "ionRadius", "vanDelWaalsRadius", "ionizationEnergy",
        "electronAffinity", "meltingPoint", "boilingPoint", "density"
    ]

    for el in pt:
        for key in missing_keys:
            del el[key]
        for key in numeric_keys:
            if not isinstance(el[key], int) and not isinstance(el[key], float):
                el[key] = None
    return pt


PT = {}
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pt.json')) as f:
    PT_LIST = _patch_pt(json.load(f))

for element in PT_LIST:
    PT[element['symbol']] = element
