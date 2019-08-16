# ReactionCompleter

This module computes valid chemical reactions using list of precursor 
and target materials. It is capable of using some simple heuristics to
complete or guess missing or excessive materials in the given lists of
materials.

[![Build Status](https://semaphoreci.com/api/v1/projects/648bbb02-9eb5-4cce-9642-16eb7108f2fa/2831982/badge.svg)](https://semaphoreci.com/cedergrouphub/reactioncompleter)

# Usage

The primary function to be used is 
`balance_recipe(precursors, targets, synthesis_paragraph)`. An example
is as follows:

```python
from reaction_completer import balance_recipe


precursors = [
    {
        "material_formula": "SrCO3",
        "material_string": "SrCO3",
        "composition": [
            {
                "formula": "SrCO3",
                "elements": {"Sr": "1.0", "C": "1.0", "O": "3.0"},
                "amount": "1.0"
            }
        ],
    },
    {
        "material_formula": "Al2O3",
        "material_string": "Al2O3",
        "composition": [
            {
                "formula": "Al2O3",
                "elements": {"Al": "2.0", "O": "3.0"},
                "amount": "1.0"
            }
        ],
    },
    {
        "material_formula": "MnO",
        "material_string": "MnO",
        "composition": [
            {
                "formula": "MnO",
                "elements": {"Mn": "1.0", "O": "1.0"},
                "amount": "1.0"
            }
        ],
    },
    {
        "material_formula": "Fe2O3",
        "material_string": "Fe2O3",
        "composition": [
            {
                "formula": "Fe2O3",
                "elements": {"Fe": "2.0", "O": "3.0"},
                "amount": "1.0"
            }
        ],
    },
    {
        "material_formula": "ZrO2",
        "material_string": "ZrO2",
        "composition": [
            {
                "formula": "ZrO2",
                "elements": {"Zr": "1.0", "O": "2.0"},
                "amount": "1.0"
            }
        ]
    },
    {
        "material_formula": "H2O",
        "material_string": "H2O",
        "composition": [
            {
                "formula": "H2O",
                "elements": {"O": "1.0", "H": "2.0"},
                "amount": "1.0"
            }
        ]
    },
]
targets = [
    {
        "material_formula": "Sr6(A2O4)6",
        "material_string": "Sr6(A2O4)6",
        "composition": [
            {
                "formula": "Sr6(Fe2O4)6",
                "elements": {"A": "12.0", "O": "24.0", "Sr": "6.0"},
                "amount": "1.0"
            }
        ],
        "elements_vars": {
            "A": ["Fe", "Al"]
        },
        "additives": ["Mn2+"]
    },
]
text = [
    "SrCO3, Al2O3, MnO and Fe2O3 are used to synthesize Mn2+doped-Sr6(A2O4)6, A=Fe, Al.",
    "Milling media is ZrO2",
    "There is some H2O found in the final product."
]

reactions = balance_recipe(precursors, targets, text)
print('Found', len(reactions), 'reactions')
for reaction in reactions:
    print(reaction)

# Output:
# [(
#     'Sr6(A2O4)6', {
#         'left': {'SrCO3': '6', 'Fe2O3': '6'},
#         'right': {'Sr6(A2O4)6': '1', 'CO2': '6'}
#     },
#     {'A': 'Fe'},
#     '6 SrCO3 + 6 Fe2O3 == 1 Sr6(A2O4)6 + 6 CO2; A = Fe ; target Sr6(A2O4)6 with additives Mn2+ via MnO'
# ),
# (
#     'Sr6(A2O4)6', {
#         'left': {'SrCO3': '6', 'Al2O3': '6'},
#         'right': {'Sr6(A2O4)6': '1', 'CO2': '6'}
#     },
#     {'A': 'Al'},
#     '6 SrCO3 + 6 Al2O3 == 1 Sr6(A2O4)6 + 6 CO2; A = Al ; target Sr6(A2O4)6 with additives Mn2+ via MnO'
# )]
``` 
