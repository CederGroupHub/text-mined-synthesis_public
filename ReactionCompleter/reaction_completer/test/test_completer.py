from unittest import TestCase

from reaction_completer import balance_recipe


class TestSimple(TestCase):
    def test_basic_completer(self):
        """
        BaCO3 + TiO2 ==== BaTiO3 + CO2
        """
        precursors = [
            {
                "material_formula": "BaCO3",
                "material_string": "BaCO3",
                "composition": [
                    {
                        "formula": "BaCO3",
                        "elements": {"Ba": "1.0", "C": "1.0", "O": "3.0"},
                        "amount": "1.0"
                    }
                ],
            },
            {
                "material_formula": "TiO2",
                "material_string": "TiO2",
                "composition": [
                    {
                        "formula": "TiO2",
                        "elements": {"Ti": "1.0", "O": "2.0"},
                        "amount": "1.0"
                    }
                ],
            },
        ]
        targets = [
            {
                "material_formula": "BaTiO3",
                "material_string": "BaTiO3",
                "composition": [
                    {
                        "formula": "BaTiO3",
                        "elements": {"Ba": "1.0", "Ti": "1.0", "O": "3.0"},
                        "amount": "1.0"
                    }
                ],
                "elements_vars": {},
                "additives": []
            },
        ]

        reactions = balance_recipe(precursors, targets)
        self.assertListEqual(reactions, [(
            'BaTiO3',
            {
                'left': {'BaCO3': '1', 'TiO2': '1'},
                'right': {'BaTiO3': '1', 'CO2': '1'}
            },
            None,
            '1 BaCO3 + 1 TiO2 == 1 BaTiO3 + 1 CO2'
        )])


class TestElementSubstitution(TestCase):
    def test_simple(self):
        """
        SrCO3 + Al2O3 + Fe2O3 ==== Sr6(A2O4)6, A=Al, Fe
        """
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
        self.assertListEqual(reactions, [
            (
                'Sr6(A2O4)6', {
                    'left': {'SrCO3': '6', 'Fe2O3': '6'},
                    'right': {'Sr6(A2O4)6': '1', 'CO2': '6'}
                },
                {'A': 'Fe'},
                '6 Fe2O3 + 6 SrCO3 == 1 Sr6(A2O4)6 + 6 CO2; A = Fe ; target Sr6(A2O4)6 with additives Mn2+ via MnO'
            ),
            (
                'Sr6(A2O4)6', {
                    'left': {'SrCO3': '6', 'Al2O3': '6'},
                    'right': {'Sr6(A2O4)6': '1', 'CO2': '6'}
                },
                {'A': 'Al'},
                '6 Al2O3 + 6 SrCO3 == 1 Sr6(A2O4)6 + 6 CO2; A = Al ; target Sr6(A2O4)6 with additives Mn2+ via MnO'
            )
        ])
