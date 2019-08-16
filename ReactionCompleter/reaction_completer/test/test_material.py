from unittest import TestCase

from reaction_completer import MaterialInformation


class TestMaterialParsing(TestCase):
    def test_volatile_elements(self):
        material = MaterialInformation(
            "BaTiO3", "BaTiO3",
            {
                "amount": "1.0",
                "elements": {"Ba": "1.0", "Ti": "1.0", "O": "3.0"},
            },
            {}
        )
        self.assertSetEqual(material.nv_elements, {"Ba", "Ti"})
        self.assertSetEqual(material.v_elements, {"O"})
