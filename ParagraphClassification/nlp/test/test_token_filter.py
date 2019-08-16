import unittest

from synthesis_project_ceder.nlp.token_filter import FilterClass


class TokenFilterTest(unittest.TestCase):
    def test_unit_regex(self):
        regex = FilterClass().num_unit
        test_cases = [
            '1.3cm', '1.3e8cm', '1.3-99999cm', '1.3e8-879cm', '1.3e2-5.7e8cm',
            '98V', '77Torr', '24cmL-1', '567weeks', '435mmHgcm-2V3A-5'
        ]

        for case in test_cases:
            x = regex.match(case)
            self.assertIsNotNone(x, 'Cannot full match for case: %s!' % case)


if __name__ == '__main__':
    unittest.main()
