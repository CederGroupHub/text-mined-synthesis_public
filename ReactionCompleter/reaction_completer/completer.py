import logging
import re
from functools import reduce
from operator import or_
from tokenize import TokenError

import sympy
from sympy import Matrix, symbols
from sympy.core.numbers import Zero, NegativeOne, One
from sympy.printing.precedence import precedence

from reaction_completer.material import MaterialInformation
from reaction_completer.errors import (
    StupidRecipe, ExpressionPrintException, TooManyPrecursors,
    CannotBalance, FormulaException)
from reaction_completer.periodic_table import ELEMENTS

__author__ = 'Haoyan Huo'
__maintainer__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'


class ReactionCompleter(object):
    FLOAT_ROUND = 3  # 3 decimal places 0.001

    def __init__(self, precursors: [MaterialInformation],
                 target: MaterialInformation,
                 target_min_nv=2):
        """
        A reaction completer that takes a set of precursors and a target,
        then calculates the possible reactions, using sympy for symbolic
        derivation.

        :param precursors: List of precursors.
        :type precursors: list(MaterialInformation)
        :param target: The target material.
        :type target: MaterialInformation
        :param target_min_nv:
        """
        self.precursors = precursors
        self.target = target
        self.target_min_nv = target_min_nv

        self._precursor_candidates = []
        self._decomposition_chemicals = {}
        self._exchange_chemicals = {}
        self._linear_eq = {}

        self._inspect_target()
        self._prepare_precursors()
        self._setup_linear_equation()

    def _inspect_target(self):
        """
        Prepare the target material into a ready-to-use structure.
        """
        if len(self.target.nv_elements) < self.target_min_nv:
            raise StupidRecipe(
                'Target must have more than 1 non volatile elements, got %r: %s' %
                (self.target.nv_elements, self.target.material_formula))

    def _prepare_precursors(self):
        # find the set of precursors

        seen_precursors = set()
        for precursor in self.precursors:
            # Skip precursors that are seen
            if precursor.material_formula in seen_precursors:
                continue

            seen_precursors.add(precursor.material_formula)

            if precursor.all_elements_dict == self.target.all_elements_dict:
                # TODO: we need a smarter comparison
                raise StupidRecipe('Precursor list contains target')

            if len(precursor.all_elements) == 0:
                logging.debug(
                    'Skipping empty precursor %s: %s',
                    precursor.material_formula)
                continue

            if len(precursor.nv_elements - self.target.nv_elements) > 0:
                logging.debug(
                    'Skipping precursor %s because it '
                    'has excessive chemical elements',
                    precursor.material_formula)
                continue

            self._precursor_candidates.append(precursor)
            self._decomposition_chemicals.update(precursor.decompose_chemicals)

        self._exchange_chemicals.update(self.target.exchange_chemicals)

        if len(self._precursor_candidates) == 0:
            raise StupidRecipe('Precursor candidates is empty')

        # check for equality
        precursors_nv_elements = reduce(
            or_, [x.nv_elements for x in self._precursor_candidates])
        missing_elements = self.target.nv_elements - precursors_nv_elements
        if len(missing_elements) > 0:
            raise StupidRecipe(
                'Precursor candidates do not '
                'provide non volatile elements: %r' % missing_elements)

    def _setup_linear_equation(self):
        all_elements = reduce(
            or_,
            [x.all_elements for x in self._precursor_candidates] +
            [self.target.all_elements])
        all_elements = sorted(list(all_elements))

        # Create the symbols that will be used for linear eq.
        chemical_symbols = ''
        for i in range(len(self._precursor_candidates)):
            chemical_symbols += 'p%d, ' % i
        for i in range(len(self._decomposition_chemicals)):
            chemical_symbols += 'r%d, ' % i
        for i in range(len(self._exchange_chemicals)):
            chemical_symbols += 'e%d, ' % i
        chemical_symbols += 't'
        chemical_symbols = symbols(chemical_symbols)

        coefficient_matrix = []
        which_side = []

        def fill_row(material_elements):
            row = [material_elements.get(element, 0) for element in all_elements]
            coefficient_matrix.append(row)

        for precursor in self._precursor_candidates:
            fill_row(precursor.all_elements_dict)
            which_side.append('fl')

        for chemical in sorted(self._decomposition_chemicals):
            fill_row(self._decomposition_chemicals[chemical])
            which_side.append('dr')

        for chemical in sorted(self._exchange_chemicals):
            fill_row(self._exchange_chemicals[chemical])
            which_side.append('dl')

        target_elements = self.target.all_elements_dict
        target_vector = [target_elements.get(i, 0) for i in all_elements]

        coefficient_matrix = Matrix(coefficient_matrix).T

        target_vector = Matrix(target_vector)

        self._linear_eq.update({
            'chemical_symbols': chemical_symbols,
            'coefficient_matrix': coefficient_matrix,
            'target_vector': target_vector,
            'which_side': which_side,
            'all_elements': all_elements,
        })

    _FLOAT_RE = re.compile(r"""
            (?P<sign>[-+])?             # Sign of the float
            (?=\d|\.\d)                 # Make sure there is some number following 
            (?P<int>\d*)                # Integer part
            (\.(?P<frac>\d*))?          # Fraction part
            ([eE](?P<exp>[-+]?\d+))?    # Exponential
            """, re.VERBOSE)

    @staticmethod
    def nicely_print_float(f_s):
        """
        Print a float number nicely.
        :param f_s: string of the float number.
        :return:
        """
        m = ReactionCompleter._FLOAT_RE.match(f_s)
        if not m:
            raise ValueError('This is not a float!')

        integer = m.group('int') or '0'
        fraction = m.group('frac') or ''
        exp = int(m.group('exp') or 0)
        sign = m.group('sign') or ''

        while exp > 0:
            if len(fraction):
                integer += fraction[0]
                fraction = fraction[1:]
            else:
                integer += '0'
            exp -= 1
        fraction = fraction.rstrip('0')
        sign = '-' if sign == '-' else ''
        floating_number = sign + integer + ('.' + fraction if len(fraction) else '')

        return floating_number

    @staticmethod
    def simplify_print(expr: sympy.Expr):
        if isinstance(expr, sympy.Float):
            # Just a float number.
            return ReactionCompleter.nicely_print_float(
                str(expr.round(ReactionCompleter.FLOAT_ROUND)))
        elif isinstance(expr, sympy.Add):
            expression = ''
            for i, ele in enumerate(expr.args):
                ele_str = ReactionCompleter.simplify_print(ele)

                if ele_str == '0':
                    continue

                if ele_str[0] == '-' or expression == '':
                    expression += ele_str
                else:
                    expression += '+' + ele_str

            if expression == '':
                return '0'
            else:
                return expression
        elif isinstance(expr, sympy.Mul):
            coefficient, _ = expr.as_coeff_Mul()
            if coefficient < 0:
                expr = -expr
                sign = '-'
            else:
                sign = ''

            exps = []
            for arg in expr.as_ordered_factors():
                exp = ReactionCompleter.simplify_print(arg)
                if exp == '0':
                    return '0'
                if exp != '1':
                    if precedence(arg) < precedence(expr):
                        exp = '(%s)' % exp
                    exps.append(exp)

            expression = sign + '*'.join(exps)
            return expression
        elif isinstance(expr, NegativeOne):
            return '-1'
        elif isinstance(expr, One):
            return '1'
        elif isinstance(expr, Zero):
            return '0'
        elif isinstance(expr, sympy.Symbol):
            return expr.name
        elif isinstance(expr, sympy.Integer):
            return str(expr.p)
        else:
            raise ExpressionPrintException(
                'Do not know how to print %r: %r' % (type(expr), expr))

    def _render_reaction(self, solution: tuple):
        balanced = {
            'left': {},
            'right': {self.target.material_formula: '1'}
        }

        solution = list(solution)
        which_side = self._linear_eq['which_side']

        precursor_solutions = solution[:len(self._precursor_candidates)]
        precursor_side = which_side[:len(self._precursor_candidates)]
        del solution[:len(self._precursor_candidates)]
        del which_side[:len(self._precursor_candidates)]

        decomposition_solutions = solution[:len(self._decomposition_chemicals)]
        decomposition_side = which_side[:len(self._decomposition_chemicals)]
        del solution[:len(self._decomposition_chemicals)]
        del which_side[:len(self._decomposition_chemicals)]

        exchange_solutions = solution.copy()
        exchange_side = which_side.copy()

        def decide_side_value(s, val):
            if s[0] == 'f':
                if s[1] == 'l':
                    return 'left', val
                elif s[1] == 'r':
                    return 'right', -val
            elif s[0] == 'd':
                if not isinstance(val, sympy.Float):
                    value_zero = val.evalf(
                        subs={x: 0.001 for x in val.free_symbols})
                    value_negative = float(value_zero) < 0
                else:
                    value_negative = float(val) < 0

                if s[1] == 'l':
                    return ('left', val) if not value_negative else ('right', -val)
                elif s[1] == 'r':
                    return ('right', -val) if value_negative else ('left', val)

        for precursor, amount, side in zip(
                self._precursor_candidates, precursor_solutions, precursor_side):
            material_formula = precursor.material_formula
            side, value = decide_side_value(side, amount)

            value_s = self.simplify_print(value)
            if value_s != '0':
                balanced[side][material_formula] = value_s

        for chemical, amount, side in zip(
                sorted(self._decomposition_chemicals), decomposition_solutions, decomposition_side):
            side, value = decide_side_value(side, amount)

            value_s = self.simplify_print(value)
            if value_s != '0':
                balanced[side][chemical] = value_s

        for chemical, amount, side in zip(
                sorted(self._exchange_chemicals), exchange_solutions, exchange_side):
            side, value = decide_side_value(side, amount)

            value_s = self.simplify_print(value)
            if value_s != '0':
                balanced[side][chemical] = value_s

        return balanced

    def compute_reactions(self):
        try:
            a = self._linear_eq['coefficient_matrix']
            b = self._linear_eq['target_vector']
            solution, params = a.gauss_jordan_solve(b)

            if len(params) > 0:
                raise TooManyPrecursors(
                    'Too many precursors to balance %r ==> %r' % (
                        [x.material_formula for x in self.precursors],
                        self.target.material_formula))

            solution = solution.T[:1, :]
        except ValueError:
            raise CannotBalance('Too few precursors to balance')

        return self._render_reaction(solution)


ions_regex = re.compile('|'.join(sorted(ELEMENTS, key=lambda x: (-len(x), x))))
OMIT_IONS = {'O', 'H', 'N'}


def find_ions(string):
    return ions_regex.findall(string)


def render_reaction(precursors, target, reaction, element_substitution=None):
    element_substitution = element_substitution or {}

    left_strings = []
    # Alphabetic ordering for left part
    for material, amount in sorted(reaction['left'].items()):
        left_strings.append('%s %s' % (amount, material))
    right_strings = []
    # Alphabetic ordering for right part, except
    # for target material it's the first element
    for material, amount in sorted(
            reaction['right'].items(),
            key=lambda x: (x[0] != target['material_formula'], x[0], x[1])):
        right_strings.append('%s %s' % (amount, material))

    reaction_string = ' + '.join(left_strings) + ' == ' + ' + '.join(right_strings)

    if len(element_substitution) > 0:
        subs_string = ', '.join(['%s = %s' % (sub, subs)
                                 for (sub, subs) in element_substitution.items()])
        reaction_string += '; ' + subs_string

    # Populate additives
    if target['additives']:
        additive_ions = set(find_ions(' '.join(target['additives'])))
        added_anions = set()
        additive_precursors = []

        for precursor in precursors:
            compositions = []
            for comp in precursor['composition']:
                compositions.append({
                    'amount': comp['amount'],
                    'elements': comp['elements'],
                })

            try:
                mat_info = MaterialInformation(
                    precursor['material_string'],
                    precursor['material_formula'],
                    compositions)

                if mat_info.all_elements and \
                        any(x in additive_ions for x in mat_info.all_elements):
                    added_anions.update(mat_info.all_elements)
                    additive_precursors.append(precursor['material_formula'])
            except FormulaException:
                pass

        reaction_string += ' ; target %s with additives %s via %s' % (
            target['material_formula'],
            ', '.join(target['additives']),
            ', '.join(sorted(additive_precursors))
        )

    return reaction_string


def balance_recipe(precursors, targets, sentences=None):
    """
    Balance a recipe extracted using synthesis project pipeline.

    If argument "sentences" is a list of sentences in the paragraph, when
    too many materials are given and we are unable to determine the right
    set of precursors, this function will try to gather precursors in the
    same sentence, and use the subset to balance the reaction.

    Example usage:
    >>>
    >>> precursors = [
    >>>     {
    >>>         "material_formula": "SrCO3",
    >>>         "material_string": "SrCO3",
    >>>         "composition": [
    >>>             {
    >>>                 "formula": "SrCO3",
    >>>                 "elements": {"Sr": "1.0", "C": "1.0", "O": "3.0"},
    >>>                 "amount": "1.0"
    >>>             }
    >>>         ],
    >>>     },
    >>>     {
    >>>         "material_formula": "Al2O3",
    >>>         "material_string": "Al2O3",
    >>>         "composition": [
    >>>             {
    >>>                 "formula": "Al2O3",
    >>>                 "elements": {"Al": "2.0", "O": "3.0"},
    >>>                 "amount": "1.0"
    >>>             }
    >>>         ],
    >>>     },
    >>>     {
    >>>         "material_formula": "MnO",
    >>>         "material_string": "MnO",
    >>>         "composition": [
    >>>             {
    >>>                 "formula": "MnO",
    >>>                 "elements": {"Mn": "1.0", "O": "1.0"},
    >>>                 "amount": "1.0"
    >>>             }
    >>>         ],
    >>>     },
    >>>     {
    >>>         "material_formula": "Fe2O3",
    >>>         "material_string": "Fe2O3",
    >>>         "composition": [
    >>>             {
    >>>                 "formula": "Fe2O3",
    >>>                 "elements": {"Fe": "2.0", "O": "3.0"},
    >>>                 "amount": "1.0"
    >>>             }
    >>>         ],
    >>>     },
    >>>     {
    >>>         "material_formula": "ZrO2",
    >>>         "material_string": "ZrO2",
    >>>         "composition": [
    >>>             {
    >>>                 "formula": "ZrO2",
    >>>                 "elements": {"Zr": "1.0", "O": "2.0"},
    >>>                 "amount": "1.0"
    >>>             }
    >>>         ]
    >>>     },
    >>>     {
    >>>         "material_formula": "H2O",
    >>>         "material_string": "H2O",
    >>>         "composition": [
    >>>             {
    >>>                 "formula": "H2O",
    >>>                 "elements": {"O": "1.0", "H": "2.0"},
    >>>                 "amount": "1.0"
    >>>             }
    >>>         ]
    >>>     },
    >>> ]
    >>> targets = [
    >>>     {
    >>>         "material_formula": "Sr6(A2O4)6",
    >>>         "material_string": "Sr6(A2O4)6",
    >>>         "composition": [
    >>>             {
    >>>                 "formula": "Sr6(Fe2O4)6",
    >>>                 "elements": {"A": "12.0", "O": "24.0", "Sr": "6.0"},
    >>>                 "amount": "1.0"
    >>>             }
    >>>         ],
    >>>         "elements_vars": {
    >>>             "A": ["Fe", "Al"]
    >>>         },
    >>>         "additives": ["Mn2+"]
    >>>     },
    >>> ]
    >>> text = [
    >>>     "SrCO3, Al2O3, MnO and Fe2O3 are used to synthesize Mn2+doped-Sr6(A2O4)6, A=Fe, Al.",
    >>>     "Milling media is ZrO2",
    >>>     "There is some H2O found in the final product."
    >>> ]
    >>>
    >>> reactions = balance_recipe(precursors, targets, text)
    >>> print('Found', len(reactions), 'reactions')
    >>> for reaction in reactions:
    >>>     print(reaction)

    :param precursors: List of precursors
    :param targets: List of targets
    :param sentences: List of sentences
    :return:
    """
    sentences = sentences or []

    # Generate all possible combinations of (target, element_substitution)
    targets_to_balance = []
    for target in targets:
        has_element_vars = len(target['elements_vars']) != 0
        target_elements = reduce(or_, [set(x['elements'].keys()) for x in target['composition']])
        element_vars_used = set(target['elements_vars'].keys()) & target_elements
        if has_element_vars and len(element_vars_used) > 0:
            for sub in element_vars_used:
                for subs in target['elements_vars'][sub]:
                    targets_to_balance.append((target, {sub: subs}))
        else:
            targets_to_balance.append((target, None))

    def material_dict_to_object(material_dict, sub_dict=None):
        compositions = []
        for comp in material_dict['composition']:
            compositions.append({
                'amount': comp['amount'],
                'elements': comp['elements'],
            })
        return MaterialInformation(
            material_dict['material_string'],
            material_dict['material_formula'],
            compositions, sub_dict)

    precursor_objects = []
    for precursor in precursors:
        try:
            precursor_objects.append(material_dict_to_object(
                precursor))
        except FormulaException:
            continue

    solutions = []
    for target, substitution in targets_to_balance:
        try:
            target_object = material_dict_to_object(target, substitution)
        except FormulaException:
            continue

        try:
            completer = ReactionCompleter(precursor_objects, target_object)
            solution = completer.compute_reactions()
            solutions.append((
                target_object.material_formula,
                solution,
                substitution,
                render_reaction(precursors, target, solution, substitution)
            ))
        except TooManyPrecursors as e:
            # try eliminate the precursors not in the same sentence.
            precursor_candidates = []

            # Find the list of precursors that are in the same sentence
            for sentence in sentences:
                candidates = []
                for precursor in precursor_objects:
                    # TODO: using precursor.material_string will generate less reactions...
                    if precursor.material_formula in sentence:
                        candidates.append(precursor)
                if candidates:
                    precursor_candidates.append(candidates)

            if not precursor_candidates:
                logging.debug('No possible precursor subsets for target: %s, precursors: %r: %r',
                              target_object.material_formula,
                              [x.material_formula for x in precursor_objects], e)
            else:
                success = False
                # Iterate over all candidate precursors, and find the first success
                # reaction that can be completed.
                for candidates in precursor_candidates:
                    try:
                        completer = ReactionCompleter(candidates, target_object)
                        solution = completer.compute_reactions()
                        solutions.append((
                            target_object.material_formula,
                            solution,
                            substitution,
                            render_reaction(precursors, target, solution, substitution)
                        ))
                        success = True
                        break
                    except (CannotBalance, TokenError) as e_subset:
                        logging.debug(
                            'Failed trying precursor subset for target: %s, '
                            'precursors: %r: %r',
                            target_object.material_formula,
                            [x.material_formula for x in candidates], e_subset)

                if not success:
                    logging.debug('Cannot find a subset of precursors for '
                                  'target: %s, precursors: %r: %r',
                                  target_object.material_formula,
                                  [x.material_formula for x in precursor_objects], e)

        except (CannotBalance, TokenError) as e:
            logging.debug('Cannot balance reaction for target: %s, precursors: %r: %r',
                          target_object.material_formula,
                          [x.material_formula for x in precursor_objects], e)
        except Exception as e:
            logging.warning('Unexpected error for target: %s, precursors: %r: %r',
                            target_object.material_formula,
                            [x.material_formula for x in precursor_objects], e)
    return solutions
