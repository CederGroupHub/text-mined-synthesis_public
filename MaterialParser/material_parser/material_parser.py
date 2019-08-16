# coding=utf-8

__author__ = "Olga Kononova"
__maintainer__ = "Olga Kononova"
__email__ = "0lgaGkononova@yandex.ru"
__version__ = "5.6.1"

import regex as re
import collections
import sympy as smp
from sympy.abc import _clash
import pubchempy as pcp
import os
import json
from pprint import pprint


# noinspection PyBroadException
class MaterialParser:
    def __init__(self, verbose=False, pubchem_lookup=False, fails_log=False, dictionary_update=False):
        print('MaterialParser version 5.6.1')

        self.__list_of_elements_1 = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'K', 'V', 'Y', 'I', 'W', 'U']
        self.__list_of_elements_2 = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr',
                                     'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                                     'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe',
                                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                                     'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                                     'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
                                     'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
                                     'Fl', 'Lv']
        self.__greek_letters = [chr(i) for i in range(945, 970)]

        self.__filename = os.path.dirname(os.path.realpath(__file__))

        self.__ions = json.loads(open(os.path.join(self.__filename, 'rsc/ions_dictionary.json')).read())
        self.__anions = {ion['c_name']: {'valency': ion['valency'], 'e_name': ion['e_name'], 'n_atoms': ion['n_atoms']}
                         for ion in self.__ions['anions']}
        self.__cations = {ion['c_name']: {'valency': ion['valency'], 'e_name': ion['e_name'], 'n_atoms': ion['n_atoms']}
                          for ion in self.__ions['cations']}
        self.__chemicals = self.__ions['chemicals'] + \
                           [ion['c_name'] for ion in self.__ions['cations']] + \
                           [ion['c_name'] for ion in self.__ions['anions']]
        self.__element2name = self.__ions['elements']

        self.__prefixes2num = {'': 1, 'mono': 1, 'di': 2, 'tri': 3, 'tetra': 4, 'pent': 5, 'penta': 5, 'hexa': 6,
                               'hepta': 7, 'octa': 8, 'nano': 9, 'ennea': 9, 'nona': 9, 'deca': 10, 'undeca': 11,
                               'dodeca': 12}
        self.__neg_prefixes = ['an', 'de', 'non']

        self.__rome2num = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}

        self.__diatomic_molecules = {'O2': {'O': '2.0'},
                                     'N2': {'N': '2.0'},
                                     'H2': {'H': '2.0'}}

        self.__pubchem_dictionary = json.loads(open(os.path.join(self.__filename, 'rsc/pubchem_dict.json')).read())
        self.__abbreviations = json.loads(open(os.path.join(self.__filename, 'rsc/abbreviations.json')).read())

        self.__fails_log = fails_log
        if fails_log:
            self.__pubchem_file = open(os.path.join(self.__filename, 'pubchem_log'), 'w')
            self.__pubchem_file.close()

        self.__pubchem = pubchem_lookup
        if pubchem_lookup:
            print ('Pubchem lookup is on! Will search for unknown materials name in PubChem DB.')

        self.__dictionary_update = dictionary_update
        if dictionary_update:
            self.__dictionary_file =  open(os.path.join(self.__filename, 'dictionary_update'), 'w')
            self.__dictionary_file.close()

        self.__verbose = verbose

    ###################################################################################################################
    # Parsing material name
    ###################################################################################################################

    def parse_material(self, material_string_):
        """
        Main method to parse material string into chemical structure and convert chemical name into chemical formula
        :param material_string_: < str> material name/formula
        :return: dict(material_string: <str> initial material string,
                     material_name: <str> chemical name of material found in the string
                     material_formula: <str> chemical formula of material
                     additives: <list> list of dopped materials/elements appeared in material string
                     phase: <str> material phase appeared in material string
                     hydrate: <float> if material is hydrate fraction of H2O
                     is_mixture: <bool> material is mixture/composite/alloy/solid solution
                     is_abbreviation: <bool> material is similar to abbreviation
                     amount_vars: {amount variable: <list> values}
                     elements_vars: {elements variable: <list> values}
                     composition: <list> of dict(
                                formula: <str> compound formula
                                amount: <str> fraction of compound in mixture
                                elements: {element: amount}
                     )
        """

        material_string = self.cleanup_name(material_string_)

        if self.__verbose:
            print ('After cleaning up string:')
            print (material_string_, '-->', material_string)

        if material_string in self.__list_of_elements_1+self.__list_of_elements_2:
            return dict(
                material_string=material_string_,
                material_name='',
                material_formula=material_string,
                additives=[],
                phase='',
                is_abbreviation_like=False,
                oxygen_deficiency = '',
                amounts_vars={},
                elements_vars={},
                composition=[dict(
                    formula = material_string,
                    amount = '1.0',
                    elements = {material_string: '1.0'}
                )]
            )

        additives, material_string = self.get_additives(material_string)
        if self.__verbose:
            print ('After additives extraction:')
            print (material_string, 'WITH', additives)

        material_string = material_string.lstrip(') -').rstrip('( ,.:;-±/+')
        material_name, material_formula, material_structure = self.split_material_name(material_string)

        if self.__verbose:
            print ('After material name parsing:')
            print (material_string, '-->', material_name, 'AND', material_formula)

        # if material_string contains chemical formula
        if material_structure['elements'] != {}:

            output_structure = dict(
                material_string=material_string_,
                material_name=material_name,
                material_formula="",
                additives=additives,
                phase=material_structure['phase'],
                is_abbreviation_like=False,
                oxygen_deficiency=material_structure['oxygen_deficiency'],
                amounts_vars=material_structure['amounts_vars'],
                elements_vars=material_structure['elements_vars'],
                composition = [dict(
                    formula = material_structure['formula'],
                    amount = '1.0',
                    elements = material_structure['elements']
                )]
            )
            if material_structure['hydrate'] != '':
                output_structure['composition'].append(dict(
                    formula='H2O',
                    amount=material_structure['hydrate'],
                    elements={'H': '2.0', 'O': '1.0'}
                ))

            output_structure['material_formula'] = self.__combine_formula(output_structure['composition'])
            return output_structure
        else:
            material_formula = ''

        # if material_string is chemical name reconstructing its formula
        if all(part[1:].islower() for part in re.sub('[IV\(\)]', '', material_name).split(' ') if part[1:] != ''):
            material_formula = self.reconstruct_formula(material_name)

            if material_formula != '' and self.__dictionary_update:
                with open(os.path.join(self.__filename, 'dictionary_update'), 'a') as f_dictionary:
                    f_dictionary.write(material_name + ' - ' + material_formula + '\n')

            if material_formula == '':
                for m in [material_name,
                          material_name.lower(),
                          material_name.replace('-', ' '),
                          material_name.replace('-', ' ').lower()]:
                    if m in self.__pubchem_dictionary and material_formula == '':
                        material_formula = self.__pubchem_dictionary[m]


        if self.__verbose:
            print ('After formula reconstruction:')
            print (material_string, '-->', material_name, 'AND', material_formula)


        material_formula = material_string if material_formula == '' else material_formula

        if material_formula in self.__diatomic_molecules:
            return dict(
                material_string=material_string_,
                material_name=material_string,
                material_formula=material_formula,
                additives=[],
                phase='',
                is_abbreviation_like=False,
                oxygen_deficiency = '',
                amounts_vars={},
                elements_vars={},
                composition=[dict(
                    formula=material_formula,
                    amount='1.0',
                    elements=self.__diatomic_molecules[material_formula]
                )]
            )

        # noinspection PyBroadException
        try:
            material_parts = self.split_material(material_formula)
        except:
            material_parts = [(material_formula, '1.0')]

        if self.__verbose:
            print ('After splitting:')
            print (material_string, '-->', material_name, material_parts)

        output_structure = dict(
            material_string=material_string_,
            material_name=material_name,
            material_formula=material_formula.replace(' ', ''),
            phase='',
            additives=additives,
            is_abbreviation_like=False,
            oxygen_deficiency = '',
            amounts_vars={},
            elements_vars={},
            composition=[]
        )

        hydrate = ''
        oxygen_deficiency = ''
        for compound, amount in material_parts:
            try:
                compound = self.__check_parentheses(compound)
                if compound in self.__abbreviations:
                    compound = self.__abbreviations[compound]
                structure = self.get_structure_by_formula(compound)
                output_structure['phase'] = structure['phase']
                output_structure['amounts_vars'].update(structure['amounts_vars'])
                output_structure['elements_vars'].update(structure['elements_vars'])
                if compound == 'H2O':
                    hydrate = amount
                elif structure['elements'] != {}:
                    output_structure['composition'].append(dict(
                        formula = structure['formula'],
                        amount = amount,
                        elements = structure['elements']
                    ))
                if structure['hydrate'] != '':
                    hydrate = structure['hydrate']

                if structure['oxygen_deficiency'] != '':
                    oxygen_deficiency = structure['oxygen_deficiency']
            except:
                output_structure['composition'].append(dict(
                    formula = compound,
                    amount = amount,
                    elements = {}
                ))

        if hydrate != '':
            output_structure['composition'].append(dict(
                    formula = 'H2O',
                    amount = hydrate,
                    elements = {'H': '2.0', 'O': '1.0'}
                ))

        output_structure['oxygen_deficiency'] = oxygen_deficiency

        # substituting additive into composition if it makes fractions to sum-up to integer
        output_structure['additives'] = [elem.strip(' ') for additive in output_structure['additives'] for elem in re.split('[\s,]', additive) if elem != '']
        additive = additives[0].strip(' ') if len(additives) == 1 else ''
        #print('-->', additive)
        if additive != '' and all(c['elements'] != {} for c in output_structure['composition']):
            formula, composition = self.__substitute_additive(additive, material_formula, output_structure['composition'])
            if formula != material_formula:
                output_structure['additives'] = []
            output_structure['material_formula'] = formula
            output_structure['composition'] = composition

        # negative stoichiometry
        try:
            if any(float(s) < 0.0 for compound in output_structure['composition'] for e, s in compound['elements'].items()):
                output_structure['composition'] = []
        except:
            pass

        for compound in output_structure['composition']:
            if len(re.findall('[b-mo-w]+', compound['amount'])) > 0:
                compound['amount'] = "1.0"


        if output_structure['composition'] == [] and self.__fails_log:
            with open(os.path.join(self.__filename, 'fails_log'), 'a') as f_log:
                f_log.write(material_name + '\n')

        # checking abbreviation
        output_structure['is_abbreviation_like'] = self.__is_abbreviation_like(output_structure)
            #len([el for el in output_structure['elements_vars'].keys() if len(el) == 1 and el.isupper()]) > 1

        output_structure['material_formula'] = self.__combine_formula(output_structure['composition'])

        return output_structure

    def get_structure_by_formula(self, formula):
        """
        Parsing chemical formula in composition
        :param formula: <str> chemical formula
        :return: dict(formula: <str> formula string corresponding to obtained composition
                     composition: <dict> element: fraction
                     fraction_vars: <dict> elements fraction variables: <list> values
                     elements_vars: <dict> elements variables: <list> values
                     hydrate: <str> if material is hydrate fraction of H2O
                     phase: <str> material phase appeared in formula
                    )
        """

        formula = formula.replace(' ', '')
        formula = formula.replace('−', '-')
        formula = formula.replace('[', '(')
        formula = formula.replace(']', ')')
        formula = formula.replace('{', '(')
        formula = formula.replace('}', ')')

        formula = formula.strip(' ')

        #print ('->', formula)

        # is there any phase specified
        phase = ''
        if formula[0].islower():
            for m in re.finditer('([' + ''.join(self.__greek_letters) + ']*)-{0,1}(.*)', formula):
                if m.group(2) != '':
                    phase = m.group(1)
                    formula = m.group(2)

        # oxygen deficiency
        oxygen_deficiency = ''
        oxygen_deficiency_sym = ''
        r = ''.join([s for s in self.__greek_letters])
        r = 'O[0-9]*([-+±∓]{1})[a-z' + r + ']{1}[0-9]*$'
        for m in re.finditer(r, formula.rstrip(')')):
            end = formula[m.start():m.end()]
            splt = re.split('[-+±]', end)
            oxygen_deficiency_sym = splt[-1]
            oxygen_deficiency = m.group(1)
            formula = formula[:m.start()]+formula[m.start():].replace(end, splt[0])


        # checking for hydrate
        hydrate = ''
        if 'H2O' in formula and any(c in formula for c in ['·', '•', '-', '×', '⋅']):
            formula = formula[1:-1] if formula[0] == '(' and formula[-1] == ')' else formula
            hyd_i = formula.find('H2O') - 1
            hydrate_num = []
            while hyd_i > 0 and formula[hyd_i] not in ['·', '•', '-', '×', '⋅']:
                hydrate_num.append(formula[hyd_i])
                hyd_i -= 1
            hydrate = ''.join([c for c in reversed(hydrate_num)])
            if hydrate == '':
                hydrate = '1.0'
            if hyd_i > 0:
                formula = formula[:hyd_i]


        elements_variables = collections.defaultdict(str)
        stoichiometry_variables = collections.defaultdict(str)

        # convert fractions a(b-/+x) into a*b-/+a*x
        for m in re.findall('([0-9\.]+)(\([0-9\.a-z]+[-+]+[0-9\.a-z]+\))', formula):
            expr = str(smp.simplify(m[0] + '*' + m[1]))
            if expr[0] == '-':
                s_expr = re.split('\+', expr)
                expr = s_expr[1] + s_expr[0]
            expr = expr.replace(' ', '')
            formula = formula.replace(m[0] + m[1], expr, 1)


        # check for any weird syntax
        r = "\(([^\(\)]+)\)\s*([-*\.\da-z\+/]*)"
        for m in re.finditer(r, formula):
            if ',' in m.group(1):
                elements_variables['M'] = re.split(',', m.group(1))
                formula = formula.replace('(' + m.group(1) + ')' + m.group(2), 'M' + m.group(2), 1)
            if not m.group(1).isupper() and m.group(2) == '':
                formula = formula.replace('(' + m.group(1) + ')', m.group(1), 1)


        #print ('->', formula)
        composition = self.__parse_formula(formula)
        #print ('Final composition:')
        #pprint(composition)

        if re.findall('[a-z]{4,}', formula) != [] and composition != {}:
            #composition = collections.defaultdict(str)
            composition = collections.OrderedDict()
            #print ('->', formula)
            #print (composition)

        # looking for variables in elements and stoichiometry
        for el, amt in composition.items():
            if el not in self.__list_of_elements_1 + \
                    self.__list_of_elements_2 + \
                    list(elements_variables.keys()) + ['□']:
                elements_variables[el] = []
            for var in re.findall('[a-z' + ''.join(self.__greek_letters) + ']', amt):
                stoichiometry_variables[var] = {}

        rename_variables = [('R', 'E'), ('A', 'E'), ('T', 'M')]
        for v1, v2 in rename_variables:
            if v1 in elements_variables and v2 in elements_variables and v1+v2 in formula:
                elements_variables[v1+v2] = []
                del elements_variables[v2]
                del elements_variables[v1]
                composition[v1+v2] = composition[v2]
                del composition[v1]
                del composition[v2]

        if 'M' in elements_variables and 'e' in stoichiometry_variables:
            elements_variables['Me'] = []
            del elements_variables['M']
            del stoichiometry_variables['e']
            c = composition['M'][1:]
            composition['Me'] = c if c != '' else '1.0'
            del composition['M']

        if oxygen_deficiency != '' and oxygen_deficiency_sym in stoichiometry_variables:
            oxygen_deficiency = ''

        formula_structure = dict(elements=composition, #{e: s for e, s in composition.items()},
                                 amounts_vars={x: v for x, v in stoichiometry_variables.items()},
                                 elements_vars={e: v for e, v in elements_variables.items()},
                                 hydrate=hydrate,
                                 phase=phase,
                                 formula=formula,
                                 oxygen_deficiency=oxygen_deficiency)

        return formula_structure

    def __parse_formula(self, init_formula):

        #formula_dict = collections.defaultdict(str)
        formula_dict = collections.OrderedDict()

        formula_dict = self.__parse_parentheses(init_formula, "1", formula_dict)

        """
        refinement of non-variable values
        """
        incorrect = []
        for el, amt in formula_dict.items():
            formula_dict[el] = self.__simplify(amt)
            if any(len(c) > 1 for c in re.findall('[A-Za-z]+', formula_dict[el])):
                incorrect.append(el)

        for el in incorrect:
            del formula_dict[el]

        return formula_dict

    def __parse_parentheses(self, init_formula, init_factor, curr_dict):
        #print ('Input:', init_formula, init_factor)
        #print ('Current dictionary:', curr_dict)
        r = "\(((?>[^\(\)]+|(?R))*)\)\s*([-*\.\da-z\+/]*)"

        #print ('-->', init_formula, init_factor)

        for m in re.finditer(r, init_formula):
            factor = "1"
            if m.group(2) != "":
                factor = m.group(2)

            #print ('-->', m.group(1), factor, curr_dict)

            factor = self.__simplify('(' + str(init_factor) + ')*(' + str(factor) + ')')

            unit_sym_dict = self.__parse_parentheses(m.group(1), factor, curr_dict)

            init_formula = init_formula.replace(m.group(0), '')

        unit_sym_dict = self.__get_sym_dict(init_formula, init_factor)
        #print ('To update:', unit_sym_dict)
        for el, amt in unit_sym_dict.items():
            if el in curr_dict:
                if len(curr_dict[el]) == 0:
                    curr_dict[el] = amt
                else:
                    curr_dict[el] = '(' + str(curr_dict[el]) + ')' + '+' + '(' + str(amt) + ')'
            else:
                curr_dict[el] = amt

        return curr_dict

    def __get_sym_dict(self, f, factor):
        #print ('-->', f, factor)
        sym_dict = collections.OrderedDict()
        r = "([A-Z□]{1}[a-z]{0,1})\s*([-\*\.\da-z" + ''.join(self.__greek_letters) + "\+\/]*)"

        def get_code_value(code, iterator):
            code_mapping = {'01': (iterator.group(1), iterator.group(2)),
                            '11': (iterator.group(1), iterator.group(2)),
                            '10': (iterator.group(1)[0], iterator.group(1)[1:] + iterator.group(2)),
                            '00': (iterator.group(1)[0], iterator.group(1)[1:] + iterator.group(2))}
            return code_mapping[code]

        el = ""
        amt = ""
        for m in re.finditer(r, f):
            """
            checking for correct elements names
            """
            el_bin = "{0}{1}".format(str(int(m.group(1)[0] in self.__list_of_elements_1 + ['M', '□'])), str(
                int(m.group(1) in self.__list_of_elements_1 + self.__list_of_elements_2 + ['Ln', 'M', '□'])))
            el, amt = get_code_value(el_bin, m)
            # if el_bin in ['01', '11']:
            #     el = m.group(1)
            #     amt = m.group(2)
            # if el_bin in ['10', '00']:
            #     el = m.group(1)[0]
            #     amt = m.group(1)[1:] + m.group(2)

            #print ('-->', el, amt)

            # if len(sym_dict[el]) == 0:
            #     sym_dict[el] = "0"
            #print ('-->', el, amt)
            if amt.strip() == "":
                amt = "1"
            if el in sym_dict:
                # if len(sym_dict[el]) == 0:
                #      sym_dict[el] = "0"
                sym_dict[el] = '(' + sym_dict[el] + ')' + '+' + '(' + amt + ')' + '*' + '(' + str(factor) + ')'
            else:
                sym_dict[el] = '(' + amt + ')' + '*' + '(' + str(factor) + ')'
            f = f.replace(m.group(), "", 1)
        if f.strip():
            return collections.OrderedDict()
            # print("{} is an invalid formula!".format(f))

        """
        refinement of non-variable values
        """
        for el, amt in sym_dict.items():
            sym_dict[el] = self.__simplify(amt)

        #print ('Get sym_dict output:', sym_dict)

        return sym_dict

    ###################################################################################################################
    # Reconstruct formula / dictionary lookup
    ###################################################################################################################

    def split_material_name(self, material_string):
        """
        Splitting material string into chemical name + chemical formula
        :param material_string: in form of "chemical name chemical formula"/"chemical name [chemical formula]"
        :return: name: <str> chemical name found in material string
                formula: <str> chemical formula found in material string
                structure: <dict> output of get_structure_by_formula()
        """
        formula = ''
        structure = self.__empty_structure().copy()
        material_string = material_string.replace('[', ' [')

        split = re.split('\s', material_string)

        if len(split) == 1:
            return material_string, formula, structure

        formula_e = split[-1].strip('[]')
        #print ('->', formula_e)
        if formula_e == '':
            return '', '', self.__empty_structure().copy()

        if re.match('(\s*\([IV,]+\))', formula_e):
            formula_e = ''
        try:
            structure_e = self.get_structure_by_formula(formula_e)
            composition_e = structure_e['elements']
        except:
            composition_e = {}
            formula_e = ''
            structure_e = self.__empty_structure().copy()
        #pprint(structure_e)

        formula_b = split[0].strip('[]')
        #print ('->', formula_b)
        if formula_e == '':
            return '', '', self.__empty_structure().copy()

        if re.match('(\s*\([IV,]+\))', formula_b):
            formula_b = ''
        try:
            structure_b = self.get_structure_by_formula(formula_b)
            composition_b = structure_b['elements']
        except:
            composition_b = {}
            formula_b = ''
            structure_b = self.__empty_structure().copy()
        #pprint(structure_e)

        if composition_e != {} and formula_e not in self.__list_of_elements_1 + self.__list_of_elements_2:
            split = split[:-1]
            structure = structure_e
            formula = formula_e
        elif composition_b != {} and formula_b not in self.__list_of_elements_1 + self.__list_of_elements_2:
            split = split[1:]
            structure = structure_b
            formula = formula_b
        else:
            formula = ''
            structure = self.__empty_structure().copy()

        #print("Final structure:")
        #pprint(structure)

        name_terms = [p for p in split if
                      p.lower().strip('., -;:').rstrip('s') in self.__chemicals or 'hydrate' in p]
        # TODO: need better approach to sort out chemical names from trash


        if len(name_terms) > 0:
            name = ''.join([t + ' ' for t in split]).strip(' ')
        else:
            name = formula
            formula = ''
            structure = self.__empty_structure().copy()

        return name, formula, structure

    def reconstruct_formula(self, material_name, valency=''):
        """
        reconstructing chemical formula for simple chemical names anion + cation
        :param material_name: <str> chemical name
        :param valency: <str> anion valency
        :return: <str> chemical formula
        """

        output_formula = ''
        material_name = re.sub('(\([IV]*\))', ' \\1 ', material_name)
        material_name = re.sub('\s{2,}', ' ', material_name)

        terms_list = []
        valency_list = []
        hydrate = ''
        cation_prefix_num = 0
        cation_data = {"c_name": "", "valency": [], "e_name": "", "n_atoms": 0}

        #ions_valency = {}
        #prev_ion = ''
        for t in material_name.split(' '):
            if t.strip('()') in self.__rome2num:
                valency_list.append(self.__rome2num[t.strip('()')])
                #ions_valency[prev_ion] = self.__rome2num[t.strip('()')]
                continue
            if 'hydrate' in t.lower():
                hydrate = t
                continue
            #prev_ion = t.lower().rstrip('s')
            #ions_valency[prev_ion] = -1
            terms_list.append(t.strip(' -'))

        # print ('->', terms_list)
        # print ('->', valency_list)
        # print ('->', hydrate)

        terms_list_upd = []
        for t in terms_list:
            if all(p not in self.__prefixes2num for p in t.split('-')):
                terms_list_upd.extend(_ for _ in t.split('-'))
            else:
                terms_list_upd.append(''.join([p for p in t.split('-')]))
        terms_list = terms_list_upd
        #print ('->', terms_list)

        t = ''.join([t + ' ' for t in terms_list]).lower().strip(' ')
        if t in self.__anions:
            return self.__anions[t]['e_name']
        if t in self.__cations:
            return self.__cations[t]['e_name']

        if len(terms_list) < 2:
            return output_formula

        # if len(valency_list) > 1:
        #     print ('WARNING! Found many valencies per chemical name ' + material_name)
        #     print (valency_list)

        anion = terms_list.pop().lower().rstrip('s')

        if valency == '':
            valency_num = max(valency_list + [0])
        else:
            valency_num = self.__rome2num[valency.strip('()')]

        #anion_valency_num = ions_valency[anion]
        next_term = terms_list.pop()
        if 'hydrogen' in next_term.lower() and len(terms_list) != 0:
            anion = next_term + ' ' + anion
        else:
            terms_list += [next_term]

        #print ('->', terms_list)
        #print ('->', anion)
        #print ('->', anion, self.__get_prefix(anion))
        _, anion_prefix_num, anion = ('', 0, anion) if anion.lower() in self.__anions else self.__get_prefix(anion)
        #print ('->', anion, anion_prefix_num)

        if anion in self.__anions:
            anion_data = self.__anions[anion].copy()
        elif anion in ['metal']:
            return self.__cations[terms_list[0]]['e_name']
        else:
            return output_formula

        if len(terms_list) >= 2:
            return output_formula

        if len(terms_list) == 1:
            cation = terms_list[0]
            #cation_valency_num = ions_valency[cation.lower().rstrip('s')]
            _, cation_prefix_num, cation = ('', 0, cation) if cation.lower() in self.__cations \
                else self.__get_prefix(cation)
            if cation.lower() in self.__cations:
                cation = cation.lower()
                cation_data = self.__cations[cation].copy()
            elif cation in self.__element2name:
                cation_data = self.__cations[self.__element2name[cation]].copy()
            else:
                return output_formula

        # print ('Formula reconstruction data:')
        # print ('->', anion, anion_prefix_num)
        # pprint(anion_data)
        # print ('->', cation, cation_prefix_num)
        # pprint(cation_data)

        if len(cation_data['valency']) > 1 and valency_num != 0:
            if valency_num not in cation_data['valency']:
                print ('WARNING! Not common valency value for ' + material_name)
                print(cation_data['valency'])
                print(valency_num)
            cation_data['valency'] = [valency_num]

        output_formula = self.__build_formula(anion=anion_data,
                                              cation=cation_data,
                                              cation_prefix_num=cation_prefix_num,
                                              anion_prefix_num=anion_prefix_num)

        if hydrate != '':
            _, hydrate_prefix_num, hydrate = self.__get_prefix(hydrate)
            hydrate_prefix = '' if hydrate_prefix_num in [0, 1] else str(hydrate_prefix_num)
            output_formula = output_formula + '·' + hydrate_prefix + 'H2O'

        return output_formula

    def __build_formula(self, cation, anion, cation_prefix_num=0, anion_prefix_num=0):

        cation_stoich = 0
        anion_stoich = 0

        # print ('Building formula:')
        # pprint (cation)
        # print (cation_prefix_num)
        # pprint (anion)
        # print (anion_prefix_num)

        if anion_prefix_num + cation_prefix_num == 0 or anion_prefix_num * cation_prefix_num != 0:
            v_cation = abs(cation['valency'][0])
            v_anion = abs(anion['valency'][0])
            cm = self.__lcm(v_cation, v_anion)
            cation_stoich = cm // v_cation
            anion_stoich = cm // v_anion

        if anion_prefix_num != 0:
            anion_stoich = anion_prefix_num
            i = 0
            cation_stoich = 0
            while cation_stoich == 0 and i < len(cation['valency']):
                cation_stoich = anion_prefix_num * abs(anion['valency'][0]) // abs(cation['valency'][i])
                i = i+1

        if cation_prefix_num != 0:
            cation_stoich = cation_prefix_num
            anion_stoich = cation_prefix_num * abs(cation['valency'][0]) // abs(anion['valency'][0])

        anion_name_el = anion['e_name']
        if anion_stoich > 1 and anion['n_atoms'] > 1:
            anion_name_el = '(' + anion_name_el + ')'

        cation_name_el = cation['e_name']
        if cation_stoich > 1 and cation['n_atoms'] > 1:
            cation_name_el = '(' + cation_name_el + ')'

        return "{0}{1}{2}{3}".format(cation_name_el, self.__cast_stoichiometry(cation_stoich), anion_name_el,
                                     self.__cast_stoichiometry(anion_stoich))

    def __get_prefix(self, material_name):

        pref = ''
        pref_num = 0
        material_name_upd = material_name

        for p in self.__prefixes2num.keys():
            if material_name.lower().find(p) == 0 and p != '':
                pref = p
                pref_num = self.__prefixes2num[p]
                material_name_upd = material_name_upd[len(p):].strip('-')

                if material_name_upd == 'xide':
                    material_name_upd = 'oxide'

        return pref, pref_num, material_name_upd

    ###################################################################################################################
    # Splitting mixtures
    ###################################################################################################################

    def split_material(self, material_name):
        """
        splitting mixture/composite/solid solution/alloy into compounds with fractions
        :param material_name: <str> material formula
        :return: <list> of <tuples>: (compound, fraction)
        """

        split = self.__split_name(material_name)
        l = 0
        while len(split) != l:
            l = len(split)
            split = [p for s in split for p in self.__split_name(s[0], s[1])]

        output = []
        for m, f in split:
            try:
                f = smp.simplify(f)
                if f.is_Number:
                    f = round(float(f), 3)
                f = str(f)
            except:
                f = '1'

            f = self.__simplify(f)
            output.append((m, f))

        return output

    def __split_name(self, material_name_, init_fraction='1'):
        #print ('Split name:', material_name_)

        re_str = "(?<=[0-9\)])[-⋅·∙\∗](?=[\(0-9])|(?<=[A-Z])[-⋅·∙\∗](?=[\(0-9])|(?<=[A-Z\)])[-⋅·∙\∗](?=[A-Z])|(?<=[" \
                 "0-9\)])[-⋅·∙\∗](?=[A-Z])"
        re_str = re_str+''.join(['|(?<='+e+')[-⋅·∙\∗](?=[\(0-9A-Z])' for e in self.__list_of_elements_1+self.__list_of_elements_2])

        material_name = material_name_.replace(' ', '')
        material_name = material_name.replace('[', '(')
        material_name = material_name.replace(']', ')')

        if '(1-x)' == material_name[0:5]:
            material_name = material_name.replace('(x)', 'x')
            parts = re.findall('\(1-x\)(.*)[-+·∙\∗⋅]x(.*)', material_name)
            parts = parts[0] if parts != [] else (material_name[5:], '')
            return [(parts[0].lstrip(' ·*⋅'), '1-x'), (parts[1].lstrip(' ·*'), 'x')]

        parts = re.split(re_str, material_name)

        #print ('-->', parts)

        if len(parts) > 1:
            parts_upd = [p for part in parts for p in
                         re.split('(?<=[A-Z\)])[-·∙\∗⋅](?=[xyz])|(?<=O[0-9\)]+)[-·∙\∗⋅](?=[xyz])', part)]
        else:
            parts_upd = parts

        #print ('-->', parts_upd)

        if any(m.strip('0987654321') in self.__list_of_elements_1 + self.__list_of_elements_2 for m in parts_upd[:-1]):
            parts_upd = [''.join([p+'-' for p in parts_upd]).rstrip('-')]

        #print ('-->', parts_upd)

        merged_parts = [parts_upd[0]]
        for m in parts_upd[1:]:
            if re.findall('[A-Z]', m) == ['O']:
                to_merge = merged_parts.pop() + '-' + m
                merged_parts.append(to_merge)
            else:
                merged_parts.append(m)

        #print ('-->', merged_parts)

        composition = []
        for m in merged_parts:
            fraction = ''
            i = 0
            while i < len(m) and not m[i].isupper():
                fraction = fraction + m[i]
                i += 1
            fraction = fraction.strip('()')
            if fraction == '':
                fraction = '1'
            else:
                m = m[i:]

            fraction = '(' + fraction + ')*(' + init_fraction + ')'

            if m != '':
                composition.append((m, fraction))

        #print ('-->', composition)
        return composition

    def get_additives(self, material_name):
        """
        resolving doped part in material string
        :param material_name: <str> material string
        :return: <list> of additives,
                <str> new material name
        """
        new_material_name = material_name
        additives = []

        # additions = ['doped', 'stabilized', 'activated','coated', 'modified']

        new_material_name = new_material_name.replace('codoped', 'doped')

        # checking for "doped with"
        for r in ['coated', 'activated', 'modified', 'stabilized', 'doped']:
            parts = [w for w in re.split(r + ' with', new_material_name) if w != '']
            if len(parts) > 1:
                new_material_name = parts[0].strip(' -+')
                additives.append(parts[1].strip())

        # checking for element-doped prefix
        for r in ['coated', 'activated', 'modified', 'stabilized', 'doped']:
            parts = [w for w in re.split('(.*)[-\s]{1}' + r + ' (.*)', new_material_name) if w != '']
            if len(parts) > 1:
                new_material_name = parts.pop()
                additives.extend(p for p in parts)

        if '%' in new_material_name:
            new_material_name = new_material_name.replace('.%', '%')
            parts = re.split('[\-+:·]\s*[0-9\.]*\s*[vmolwtx\s]*\%', new_material_name)

        if len(parts) > 1:
            new_material_name = parts[0].strip(' -+')
            additives.extend(d.strip(' ') for d in parts[1:] if d != '')

        for part_ in new_material_name.split(':'):
            part_ = part_.strip(' ')

            part = part_
            if any(e in part for e in self.__list_of_elements_2):
                for e in self.__list_of_elements_2:
                    part = part.replace(e, '&&')

            #print (part_, part)
            #print ('->', re.split('[\s,]', part))
            #print ('->', part[0].strip('yx,+0987654321. '))

            if all(e.strip('zyx,+0987654321. ') in self.__list_of_elements_1 + ['R'] + ['&&']
                   for e in re.split('[\s,/]', part) if e != ''):
                additives.append(part_.strip(' '))
            else:
                new_material_name = part_.strip(' ')

        return additives, new_material_name

    ###################################################################################################################
    # Resolving abbreviations
    ###################################################################################################################

    def __is_abbreviation(self, word):
        if all(c.isupper() for c in re.sub('[0-9x\-\(\)\.]', '', word)) and len(re.findall('[A-NP-Z]', word)) > 1:
            return True

        return False

    def build_abbreviations_dict(self, materials_list, paragraph):
        """
        constructing dictionary of abbreviations appeared in material list
        :param paragraph: <list> of <str> list of sentences to look for abbreviations names
        :param materials_list: <list> of <str> list of materials entities
        :return: <dict> abbreviation: corresponding string
        """

        abbreviations_dict = {t: '' for t in materials_list if self.__is_abbreviation(t.replace(' ', '')) and t != ''}
        not_abbreviations = list(set(materials_list) - set(abbreviations_dict.keys()))

        # first find abreviations in current materials list
        for abbr in abbreviations_dict.keys():

            for material_name in not_abbreviations:
                if sorted(re.findall('[A-NP-Z]', abbr)) == sorted(re.findall('[A-NP-Z]', material_name)):
                    abbreviations_dict[abbr] = material_name

        # for all other abbreviations going through the paper text
        for abbr, name in abbreviations_dict.items():
            sents = ' '.join([s + ' ' for s in paragraph if abbr in s]).strip(' ').split(abbr)
            i = 0
            while abbreviations_dict[abbr] == '' and i < len(sents):
                sent = sents[i]
                for tok in sent.split(' '):
                    if sorted(re.findall('[A-NP-Z]', tok)) == sorted(re.findall('[A-NP-Z]', abbr)):
                        abbreviations_dict[abbr] = tok
                i += 1

        for abbr in abbreviations_dict.keys():
            parts = re.split('-', abbr)
            if all(p in abbreviations_dict for p in parts) and abbreviations_dict[abbr] == '' and len(parts) > 1:
                name = ''.join('(' + abbreviations_dict[p] + ')' + '-' for p in parts).rstrip('-')
                abbreviations_dict[abbr] = name

        empty_list = [abbr for abbr, name in abbreviations_dict.items() if name == '']
        for abbr in empty_list:
            del abbreviations_dict[abbr]

        return abbreviations_dict

    ###################################################################################################################
    # Methods to substitute variables
    ###################################################################################################################

    def __get_values(self, string, mode):
        values = []
        max_value = None
        min_value = None

        if len(string) == 0:
            return dict(values=[], max_value=None, min_value=None)

        # given range
        if mode == 'range':
            min_value, max_value = string[0]
            max_value = max_value.rstrip('., ')
            min_value = min_value.rstrip('., ')
            max_value = re.sub('[a-z]*', '', max_value)
            min_value = re.sub('[a-z]*', '', min_value)
            try:
                max_value = round(float(smp.simplify(max_value)), 4)
                min_value = round(float(smp.simplify(min_value)), 4) if min_value != '' else None
            except Exception as ex:
                max_value = None
                min_value = None
                template = "An exception of type {0} occurred when use sympy. Arguments:\n{1!r}."
                message = template.format(type(ex).__name__, ex.args)
                print(message)
            values = []

            return dict(values=values, max_value=max_value, min_value=min_value)

        # given list
        if mode == 'values':
            # values = re.split('[,\s]', string[0])
            values = re.split('[,\s]', re.sub('[a-z]+', '', string[0]))
            try:
                values = [round(float(smp.simplify(c.rstrip('., '))), 4) for c in values if
                          c.rstrip('., ') not in ['', 'and']]
                max_value = max(values) if values != [] else None
                min_value = min(values) if len(values) > 1 else None
            except Exception as ex:
                values = []
                max_value = None
                min_value = None
                template = "An exception of type {0} occurred when use sympy. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)

        return dict(values=values, max_value=max_value, min_value=min_value)

    def get_stoichiometric_values(self, var, sentence):
        """
        find numeric values of var in sentence
        :param var: <str> variable name
        :param sentence: <str> sentence to look for
        :return: <dict>: max_value: upper limit
                        min_value: lower limit
                        values: <list> of <float> numeric values
        """
        values = dict(values=[], max_value=None, min_value=None)

        regs = [(var + '\s*=\s*([-]{0,1}[0-9\.\,/and\s]+)[\s\)\]\,]', 'values'),
                (var + '\s*=\s*([0-9\.]+)\s*[-–]\s*([0-9\.\s]+)[\s\)\]\,m\%]', 'range'),
                ('([0-9\.\s]*)\s*[<≤⩽]{0,1}\s*' + var + '\s*[<≤⩽>]{1}\s*([0-9\.\s]+)[\s\)\]\.\,]', 'range'),
                (var + '[a-z\s]*from\s([0-9\./]+)\sto\s([0-9\./]+)', 'range'),
                ]

        for r, m in regs:
            #print ('-->', r)
            if values['values'] == [] and values['max_value'] is None:
                r_res = re.findall(r, sentence.replace(' - ', '-'))
                #print ('-->', r_res)
                values = self.__get_values(r_res, m)
                #print ('-->', values)
            #print ('----')

        return values

    def get_elements_values(self, var, sentence):
        """
        find elements values for var in the sentence
        :param var: <str> variable name
        :param sentence: <str> sentence to look for
        :return: <list> of <str> found values
        """
        values = re.findall(var + '\s*[=:]{1}\s*([A-Za-z0-9\+,\s]+)', sentence)
        values = [c.rstrip('0987654321+') for v in values for c in re.split('[,\s]', v)
                  if c.rstrip('0987654321+') in self.__list_of_elements_1 + self.__list_of_elements_2]

        return list(set(values))

    ###################################################################################################################
    # Splitting list of materials
    ###################################################################################################################

    def is_materials_list(self, material_string):
        if (any(a + 's' in material_string.lower() for a in self.__anions.keys()) or 'metal' in material_string) and \
                any(w in material_string for w in ['and ', ',', ' of ']):
            return True

        return False

    def reconstruct_list_of_materials(self, material_string):
        """
        split material string into list of compounds when it's given in form cation + several anions
        :param material_string: <str>
        :return: <list> of <str> chemical names
        """

        parts = [p for p in re.split('[\s\,]', material_string) if p != '']

       #print (self.__anions.keys())

        anion = [(i, p[:-1]) for i, p in enumerate(parts) if p[:-1].lower() in self.__anions.keys() or p[:-1].lower() == 'metal']
        cation = [(i, p) for i, p in enumerate(parts) if p.lower() in self.__cations.keys()
                  or p in self.__list_of_elements_1 + self.__list_of_elements_2]
        valencies = [(i - 1, p.strip('()')) for i, p in enumerate(parts) if p.strip('()') in self.__rome2num and i != 0]

       # print ('->', anion, cation)

        result = []
        if len(anion) == 1:
            for c_i, c in cation:

                if c in self.__element2name:
                    name = [self.__element2name[c]]
                else:
                    name = [c.lower()]
                valency = ''.join([v for v_i, v in valencies if v_i == c_i])
                if valency != '':
                    name.append('(' + valency + ')')
                name.append(anion[0][1])
                # formula = mp.reconstruct_formula(name.copy(), valency)
                hydr_i = material_string.find('hydrate')
                if hydr_i > -1:
                    pref = []
                    while material_string[hydr_i - 1] != ' ' and hydr_i > 0:
                        pref.append(material_string[hydr_i - 1])
                        hydr_i -= 1

                    pref = ''.join([p for p in reversed(pref)])

                    if pref not in self.__neg_prefixes:
                        # formula = formula+'·'+str(prefixes2num[pref])+'H2O'
                        name.append(pref + 'hydrate')
                result.append((''.join([n + ' ' for n in name]).strip(' '), valency))

        return result

    ###################################################################################################################
    # Misc
    ###################################################################################################################

    def cleanup_name(self, material_name):
        """
        cleaning up material name - fix due to tokenization imperfectness
        :param material_name: <str> material string
        :return: <str> updated material string
        """

        # print (material_name)

        # correct dashes
        dashes = [173, 8722, ord('\ue5f8')] + [i for i in range(8208, 8214)]
        re_str = ''.join([chr(c) for c in dashes])
        re_str = '[' + re_str + ']'
        material_name = re.sub(re_str, chr(45), material_name)

        material_name = material_name.replace(chr(160), '')

        # correcting dots
        dots = [42, 8226, 8270, 8729, 8901, 215, 65106, 65381, 12539, 9072]
        re_str = ''.join([chr(c) for c in dots])
        re_str = '[\\' + re_str + ']'
        material_name = re.sub(re_str, chr(183), material_name)

        #correcting slashes
        slashes = [8725]
        re_str = ''.join([chr(c) for c in slashes])
        re_str = '[\\' + re_str + ']'
        material_name = re.sub(re_str, chr(47), material_name)

        material_name = re.sub('\s*([-+±]){1}\s*(['+''.join([c for c in self.__greek_letters])+']{1})', '\\1δ', material_name)


        # removing phase
        for c in ['(s)', '(l)', '(g)', '(aq)']:
            material_name = material_name.replace(c, '')

        if any(a in material_name for a in ['→', '⟶','↑','↓','↔','⇌','⇒','⇔', '⟹']):
            return ''

        if 'hbox' in material_name.lower():
            material_name = re.sub('(\\\\[a-z\(\)]+)', '', material_name)
            for t in ['{', '}', '_', ' ']:
                material_name = material_name.replace(t, '')
            material_name = material_name.rstrip('\\')

        material_name = re.sub('\({0,1}[0-9\.]*\s*[⩽≤<]{0,1}\s*[x,y]{0,1}\s*[⩽=≤<]\s*[0-9\.-]*\){0,1}', '', material_name)

        if material_name == '' or len([c for c in material_name if c.isalpha()]) < 1:
            return ''

        for c in ['\(⩾99', '\(99', '\(98', '\(90', '\(95', '\(96', '\(Alfa', '\(Aldrich', '\(A.R.', '\(Aladdin', '\(Sigma',
                  '\(A.G', '\(Fuchen', '\(Furuuchi', '\(AR\)', '（x', '\(x', '\(Acr', '\(Koj', '\(Sho', '\(＞99']:
            split = re.split(c, material_name)
            if len(split) > 1 and (len(split[-1]) == '' or all(not s.isalpha() for s in split[-1])):
                #material_name = material_name.replace(split[-1], '')
                material_name = ''.join([s for s in split[:-1]])


        replace_dict = {'oxyde': 'oxide',
                        'luminum': 'luminium',
                        'magneshium': 'magnesium',
                        'stanate': 'stannate',
                        'sulph': 'sulf',
                        'buter': 'butyr',
                        'butir': 'butyr',
                        'butly': 'butyl',
                        'ethly': 'ethyl',
                        'ehtyl': 'ethyl',
                        'Abstract ': '',
                        'phio': 'thio',
                        'uim': 'ium',
                        'butryal': 'butyral',
                        'ooper': 'opper',
                        'acac': 'CH3COCHCOCH3',
                        'glasses': '',
                        'glass': '',
                        'ceramics': '',
                        'europeam': 'europium',
                        'siliminite': 'sillimanite',
                        'acethylene': 'acetylene',
                        'iso-pro': 'isopro',
                        'anhydrous': '',
                        'lathanum': 'lanthanum',
                        'bulk': '',
                        'Bulk': '',
                        '()': '',
                        'uium': 'ium',
                        'Anhydrous': '',
                        'sodiam': 'sodium'
                        }

        for typo, correct in replace_dict.items():
            material_name = material_name.replace(typo, correct)

        if material_name[-2:] == '/C':
            material_name = material_name[:-2]

        material_name = re.sub('(poly[\s-])(?=[a-z])', 'poly', material_name)

        if material_name[-2:] == '/C':
            material_name = material_name[:-2]

        if len(material_name.split(' ')) > 1:
            for v in re.findall('[a-z](\([IV,]+\))', material_name):
                material_name = material_name.replace(v, ' ' + v)

        if material_name != '':
            material_name = self.__check_parentheses(material_name)

        trash_symbs = ['#', '$', '!', '@', '©', '®', chr(8201), 'Ⓡ']
        for c in trash_symbs:
            material_name = material_name.replace(c, '')

        material_name = material_name.replace('[', '(')
        material_name = material_name.replace(']', ')')
        material_name = material_name.replace('{', '(')
        material_name = material_name.replace('}', '(')

        material_name = material_name.lstrip(') -')
        material_name = material_name.rstrip('( ,.:;-±/∓')

        if len(material_name) == 1 and material_name not in self.__list_of_elements_1:
            return ''

        if len(material_name) == 2 and \
                material_name not in self.__list_of_elements_2 and \
                material_name.rstrip('234') not in self.__list_of_elements_1 and \
                any(c not in self.__list_of_elements_1 for c in material_name):
            return ''

        material_name = re.sub('\s([0-9\.]*H2O)$', chr(183)+'\\1', material_name)

        return material_name

    def __check_parentheses(self, formula):

        new_formula = formula

        #new_formula = new_formula.replace('[', '(')
        #new_formula = new_formula.replace(']', ')')
        new_formula = new_formula.replace('{', '(')
        new_formula = new_formula.replace('}', ')')

        par_open = re.findall('\(', new_formula)
        par_close = re.findall('\)', new_formula)

        if new_formula[0] == '(' and new_formula[-1] == ')' and len(par_close) == 1 and len(par_open) == 1:
            new_formula = new_formula[1:-1]

        if len(par_open) == 1 and len(par_close) == 0:
            if new_formula.find('(') == 0:
                new_formula = new_formula.replace('(', '')
            else:
                new_formula += ')'

        if len(par_close) == 1 and len(par_open) == 0:
            if new_formula[-1] == ')':
                new_formula = new_formula.rstrip(')')
            else:
                new_formula = '(' + new_formula
        #
        # if len(par_close) - len(par_open) == 1 and new_formula[-1] == ')':
        #     new_formula = new_formula.rstrip(')')

        return new_formula

    def __simplify(self, value):

        """
        simplifying stoichiometric expression
        :param value: string
        :return: string
        """

        for l in self.__greek_letters:
            _clash[l] = smp.Symbol(l)

        new_value = value
        for i, m in enumerate(re.finditer('(?<=[0-9])([a-z' + ''.join(self.__greek_letters) + '])', new_value)):
            new_value = new_value[0:m.start(1) + i] + '*' + new_value[m.start(1) + i:]
        new_value = smp.simplify(smp.sympify(new_value, _clash))
        if new_value.is_Number:
            new_value = round(float(new_value), 4)

        return str(new_value).replace(' ', '')

    def __lcm(self, x, y):
        """This function takes two
        integers and returns the L.C.M."""

        # choose the greater number
        lcm = None
        if x > y:
            greater = x
        else:
            greater = y

        found = False
        while not found:
            if (greater % x == 0) and (greater % y == 0):
                lcm = greater
                found = True
            greater += 1

        return lcm

    # def __cast_stoichiometry(self, value):
    #     if value == 1:
    #         return ''
    #
    #     return str(value)

    def __cast_stoichiometry(self, value):

        value = float(value)
        if value == 1.0:
            return ''
        if value * 1000 % 1000 == 0.0:
            return str(int(value))

        return str(value)

    def __empty_structure(self):
        return {'elements': collections.OrderedDict(),
                'amounts_vars': {},
                'elements_vars': {},
                'hydrate': '',
                'phase': '',
                'formula': '',
                'oxygen_deficiency': ''}

    def __is_int(self, num):
        try:
            return round(float(num), 4) == round(float(num), 0)
        except:
            return False

    def __substitute_additive(self, additive, material_formula, material_composition):

        new_material_composition = []
        new_material_formula = material_formula

        if additive[-1] == '+':
            additive = additive.rstrip('+0987654321')

        #print ('-->', additive)

        r = '^[x0-9\.]+|[x0-9\.]+$'

        coeff = re.findall(r, additive)
        element = [s for s in re.split(r, additive) if s != ''][0]

        #print ('-->', coeff, element)

        if coeff == [] or element not in self.__list_of_elements_1+self.__list_of_elements_2:
            return new_material_formula, material_composition

        for compound in material_composition:
            expr = ''.join(['(' + v + ')+' for e, v in compound['elements'].items()]).rstrip('+')

            coeff = coeff[0] if not re.match('^[0]+[1-9]', coeff[0]) else '0.' + coeff[0][1:]
            expr = expr + '+(' + coeff + ')'
            #print ('-->', expr, self.__simplify(expr), self.__is_int(self.__simplify(expr)))

            if self.__is_int(self.__simplify(expr)):
                #print('-->', element, coeff)
                new_name = element + coeff + compound['formula']
                new_composition = compound['elements'].copy()
                #new_composition[element] = coeff
                new_composition.update({element: coeff})
                new_composition.move_to_end(element, last=False)

                new_material_composition.append(dict(
                    formula=new_name,
                    amount=compound['amount'],
                    elements=new_composition
                ))
                new_material_formula = new_material_formula.replace(compound['formula'], new_name)
            else:
                new_material_composition.append(dict(
                    formula = compound['formula'],
                    amount = compound['amount'],
                    elements = compound['elements']
                ))

        return new_material_formula, new_material_composition

    def __is_abbreviation_like(self, structure):
        if any(all(e.isupper() and s in ['1.0', '1'] for e, s in compound['elements'].items()) for compound in
               structure['composition']):
            # if all(e.isupper() and s in ['1.0', '1'] for f, c in m_struct['composition'].items() for e, s in c.items()):
            return True

        elements_vars = [el for el in structure['elements_vars'].keys() if len(el) == 1 and el.isupper()]
        if len(elements_vars) > 1:
            return True

        if all(c.isupper() for c in structure['material_formula']) and any(
                c not in self.__list_of_elements_1 for c in structure['material_formula']):
            return True

        if re.findall('[A-Z]{3,}', structure['material_formula']) != [] and \
                all(w not in structure['material_formula'] for w in ['CH', 'COO', 'OH']):
            return True

        if 'PV' == structure['material_formula'][0:2]:
            return True

        return False

    def get_element(self, name):
        if name in self.__anions:
            return self.__anions[name]['e_name']
        if name in self.__cations:
            return self.__cations[name]['e_name']
        return ''

    def __combine_formula(self, material_composition):
        formula = ''
        if len(material_composition) == 1:
            return material_composition[0]['formula'].replace('*', '')

        coeff = ''
        for c in material_composition:
            #if c['amount'] in ['1.0', '1']:
            #    coeff = ''
            if coeff != '':
                coeff = self.__simplify(coeff)
            if all(ch.isdigit() or ch == '.' for ch in c['amount']):
                coeff = self.__cast_stoichiometry(c['amount'])
            else:
                coeff = '(' + c['amount'] + ')'

            sign = '-'
            if 'H2O' in c['formula']:
                sign = '·'

            formula = formula + sign + coeff + c['formula']

        formula = formula.replace('*', '')

        return formula.lstrip('-')
