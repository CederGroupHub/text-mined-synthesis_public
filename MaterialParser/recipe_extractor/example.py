# coding=utf-8

import recipe_extractor
import json
from pprint import pprint
from chemdataextractor.doc import Paragraph
from progressbar import ProgressBar

rex = recipe_extractor.RecipeExtractor(verbose=False, pubchem_lookup=False)

test_set = json.loads(open('test_data.json').read())

for item in test_set:

    synthesis_materials = dict(
        targets = item['materials']['targets'],
        precursors = item['materials']['precursors'],
        others = item['materials']['others'],
    )

    abstract_materials = item['materials']['abstract']
    abstract = item['abstract']
    synthesis = item['syn_paragraph']

    extracted_materials, fails, abbreviations = rex.get_composition(abstract_materials, synthesis_materials, abstract, synthesis)

    fraction_substitution = []
    elements_substitution = []
    for structure in extracted_materials['targets'] + extracted_materials['precursors']:
        if structure['fraction_vars'] != {}:
            fraction_substitution.extend(m for m in rex.substitute_fraction(structure))

        if structure['elements_vars'] != {}:
            elements_substitution.extend(m for m in rex.substitute_elements(structure))

    composition = {'targets': extracted_materials['targets'],
                   'precursors': sorted(extracted_materials['precursors'], key=lambda x: x['material_string']),
                   'others': sorted(extracted_materials['others'], key=lambda x: x['material_string'])
                   }

    correct_abbreviations = item['abbreviations']
    correct_composition = item['composition']
    correct_fraction_subs = item['fraction_substitution']
    correct_elements_subs = item['elements_substitution']

    if correct_abbreviations != abbreviations:
        print ('Failed on abbreviations: '+item['doi'])
    if correct_composition != composition:
        print ('Failed on composition: '+item['doi'])
        if composition['targets'] != correct_composition['targets']:
            print ('in targets')
        if composition['precursors'] != correct_composition['precursors']:
            print ('in precursors')
        if composition['others'] != correct_composition['others']:
            print ('in others')


    if correct_fraction_subs != fraction_substitution:
        print ('Failed on fraction variables substitution: ' + item['doi'])
    if correct_elements_subs != elements_substitution:
        print ('Failed on elements variables substitution: ' + item['doi'])

print ('Done!')