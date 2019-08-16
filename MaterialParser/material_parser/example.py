# coding=utf-8
from material_parser import MaterialParser
import json
from pprint import pprint


mp = MaterialParser(pubchem_lookup=False, verbose=False)

test_set = json.loads(open('test_data.json').read())

for item in test_set:

    material = item['material']
    correct = item['parser_output']
    print (material)

    list_of_materials = mp.reconstruct_list_of_materials(material)
    list_of_materials = list_of_materials if list_of_materials != [] else [(material, '')]
    structure = []
    for m, val in list_of_materials:
        structure.append(mp.parse_material(m))
    #pprint(structure)

    if structure != correct:
        print (material)
        print ('Found:')
        pprint(structure)
        print ('Correct:')
        pprint(correct)

print ('Done!')




