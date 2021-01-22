from .utils import found_package
import regex

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# constant
if found_package('material_parser'):
    from material_parser.material_parser import MaterialParser
    mp = MaterialParser(pubchem_lookup=False)
allNonMetalElements = set(['C', 'H', 'O', 'N', 'Cl', 'F', 'P', 'S', 'Br', 'I', 'Se'] + ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'])
# element table by symbol of elements
elementTable = {
'H': [1, 'hydrogen'],  'He': [2, 'helium'],  'Li': [3, 'lithium'],  'Be': [4, 'beryllium'],  'B': [5, 'boron'],  
'C': [6, 'carbon'],  'N': [7, 'nitrogen'],  'O': [8, 'oxygen'],  'F': [9, 'fluorine'],  'Ne': [10, 'neon'],  
'Na': [11, 'sodium'],  'Mg': [12, 'magnesium'],  'Al': [13, 'aluminium'],  'Si': [14, 'silicon'],  'P': [15, 'phosphorus'],  
'S': [16, 'sulfur'],  'Cl': [17, 'chlorine'],  'Ar': [18, 'argon'],  'K': [19, 'potassium'],  'Ca': [20, 'calcium'],  
'Sc': [21, 'scandium'],  'Ti': [22, 'titanium'],  'V': [23, 'vanadium'],  'Cr': [24, 'chromium'],  'Mn': [25, 'manganese'],  
'Fe': [26, 'iron'],  'Co': [27, 'cobalt'],  'Ni': [28, 'nickel'],  'Cu': [29, 'copper'],  'Zn': [30, 'zinc'],  
'Ga': [31, 'gallium'],  'Ge': [32, 'germanium'],  'As': [33, 'arsenic'],  'Se': [34, 'selenium'],  'Br': [35, 'bromine'],  
'Kr': [36, 'krypton'],  'Rb': [37, 'rubidium'],  'Sr': [38, 'strontium'],  'Y': [39, 'yttrium'],  'Zr': [40, 'zirconium'],  
'Nb': [41, 'niobium'],  'Mo': [42, 'molybdenum'],  'Tc': [43, 'technetium'],  'Ru': [44, 'ruthenium'],  'Rh': [45, 'rhodium'],  
'Pd': [46, 'palladium'],  'Ag': [47, 'silver'],  'Cd': [48, 'cadmium'],  'In': [49, 'indium'],  'Sn': [50, 'tin'],  
'Sb': [51, 'antimony'],  'Te': [52, 'tellurium'],  'I': [53, 'iodine'],  'Xe': [54, 'xenon'],  'Cs': [55, 'caesium'],  
'Ba': [56, 'barium'],  'La': [57, 'lanthanum'],  'Ce': [58, 'cerium'],  'Pr': [59, 'praseodymium'],  'Nd': [60, 'neodymium'],  
'Pm': [61, 'promethium'],  'Sm': [62, 'samarium'],  'Eu': [63, 'europium'],  'Gd': [64, 'gadolinium'],  'Tb': [65, 'terbium'],  
'Dy': [66, 'dysprosium'],  'Ho': [67, 'holmium'],  'Er': [68, 'erbium'],  'Tm': [69, 'thulium'],  'Yb': [70, 'ytterbium'],  
'Lu': [71, 'lutetium'],  'Hf': [72, 'hafnium'],  'Ta': [73, 'tantalum'],  'W': [74, 'tungsten'],  'Re': [75, 'rhenium'],  
'Os': [76, 'osmium'],  'Ir': [77, 'iridium'],  'Pt': [78, 'platinum'],  'Au': [79, 'gold'],  'Hg': [80, 'mercury'],  
'Tl': [81, 'thallium'],  'Pb': [82, 'lead'],  'Bi': [83, 'bismuth'],  'Po': [84, 'polonium'],  'At': [85, 'astatine'],  
'Rn': [86, 'radon'],  'Fr': [87, 'francium'],  'Ra': [88, 'radium'],  'Ac': [89, 'actinium'],  'Th': [90, 'thorium'],  
'Pa': [91, 'protactinium'],  'U': [92, 'uranium'],  'Np': [93, 'neptunium'],  'Pu': [94, 'plutonium'],  'Am': [95, 'americium'],  
'Cm': [96, 'curium'],  'Bk': [97, 'berkelium'],  'Cf': [98, 'californium'],  'Es': [99, 'einsteinium'],  'Fm': [100, 'fermium'],  
'Md': [101, 'mendelevium'],  'No': [102, 'nobelium'],  'Lr': [103, 'lawrencium'],  'Rf': [104, 'rutherfordium'],  'Db': [105, 'dubnium'],  
'Sg': [106, 'seaborgium'],  'Bh': [107, 'bohrium'],  'Hs': [108, 'hassium'],  'Mt': [109, 'meitnerium'],  'Ds': [110, 'darmstadtium'],  
'Rg': [111, 'roentgenium'],  'Cn': [112, 'copernicium'],  'Nh': [113, 'nihonium'],  'Fl': [114, 'flerovium'],  'Mc': [115, 'moscovium'],  
'Lv': [116, 'livermorium'],  'Ts': [117, 'tennessine'],  'Og': [118, 'oganesson'],  
}
allElements = list(elementTable.keys())
allElements = sorted(allElements, key=lambda ele: len(ele), reverse=True)
allEleText = '|'.join(allElements)
pattern_species = regex.compile(r'^\b(((' + allEleText + r')[\·0-9]{0,5})+)\b$')

def parse_material(material_text, para_text):
    # goal
    parsed_material = {'dopants': None, 'composition': None}
    #     get dopants
    dopants, new_material = mp.separate_additives(material_text)

    #     get abbreviation
    tmp_abbr = mp.build_acronyms_dict([new_material], [para_text])
    new_material2 = []
    if new_material in tmp_abbr:
        new_material2 = tmp_abbr[new_material]
    else:
        new_material2 = new_material

    if len(dopants) > 0:
        parsed_material['dopants'] = dopants
    try:
        # material parser version 6.1.0
        list_of_materials = mp.split_materials_list(new_material2)
        list_of_materials = list_of_materials if list_of_materials != [] else [(new_material2, '')]
        tmp_structure = []
        for m, val in list_of_materials:
            tmp_structure.extend(mp.parse_material_string(m)['composition'])
        tmp_comp = merge_struct_comp(tmp_structure)
        if tmp_comp != None and len(tmp_comp) > 0:
            parsed_material['composition'] = tmp_comp
        else:
            # print('unresolved')
            pass
    except:
        print('unresolved')
        pass
    return parsed_material


# used to merge structure compositions from material parser
def merge_struct_comp(struct_list):
    # goal
    combined_comp = {}
    all_comps = []
    for tmp_struct in struct_list:
        if tmp_struct.get('amount', '1.0') != '1.0':
            # multiply by coefficient if amount is not 1
            tmp_comp = {}
            for ele, num in tmp_struct['elements'].items():
                tmp_comp[ele] = '(' + num + ')*(' + tmp_struct['amount'] + ')'
            all_comps.append(tmp_comp)
        else:
            all_comps.append(tmp_struct['elements'])

    for tmp_comp in all_comps:
        for k, v in tmp_comp.items():
            if k not in combined_comp:
                combined_comp[k] = v
            else:
                combined_comp[k] += (' + ' + v)
    return combined_comp

def count_metal_ele(material_text, para_text):
    metal_ele_num = 0
    parsed_material = parse_material(material_text, para_text)
    if parsed_material['composition']:
        ele_set = set(parsed_material['composition'].keys())
        if ele_set.issubset(allElements):
            metal_ele_num = len(ele_set-allNonMetalElements)
    return metal_ele_num

# get ele feature for one material
def get_ele_feature(material_text, para_text):
    metal_ele_num = 0
    only_CHO = 0
    parsed_material = parse_material(material_text, para_text)
    if parsed_material['composition']:
        ele_set = set(parsed_material['composition'].keys())
        if ele_set.issubset(allElements):
            metal_ele_num = len(ele_set-allNonMetalElements)
            only_CHO = ele_set.issubset({'C', 'H', 'O'})
    if only_CHO:
        only_CHO = 1
    else:
        only_CHO = 0
    return metal_ele_num, only_CHO

# get ele features for all materials in a sentence
def get_ele_features(input_tokens, original_para_text):
    metal_ele_nums = []
    only_CHOs = []
    for i, t in enumerate(input_tokens):
        if t['text'] == '<MAT>':
            mat_text = original_para_text[t['start']: t['end']] 
            metal_ele_num, only_CHO = get_ele_feature(material_text=mat_text, para_text=original_para_text)
            metal_ele_nums.append(metal_ele_num)
            only_CHOs.append(only_CHO)
        else:
            metal_ele_nums.append(0)
            only_CHOs.append(0)
    return metal_ele_nums, only_CHOs


if __name__ == '__main__':
    print(get_ele_feature('ethanol', 'LiMnO2 (LMO) is synthesized from Li2CO3, Fe2O3, and H3PO4.'))
    print(get_ele_feature('Li2CO3', 'LiMnO2 (LMO) is synthesized from Li2CO3, Fe2O3, and H3PO4.'))
    print(get_ele_feature('LiMnO2', 'LiMnO2 (LMO) is synthesized from Li2CO3, Fe2O3, and H3PO4.'))
    print(get_ele_feature('LMO', 'LiMnO2 (LMO) is synthesized from Li2CO3, Fe2O3, and H3PO4.'))
    print(get_ele_feature('Mg(OH)2·4(MgCO3)·5H2O', 'LiMnO2 (LMO) is synthesized from Li2CO3, Fe2O3, and H3PO4.'))
    print(get_ele_features([{'text': '<MAT>', 'start' : 0, 'end': 6 }], 'LiMnO2 (LMO) is synthesized from Li2CO3, Fe2O3, and H3PO4.'))