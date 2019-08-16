# Material Parser and Reactions Extractor

The package consists of two modules:

 * Material Parser: class to extract composition of a given material
 * Recipe Extractor: wrapping class on Material Parser

**Material Parser** processes one material entity per run, and allows for:

 * parsing material string into composition,
 * constructing dictionary of materials abbreviations,
 * finding values of stoichiometric and elements variables,
 * splitting mixtures/composites/alloys/solid solutions into compounds
 * reconstructing chemical formula from chemical name

**Recipe Extractor** is a wrapper on top of Material Parser.
It combines methods from Material Parser class into pipeline to build recipe: targets and their precursors.
As an input it uses list of materials found in abstract and synthesis paragraph together with the corresponsing text sentences.
It is specifically designed for the output and purposes of our project.
However, it can serve as an example of Material Parser usage.

**Recipe Extractor** does the following:

 * uses text sentences to build abbreviation vocabulary for all abbreviation found in materials list
 * resolves abbreviations in materials list (substitute formula instead of abbreviation)
 * if targets list is empty, uses targets from abstract
 * obtains materials compositions
 * substitutes values of stoichiometric and elements parameters

Matching of elements and constructing list of unbalanced reaction are suppressed after version 3
 
### Installation:
```
git clone https://github.com/CederGroupHub/MaterialParser.git
cd MaterialParser
pip install -r requirements.txt -e .
```

### Initialization:
```
from material_parser import MaterialParser
mp = MaterialParser(verbose=False, pubchem_lookup=False, fails_log=False)

from recipe_extractor import recipe_extractor
rex = RecipeExtractor()
```

### Material parser

#### Initialization

 ```
 verbose: <bool> print supplemental information
 pubchem_lookup: <bool> look for unknown chemical names in PubChem (not implemented, slows down computations significantly)
 fails_log: <bool> outputs log of materials for which mp.parse_material failed (useful when long list of materials in processes)
 ```

#### Methods to extract material composition

 * mp.parse_material(material_string)
     ```
     main method to parse material string into chemical structure and
     convert chemical name into chemical formula
    :param material_string: <str> material name/formula
    :return: dict(material_string: <str> initial material string,
                 material_name: <str> chemical name of material found in the string
                 material_formula: <str> chemical formula of material
                 dopants: <list> list of dopped materials/elements appeared in material string
                 phase: <str> material phase appeared in material string
                 hydrate: <float> if material is hydrate fraction of H2O
                 is_mixture: <bool> material is mixture/composite/alloy/solid solution
                 is_abbreviation: <bool> material is similar to abbreviation
                 fraction_vars: <dict> elements fraction variables and their values
                 elements_vars: <dict> elements variables and their values
                 composition: <dict> compound constitute of the material: composition (element: fraction) and
                                                                        fraction of compound)
     ```

 * mp.get_structure_by_formula(chemical_formula)
     ```
     parsing chemical formula in composition
    :param chemical_formula: <str> chemical formula
    :return: dict(formula: <str> formula string corresponding to obtained composition
                 composition: <dict> element: fraction
                 fraction_vars: <dict> elements fraction variables: <list> values
                 elements_vars: <dict> elements variables: <list> values
                 hydrate: <str> if material is hydrate fraction of H2O
                 phase: <str> material phase appeared in formula
                )
     ```

#### Methods to reconstruct chemical formula from material name

 * mp.split_material_name(material_string)
    ```
    splitting material string into chemical name + chemical formula
    :param material_string: <str> in form of
    "chemical name chemical formula"/"chemical name [chemical formula]"
    :return: name: <str> chemical name found in material string
            formula: <str> chemical formula found in material string
            structure: <dict> output of get_structure_by_formula()
    ```

 * mp.reconstruct_formula(material_name, valency='')
    ```
    reconstructing chemical formula for simple chemical names anion + cation
    :param material_name: <str> chemical name
    :param valency: <str> anion valency
    :return: <str> chemical formula
    ```

#### Methods to simplify material string

 * mp.split_material(material_name)
    ```
    splitting mixture/composite/solid solution/alloy into compound+fraction
    :param material_name: <str> material formula
    :return: <list> of <tuples>: (compound, fraction)
    ```

 * mp.get_dopants(material_name)
    ```
    resolving doped part in material string
    :param material_name: <str> material string
    :return: <list> of dopants,
            <str> new material name
    ```
 * mp.reconstruct_list_of_materials(material_string)
    ```
    split material string into list of compounds
    when it's given in form cation + several anions
    for example: "oxides of manganese and lithium"
    :param material_string: <str>
    :return: <list> of <str> chemical names
    ```

 * mp.cleanup_name(material_name)
    ```
    cleaning up material name - fix due to tokenization imperfectness
    :param material_name: <str> material string
    :return: <str> updated material string
    ```

#### Methods to resolve abbreviations and variables

 * mp.build_abbreviations_dict(materials_list, sentences)
    ```
    constructing dictionary of abbreviations appeared in material list
    :param paragraph: <list> of sentences to look for abbreviations names
    :param materials_list: <list> of <str> list of materials entities
    :return: <dict> abbreviation: corresponding string
    ```

 * mp.get_stoichiometric_values(var, sentence)
    ```
    find numeric values of var in sentence
    :param var: <str> variable name
    :param sentence: <str> sentence to look for
    :return: <dict>: max_value: upper limit
                    min_value: lower limit
                    values: <list> of <float> numeric values
    ```

 * mp.get_elements_values(var, sentence):
    ```
    find elements values for var in the sentence
    :param var: <str> variable name
    :param sentence: <str> sentence to look for
    :return: <list> of <str> found values
    ```

### Recipe extractor:

 * rex.get_composition(abstract_materials_, synthesis_materials, abstract=None, syn_paragraph=None)
    ```
    main method to convert list of materials into composition
    based on output of MER
    :param abstract_materials_: <dict> targets: list of materials
                                        precursors: list of materials
                                        others: list of materials
    :param synthesis_materials: <dict> targets: list of materials
                                        precursors: list of materials
                                        others: list of materials
    :param abstract: <list> of <str> abstract sentences
    :param syn_paragraph: <list> of <str> paragraph sentences
    :return: output_structure: <dict> targets: list of material structures (output of mp.parse_material)
                                      precursors: list of material structures
                                      targets: list of material structures (output of mp.parse_material)
                                      others: list of material structures
            fails: <dict> targets: list of failed material strings
                          precursors: list of failed material strings
                          targets: list of failed material strings
                          others: list of failed material strings
            abbreviations: <dict> abbreviation: corresponding name
    ```

 * rex.substitute_elements(material_structure)
    ```
    substituting values for elements variables into formula
    :param material_structure: <dict> output of mp.parse_material() with filled "element_vars"
    :return: list of structures derived from input with substitution of all element_vars
    ```

 * rex.substitute_fraction(material_structure)
    ```
    substituting values for elements fractions variables into formula
    :param material_structure: <dict> output of mp.parse_material() with filled "fraction_vars"
    :return: list of structures derived from input with substitution of all fraction_vars
    ```