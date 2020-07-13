from typing import List, Dict, NamedTuple, Optional


## Reaction formula data types

class Formula(NamedTuple):
    class FormulaPart(NamedTuple):
        amount: str
        material: str

    left_side: List[FormulaPart]
    right_side: List[FormulaPart]

    element_substitution: Dict[str, str]


## Material information data types


class Material(NamedTuple):
    class Composition(NamedTuple):
        formula: str
        amount: str
        elements: Dict[str, str]

    material_string: str  # String of the material as written in paper
    material_formula: str  # Formula of the material
    material_name: str  # **New field!** English name of the material

    phase: Optional[str]  # New field! Phase description of material
    is_acronym: bool  # **New field!** Whether the material is an acronym

    composition: List[Composition]  # List of compositions in mixture
    amounts_vars: Dict[str, List[str]]  # Amount variables (subscripts)
    elements_vars: Dict[str, List[str]]  # Chemical element variables

    additives: List[str]  # List of additives, dopants
    oxygen_deficiency: Optional[str]  # Whether the materials is oxygen deficient


## Experimental operations data types


class Operation(NamedTuple):
    class Conditions(NamedTuple):
        class Value(NamedTuple):
            min_value: float
            max_value: float
            values: List[float]
            units: str

        heating_temperature: Optional[List[Value]]
        heating_time: Optional[List[Value]]
        heating_atmosphere: Optional[str]
        mixing_device: Optional[str]
        mixing_media: Optional[str]

    type: str  # Type of the operation as classified in the pipeline
    token: str  # Token(word) of the operation as written in paper
    conditions: Conditions

## Reaction entry


class ReactionEntry(NamedTuple):
    doi: str  # DOI of the paper
    paragraph_string: str  # Paragraph text excerpt, max 100 characters.
    synthesis_type: str  # Type of synthesis as classified in the pipeline

    reaction_string: str  # Reaction formula
    reaction: Formula  # Dictionary containing parsed materials/amounts
    targets_string: List[str]  # List of synthesized target compositions
    target: Material  # Target material
    precursors: List[Material]  # List of precursor materials

    operations: List[Operation]  # List of operations
