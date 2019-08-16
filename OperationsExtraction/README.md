# Operations Extractor

 * classifies words tokens into follwoing types:

    *NotOperation*, *StartingSynthesis*, *Mixing*, *Heating*, *Drying*, *Shaping*, *Cooling*

 * extracts firing temperatures
 
### Installation:
```
git clone https://github.com/CederGroupHub/OperationsExtraction.git
cd OperationsExtraction
python setup.py install
```

### Initilization:
```
from operations_extractor import OperationsExtractor

w2v_model = 'path-to-folder/models/w2v_embeddings_lemmas_v3'
classifier_model = 'path-to-folder/models/fnn-model-1_7classes_dense32_perSentence_3'
spacy_model = 'path-to-folder/models/SpaCy_updated_v1.model'

OC = OperationsExtractor(w2v_model, classifier_model, spacy_model)
```

### Functions:

 * **get_operations(sentence_tokens)**:

        finds operation tokens and classifies them

        :param sentence: list of sentence tokens
        :returns: list of operations tuples (token_id, operation_type) found in the sentence

 * **operations_correction(sentence_tokens, sentence_operations, parsed_tokens)**:

        reassignment of some rear operation terms
        this is fix due to small amount of training data

        :param sentence_tokens: list of sentence tokens (SpaCy tokens or strings)
        :param sentence_operations: list of tuples of operations as output by get_operations()
        :param parsed_tokens: True if paragraph sentences are given as tokens parsed by SpaCy (reduces computation time)
        :returns: updated list of operations

 * **find_aqueous_mixing(sentence_tokens, sentence_operations, parsed_tokens)**:

        assigns aqueous mixing to corresponding mixing operations

        :param sentence_tokens: list of sentence tokens (SpaCy tokens or strings)
        :param sentence_operations: list of tuples of operations as output by get_operations()
        :param parsed_tokens: True if paragraph sentences are given as tokens parsed by SpaCy (reduces computation time)
        :returns: updated list of operations
 
 * **operations_refinement(paragraph_sentences, parsed_tokens)**:

        refinement of operations with respect to entire paragraph
        this is fix due to small amount of training data

        :param paragraph_sentences: list of tuples (tokenized sentence, operations=get_operations output)
        :param parsed_tokens: True if paragraph sentences are given as tokens parsed by SpaCy (reduces computation time)
        :return: list of tuples (spacy_tokens, operations=get_operations output) with updated operations

### Example:
```
from text_cleanup import TextCleanUp
from pprint import pprint
from chemdataextractor.doc import Paragraph

from operations_extractor import OperationsExtractor
oe = OperationsExtractor()

tp = TextCleanUp()

text_sents = ["LiNixMn2−xO4 (x=0.05,0.1,0.3,0.5) samples were prepared in either an air or an O2 atmosphere by solid-state reactions.",
              "Mixtures of Li2CO3,MnCO3, and NiO were heated at 700°C for 24 to 48 h with intermittent grinding.",
              "All these samples were cooled to room temperature at a controlled rate of 1°C/min.",
              "Unless specifically stated, all the samples described below were prepared in an atmosphere of air."]

paragraph_data = []
for sent in text_sents:

    text = tp.cleanup_text(sent)
    sent_toks = [tok for sent in Paragraph(text).raw_tokens for tok in sent]
    operations, spacy_tokens = oe.get_operations(sent_toks)
    updated_operations = oe.operations_correction(spacy_tokens, operations, parsed_tokens=True)
    updated_operations = oe.find_aqueous_mixing(spacy_tokens, updated_operations, parsed_tokens=True)
    paragraph_data.append((spacy_tokens, updated_operations))

paragraph_data_upd = oe.operations_refinement(paragraph_data, parsed_tokens=True)

pprint(paragraph_data_upd)
```