# coding=utf-8
from text_cleanup import TextCleanUp
from pprint import pprint
from chemdataextractor.doc import Paragraph

from operations_extractor import OperationsExtractor
oe = OperationsExtractor()

tp = TextCleanUp()

text_sents = ["LiNixMn2−xO4 (x=0.05,0.1,0.3,0.5) samples were prepared in either an air or an O2 atmosphere by solid-state reactions.",
              "Mixtures of Li2CO3,MnCO3, and NiO were heated at 700°C for 24 to 48 h with intermittent grinding.",
              "All these samples were cooled to room temperature at a controlled rate of 1°C/min."]

paragraph_data = []
for sent in text_sents:

    text = tp.cleanup_text(sent)
    sent_toks = [tok for sent in Paragraph(text).raw_tokens for tok in sent]
    # output, sentence, tokens = get_operations(sent)
    operations, spacy_tokens = oe.get_operations(sent_toks)
    updated_operations = oe.operations_correction(spacy_tokens, operations, parsed_tokens=True)
    updated_operations = oe.find_aqueous_mixing(spacy_tokens, updated_operations, parsed_tokens=True)
    paragraph_data.append((spacy_tokens, updated_operations))

paragraph_data_upd = oe.operations_refinement(paragraph_data, parsed_tokens=True)

pprint(paragraph_data_upd)