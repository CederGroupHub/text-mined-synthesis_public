import os

from synthesis_api_hub import api_method
from synthesis_api_hub.apiegg import APIEgg
from synthesis_project_ceder.utils.environments import request_linear_algebra_single_threaded

from operations_extractor.operations_extractor import OperationsExtractor


class OperationsExtractorWorker(APIEgg):
    namespace = 'operation_extractor'
    version = '2018112600'

    def __init__(self):
        request_linear_algebra_single_threaded()

        my_path = os.path.dirname(os.path.realpath(__file__))
        w2v_model = os.path.join(my_path, 'models', 'w2v_embeddings_lemmas_v3')
        classifier_model = os.path.join(my_path, 'models', 'fnn-model-1_7classes_dense32_perSentence_3')
        spacy_model = os.path.join(my_path, 'models', 'SpaCy_updated_v1.model')

        self.oc = OperationsExtractor(w2v_model, classifier_model, spacy_model)

    @api_method
    def get_operations(self, sentence):
        output, spacy_tokens = self.oc.get_operations(sentence)
        return output

    @api_method
    def operations_correction(self, sentence_tokens, sentence_operations):
        return self.oc.operations_correction(sentence_tokens, sentence_operations, parsed_tokens=False)

    @api_method
    def find_aqueous_mixing(self, sentence_tokens, sentence_operations):
        return self.oc.find_aqueous_mixing(sentence_tokens, sentence_operations, parsed_tokens=False)

    @api_method
    def operations_refinement(self, paragraph_data):
        spacy_doc, operations = zip(*self.oc.operations_refinement(paragraph_data, parsed_tokens=False))
        return operations
