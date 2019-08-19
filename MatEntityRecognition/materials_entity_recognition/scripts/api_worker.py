from synthesis_api_hub import api_method
from synthesis_api_hub.apiegg import APIEgg

from materials_entity_recognition import MatRecognition


class MERWorker(APIEgg):
    namespace = 'MER'
    version = '2018121500'

    def __init__(self):
        self.model = MatRecognition()

    @api_method
    def mat_recognize(self, paragraph):
        all_materials, precursors, targets, other_materials = self.model.mat_recognize(paragraph)
        return all_materials, precursors, targets, other_materials
