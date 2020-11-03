import json
import random
import chemdataextractor as CDE

from materials_entity_recognition import MatRecognition
from materials_entity_recognition import MatIdentification

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

# random.seed(datetime.now())
random.seed(7)


if __name__ == "__main__":
    # load data
    with open('../data/test_paras.json', 'r') as fr:
        paras = json.load(fr)

    # # find materials
    # model_new = MatIdentification()
    # for tmp_para in paras:
    #   all_materials = model_new.mat_identify(tmp_para)

    # find targets/precursors
    model_new = MatRecognition()
    # # if enabling dependency parsing as feature
    # model_new = MatRecognition(parse_dependency=True)
    for tmp_para in paras:
        CDE_para = CDE.doc.Paragraph(tmp_para)
        pre_tokens = [tmp_sent.tokens for tmp_sent in CDE_para] 
        result = model_new.mat_recognize(tmp_para, pre_tokens=pre_tokens)

