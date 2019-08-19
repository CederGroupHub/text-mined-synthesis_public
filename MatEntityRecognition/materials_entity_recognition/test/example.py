import json
from pprint import pprint

from materials_entity_recognition import MatRecognition
from materials_entity_recognition import MatIdentification

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

if __name__ == "__main__":
	# load data
	with open('../data/test_paras.json', 'r') as fr:
		paras = json.load(fr)

	# # find materials
	# model_new = MatIdentification()
	# for tmp_para in paras:
	# 	all_materials = model_new.mat_identify(tmp_para)

	# find targets/precursors
	model_new = MatRecognition()
	for tmp_para in paras:
		all_materials, precursors, targets, other_materials = model_new.mat_recognize(tmp_para)



