import json
import random
import time
import multiprocessing as mp
import os

from materials_entity_recognition import MatRecognition
from materials_entity_recognition import MatIdentification

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

def save_results(results):
	if not os.path.exists('results'):
		os.mkdir('results')
	file_name = 'results/results_' + str(hash(str(results))) + '.json'  
	with open(file_name, 'w') as fw:
		json.dump(results, fw, indent=2)
	return file_name

def pipeline_in_one_thread(paragraphs, batch_size=100):
	"""
	if batch_size > 0, then save results into files which containing batch_size of paragraphs each
	results output would be list of names of files
	if batch_size == 0, then save results in memory and output together at last rather than saving 
	results into files 
	"""

	# find targets/precursors
	results = []
	batch_results = []

	# =======================================================
	# add initialization part of the pipeline here
	model_new = MatRecognition()
	# # if enabling dependency parsing as feature
	# model_new = MatRecognition(parse_dependency=True)
	# =======================================================

	for tmp_para in paragraphs:
		# =====================================================================================
		# add pipeline executing functions here
		all_materials, precursors, targets, other_materials = model_new.mat_recognize(tmp_para)
		batch_results.append({
						'paragraph': tmp_para, 
						'all_materials': all_materials,
						'precursors': precursors,
						'target': targets,
						'other_materials': other_materials
						})
		# =====================================================================================
		
		if batch_size > 0 and len(batch_results) >= batch_size:
			results.append(save_results(batch_results)) 
			batch_results = []
	if batch_size > 0:
		if len(batch_results) > 0:
			results.append(save_results(batch_results))
	else:
		results = batch_results
	return results

if __name__ == "__main__":
	# =======================================================
	# parallel work config
	num_cores = 8
	batch_size = 2
	# =======================================================

	# load data
	with open('../data/test_paras.json', 'r') as fr:
		paras = json.load(fr)

	# execute pipeline in a parallel way
	last_time_time = time.time()
	# running
	parallel_arguments = []
	len_per_para_list = int(len(paras)/num_cores)	
	for i in range(num_cores):
		if i < num_cores-1:
			parallel_arguments.append((paras[i*len_per_para_list: (i+1)*len_per_para_list], batch_size))
		else:
			parallel_arguments.append((paras[i*len_per_para_list: ], batch_size))		

	p = mp.Pool(processes = num_cores)
	results = p.starmap(pipeline_in_one_thread, parallel_arguments)
	p.close()
	p.join()

	# reading results
	print('time used:', time.time()-last_time_time)
	# combine all results
	all_results = sum(results, [])
	print('len(all_results)', len(all_results))
	print(all_results)

