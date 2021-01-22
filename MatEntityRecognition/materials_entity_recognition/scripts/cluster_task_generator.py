# -*- coding: utf-8 -*-

import os
import shutil
import re
import json

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

def get_brc_submit_head(partition='GPU'):
    brc_submit_head = '''#!/bin/bash
# Job name:
#SBATCH --job-name=MER
#
# Account:
#SBATCH --account=fc_ceder
#
'''
    if partition == 'CPU':
        brc_submit_head += '''# Partition:
#SBATCH --partition=savio2
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=24
#
'''
    else:
        brc_submit_head += '''# Partition:
#SBATCH --partition=savio2_gpu
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=4
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:4
#
'''
    brc_submit_head += '''# Wall clock limit:
#SBATCH --time=48:00:00
#
## Command(s) to run (example):

source activate py36_tf
echo begin >> log.txt
date >> log.txt

'''
    return brc_submit_head

def get_brc_submit_body(task_dir_name, command, capability_per_work_dir):
    brc_submit_body = '''cd ''' + task_dir_name + '''
numRunning=$(ps aux | grep train -c)
while [ $numRunning -ge ''' + str(capability_per_work_dir+1) + ''' ]
 do
    sleep 600
    numRunning=$(ps aux | grep train -c)
done
date &>> result.txt
''' + command + ''' &>> result.txt &
sleep 60
cd ..

'''
    return brc_submit_body

def get_brc_submit_foot():
    brc_submit_foot = '''wait
echo allJobsCompleted >> log.txt
date >> log.txt
source deactivate

'''
    return brc_submit_foot

def get_lrc_submit_head(partition='GPU'):
    submit_head = '''#!/bin/bash
# Job name:
#SBATCH --job-name=MER
#
# Account:
#SBATCH --account=ac_ceder
#
'''
    if partition == 'CPU':
        submit_head += '''# Partition:
#SBATCH --partition=lr6
#
# QoS:
#SBATCH --qos=lr_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=32
#
'''
    else:
        submit_head += '''# Partition:
#SBATCH --partition=es1
#
# QoS:
#SBATCH --qos=es_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=4
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:4
#
'''
    if partition == 'CPU':
        submit_head += '''# Wall clock limit:
#SBATCH --time=48:00:00
#
## Command(s) to run (example):

source activate py36
echo begin >> log.txt
date >> log.txt

'''
    else:
        submit_head += '''# Wall clock limit:
#SBATCH --time=72:00:00
#
## Command(s) to run (example):

module load cuda/10.1
source activate py36
echo begin >> log.txt
date >> log.txt

'''
    return submit_head

def get_lrc_submit_body(task_dir_name, command, capability_per_work_dir):
    submit_body = '''cd ''' + task_dir_name + '''
numRunning=$(ps aux | grep train -c)
while [ $numRunning -ge ''' + str(capability_per_work_dir+1) + ''' ]
 do
    sleep 600
    numRunning=$(ps aux | grep train -c)
done
date &>> result.txt
''' + command + ''' &>> result.txt &
sleep 60
cd ..

'''
    return submit_body

def get_lrc_submit_foot():
    submit_foot = '''wait
echo allJobsCompleted >> log.txt
date >> log.txt
source deactivate

'''
    return submit_foot

def get_ginar_submit_head(partition='CPU'):
    ginar_submit_head = '''#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -N MER
#$ -m es
#$ -V
#$ -M tanjin_he@berkeley.edu
#$ -pe impi 16
#$ -o testjob_out
#$ -e testjob_er
#$ -S /bin/bash

conda activate py36_tf
echo begin >> log.txt
date >> log.txt

'''
    return ginar_submit_head

def get_ginar_submit_body(task_dir_name, command, capability_per_work_dir):
    ginar_submit_body = '''cd ''' + task_dir_name + '''
numRunning=$(ps aux | grep train -c)
while [ $numRunning -ge ''' + str(capability_per_work_dir+1) + ''' ]
 do
    sleep 600
    numRunning=$(ps aux | grep train -c)
done
date &>> result.txt
''' + command + ''' &>> result.txt &
sleep 60
cd ..

'''
    return ginar_submit_body

def get_ginar_submit_foot():
    ginar_submit_foot = '''wait
echo allJobsCompleted >> log.txt
date >> log.txt
source deactivate

'''
    return ginar_submit_foot

def get_xsede_submit_head(partition='GPU'):
    if partition == 'GPU-small':
        xsede_submit_head = '''#!/bin/bash
#SBATCH -p GPU-small
#SBATCH --ntasks-per-node 16
#SBATCH -t 7:59:00
#SBATCH --gres=gpu:p100:2

'''
    else:
        xsede_submit_head = '''#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH --ntasks-per-node 16
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:p100:2

'''
    xsede_submit_head += '''# echo commands to stdout
set -x

# load modules
module load cuda/10.1
source activate py36

log_name="log.txt"
echo begin > $log_name
date >> $log_name

'''

    return xsede_submit_head

def get_xsede_submit_body(task_dir_name, command, capability_per_work_dir):
    xsede_submit_body = '''cd ''' + task_dir_name + '''
numRunning=$(ps aux | grep train -c)
while [ $numRunning -ge ''' + str(capability_per_work_dir+1) + ''' ]
do
    sleep 600
    numRunning=$(ps aux | grep train -c)
done
date &>> result.txt
''' + command + ''' &>> result.txt &
sleep 60
cd ..

'''
    return xsede_submit_body

def get_xsede_submit_foot():
    xsede_submit_foot = '''wait
echo allJobsCompleted >> $log_name
date >> $log_name
source deactivate


'''
    return xsede_submit_foot



def get_cori_submit_head():
    xsede_submit_head = '''#!/bin/bash
# Job name:
#SBATCH -J MER
#
# QoS:
#SBATCH --qos=regular
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --tasks-per-node=32
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
# Node specification
#SBATCH --constraint=haswell
#
'''
    xsede_submit_head += '''## Command(s) to run (example):

conda activate py36
export PYTHONIOENCODING="UTF-8"
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

log_name="log.out"

echo begin >> $log_name
date >> $log_name

'''

    return xsede_submit_head

def get_cori_submit_body(task_dir_name, command, capability_per_work_dir):
    xsede_submit_body = '''cd ''' + task_dir_name + '''
numRunning=$(ps aux | grep train -c)
while [ $numRunning -ge ''' + str(capability_per_work_dir+1) + ''' ]
do
    sleep 600
    numRunning=$(ps aux | grep train -c)
done
date &>> result.txt
''' + command + ''' &>> result.txt &
sleep 60
cd ..

'''
    return xsede_submit_body

def get_cori_submit_foot():
    xsede_submit_foot = '''wait
echo allJobsCompleted >> $log_name
date >> $log_name
conda deactivate


'''
    return xsede_submit_foot

def get_bash_head():
    # only bash can use the MY_PATH part
    # cluster script cannot use the same code
    # because the job managers will cp the job script to another folder
    # and by no meaens we can utilize the location of the job script
    # there are only two ways if we want to control path in batch
    # scripts correctly:
    # 1. always submit batch script in the script folder
    # 2. use absolute path
    bash_head = '''#!/bin/bash

MY_PATH="`dirname \"$0\"`"              # relative
MY_PATH="`( cd \"$MY_PATH\" && pwd )`"  # absolutized and normalized
cd $MY_PATH

'''
    return bash_head


def get_submit_all(work_dirs, mode):
    assert mode in {'brc', 'xsede', 'ginar', 'lrc', 'cori'}
    script = get_bash_head()
    exec_file_dict = {
        'brc': 'brc_submit.sh',
        'xsede': 'xsede_submit.sh',
        'ginar': 'ginar_submit.sh',
        'lrc': 'lrc_submit.sh',
        'cori': 'cori_submit.sh',
    }
    exec_type_dict = {
        'brc': 'sbatch',
        'xsede': 'sbatch',
        'ginar': 'qsub',
        'lrc': 'sbatch',
        'cori': 'sbatch',
    }
    exec_file = exec_file_dict[mode]
    exec_type = exec_type_dict[mode]
    for dir_name in work_dirs:
        if exec_type == 'qsub':
            script += '''cd ''' + dir_name + '''
qsub ''' + exec_file + '''
echo "''' + dir_name + ''' submitted"
cd ..
sleep 10
        
'''
        else:
            script += '''cd ''' + dir_name + '''
sbatch ''' + exec_file + '''
echo "''' + dir_name + ''' submitted"
cd ..
sleep 10

'''
    return script

def get_submit_queue(mode):
    assert mode in {'xsede-small'}
    script = get_bash_head()
    exec_file_dict = {
        'xsede-small': 'xsede_submit_small.sh',
    }
    exec_file = exec_file_dict[mode]
    script += '''    
WORK='^[0-9]+.*[0-9]+/$' 
QUEUE_PATH=$(pwd)
while true ; do
	numSubbed=$(squeue -u tanjinhe | grep GPU-small -c)
	echo "number of GPU-small tasks submitted: $numSubbed" 

	for d in */ ; do
		if [[ ! $d =~ $WORK ]] ; then
			continue
		fi	
		numSubbed=$(squeue -u tanjinhe | grep GPU-small -c)
		if [[ $numSubbed<2 ]]; then
			mv $d ..			
			cd ../${d}
			sbatch ''' + exec_file + '''
			echo "$d submitted"
            cd ${QUEUE_PATH}
			sleep 3
		fi
	done		
	sleep 600
done
    
    '''
    return script

def load_task_meta(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as fr:
            data = json.load(fr)
    else:
        data = {
            'task_index': 0,
            'dir_index': 0,
        }
    return data

def save_task_meta(task_meta, file_path):
    with open(file_path, 'w') as fw:
        json.dump(task_meta, fw, indent=2)

def copy_code(code_src, code_des):
    file_list = os.listdir(code_src)
    shutil.copytree(
        os.path.join(code_src, 'scripts'),
        os.path.join(code_des, 'scripts')
    )
    for f in file_list:
        if f.endswith('.py'):
            shutil.copyfile(
                os.path.join(code_src, f),
                os.path.join(code_des, f)
            )

def add_data_args(hyper_paras, datasets, data_prefix):
    expanded_args = []
    for arg in hyper_paras:
        for data_folder in datasets:
            data_files = os.listdir(data_folder)
            new_arg = arg
            for tmp_f in data_files:
                if re.match('.*train.*', tmp_f):
                    new_arg += ' --path_train {}/{}/{} '.format(
                        data_prefix,
                        os.path.basename(data_folder),
                        tmp_f
                    )
                if re.match('.*dev.*', tmp_f):
                    new_arg += ' --path_dev {}/{}/{} '.format(
                        data_prefix,
                        os.path.basename(data_folder),
                        tmp_f
                    )
                if re.match('.*test.*', tmp_f):
                    new_arg += ' --path_test {}/{}/{} '.format(
                        data_prefix,
                        os.path.basename(data_folder),
                        tmp_f
                    )
            expanded_args.append(new_arg)
    return expanded_args

if __name__ == '__main__':
    # --------------input region-----------------------
    work_dir_template = '0000_hyper_structure'
    xsede_partition = 'GPU'
    savio_partition = 'GPU'
    lrc_partition = 'GPU'
    cori_partition = 'GPU'

    # to submit different spliting
    # TP data:
    #     datasets = ['../dataset/TP_750_1/TP_750_1_{:02d}'.format(i) for i in range(0, 12)]
    #   --emb_path ../../data_public/embedding/embedding_MAT_combine_sg_win5_size100_iter50_noLemma_4.text
    #   --emb_path ../../data_public/embedding/embedding_MAT_sg_win5_size100_iter50_noLemma.text
    # MAT mebeeding:
    #
    #   --emb_path ../../data_public/embedding/embedding_sg_win5_size100_iter50_noLemma_4.text
    #   --emb_path ../../data_public/embedding/embedding_sg_win5_size100_iter50_noLemma.text
    # abs
    # datasets = ['../dataset/MAT_750_83_241_1/MAT_750_83_241_1_{:02d}'.format(i) for i in range(0, 12)]

    to_add_data_args = True
    all_name_prefix = [
        'MAT_750_83_173_232_345_321_438_5',
        'MATTP_750_83_173_232_345_321_438_5',
        'TP_750_83_173_232_345_321_438_5',
    ]

    hyper_paras_essential = [
        '--std_out generated/output.txt '
        '--bert_path ../../data_public/bert/cased_L-12_H-768_A-12 --bert_first_trainable_layer 6 '
        '--word_dim 0 --char_dim 0 --emb_path None --singleton_unk_probability 0.0 '
        '--tag_scheme iob '
        '--num_epochs 25 --batch_size 16 '
        '--classifier_type lstm --crf True '
        '--dropout 0.5 '
        '--lr_method adamdecay@lr=1e-05@epsilon=1e-08@warmup=0.1 '
        '--loss_per_token True ',

        '--std_out generated/output.txt '
        '--bert_path ../../data_public/bert/MatBERT_20201120 --bert_first_trainable_layer 6 '
        '--word_dim 0 --char_dim 0 --emb_path None --singleton_unk_probability 0.0 '
        '--tag_scheme iob '
        '--num_epochs 25 --batch_size 16 '
        '--classifier_type lstm --crf True '
        '--dropout 0.5 '
        '--lr_method adamdecay@lr=1e-05@epsilon=1e-08@warmup=0.1 '
        '--loss_per_token True ',

    ]

    # obtain final_hyper_paras
    final_hyper_paras = []
    for name_prefix in all_name_prefix:
        datasets = [
            '../dataset/{name_prefix}/{name_prefix}_{id:02d}'.format(
                name_prefix=name_prefix, id=i,
            )
            for i in range(0, 12)
        ]
        print('len(datasets) of {}: {}'.format(name_prefix, len(datasets)))
        data_prefix = '../../data_public/{name_prefix}'.format(name_prefix=name_prefix)

        # a benchmark
        # task time avg_time
        # 4  4:11:47.491207
        # 6  4:40:14.187977
        # 8  4:59:29.079169 5/8
        # 12 5:59:51.086202 6/12
        # 24 10:20:54.829366
        capability_per_work_dir = 4
        task_per_work_dir = 4
        if to_add_data_args:
            hyper_paras = add_data_args(
                hyper_paras=hyper_paras_essential,
                datasets=datasets,
                data_prefix=data_prefix,
            )
        else:
            hyper_paras = hyper_paras_essential
        final_hyper_paras.extend(hyper_paras)

    # -----------------task generation----------------------------
    template_folder = os.path.abspath('..')
    task_meta_file = os.path.abspath('../generated/task_meta.json')
    work_dir_root = os.path.abspath('../../MER_cluster_tasks')
    work_dirs = set()
    if os.path.exists(work_dir_root):
        shutil.rmtree(work_dir_root)
    os.makedirs(work_dir_root)
    os.chdir(work_dir_root)

    # load task meta
    task_meta = load_task_meta(task_meta_file)
    task_index = task_meta['task_index']
    dir_index = task_meta['dir_index']
    task_dir_template = 'MER'

    print('len(final_hyper_paras)', len(final_hyper_paras))
    for i in range(len(final_hyper_paras)):
        work_para = final_hyper_paras[i]
        task_index += 1

        # create work_dir and submit script
        if i%task_per_work_dir == 0:
            dir_index += 1
            work_dir_name = '{}_{:04d}'.format(work_dir_template, dir_index)
            work_dir = os.path.join(work_dir_root, work_dir_name)
            os.mkdir(work_dir)
            os.chdir(work_dir)
            work_dirs.add(work_dir_name)
            # write head
            fw = open('brc_submit.sh', 'w', newline='\n')
            fw2 = open('xsede_submit.sh', 'w', newline='\n')
            fw3 = open('xsede_submit_small.sh', 'w', newline='\n')
            fw4 = open('ginar_submit.sh', 'w', newline='\n')
            fw5 = open('lrc_submit.sh', 'w', newline='\n')
            fw6 = open('cori_submit.sh', 'w', newline='\n')
            fw.write(get_brc_submit_head(partition=savio_partition))
            fw2.write(get_xsede_submit_head(partition=xsede_partition))
            fw3.write(get_xsede_submit_head(partition='GPU-small'))
            fw4.write(get_ginar_submit_head())
            fw5.write(get_lrc_submit_head(partition=lrc_partition))
            fw6.write(get_cori_submit_head())
            tasks = []

        # copy codes and write body
        task_dir_name = '{}_{:06d}'.format(task_dir_template, task_index)
        task_dir = os.path.join(task_dir_name)
        copy_code(template_folder, task_dir)
        command = '''python scripts/train.py ''' + work_para
        fw.write(get_brc_submit_body(
            task_dir_name,
            command,
            capability_per_work_dir
        ))
        fw2.write(get_xsede_submit_body(
            task_dir_name,
            command,
            capability_per_work_dir
        ))
        fw3.write(get_xsede_submit_body(
            task_dir_name,
            command,
            capability_per_work_dir
        ))
        fw4.write(get_ginar_submit_body(
            task_dir_name,
            command,
            capability_per_work_dir
        ))
        fw5.write(get_lrc_submit_body(
            task_dir_name,
            command,
            capability_per_work_dir
        ))
        fw6.write(get_cori_submit_body(
            task_dir_name,
            command,
            capability_per_work_dir
        ))
        tasks.append({
            'task_dir_name': task_dir_name,
            'command': command
        })

        # write foot
        if (i+1)%task_per_work_dir == 0 or (i+1) == len(final_hyper_paras):
            fw.write(get_brc_submit_foot())
            fw.close()
            fw2.write(get_xsede_submit_foot())
            fw2.close()
            fw3.write(get_xsede_submit_foot())
            fw3.close()
            fw4.write(get_ginar_submit_foot())
            fw4.close()
            fw5.write(get_lrc_submit_foot())
            fw5.close()
            fw6.write(get_cori_submit_foot())
            fw6.close()
            with open('task_spec.json', 'w') as fw:
                json.dump(tasks, fw, indent=2)
            os.chdir(work_dir_root)

    task_meta['task_index'] = task_index
    task_meta['dir_index'] = dir_index
    save_task_meta(task_meta, task_meta_file)

    # generate script to submit cluster scripts
    work_dirs = sorted(work_dirs)
    with open('brc_submit_all.sh', 'w', newline='\n') as fw:
        fw.write(get_submit_all(work_dirs, mode='brc'))
    with open('xsede_submit_all.sh', 'w', newline='\n') as fw2:
        fw2.write(get_submit_all(work_dirs, mode='xsede'))
    with open('xsede_submit_squeue.sh', 'w', newline='\n') as fw3:
        fw3.write(get_submit_queue(mode='xsede-small'))
    with open('ginar_submit_all.sh', 'w', newline='\n') as fw4:
        fw4.write(get_submit_all(work_dirs, mode='ginar'))
    with open('lrc_submit_all.sh', 'w', newline='\n') as fw5:
        fw5.write(get_submit_all(work_dirs, mode='lrc'))
    with open('cori_submit_all.sh', 'w', newline='\n') as fw6:
        fw6.write(get_submit_all(work_dirs, mode='cori'))
    os.chdir(os.path.join(work_dir_root, '..'))
    shutil.make_archive(
        '{}'.format(os.path.basename(work_dir_root)),
        'zip',
        work_dir_root
    )
    shutil.move('{}.zip'.format(os.path.basename(work_dir_root)), work_dir_root)
