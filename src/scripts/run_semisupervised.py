import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

# model_list = ['OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 
                        # 'AnomalyTransformer', 'TimesNet']
# model_list = ['AutoEncoder', 'CNN', 'LSTMAD', 'USAD', 'OmniAnomaly', 
#                         'AnomalyTransformer', 'TimesNet']
model_list = ['USAD', 'OmniAnomaly', 
                        'AnomalyTransformer', 'TimesNet']
model_list = ['TimesNet']
model_list = ['M2N2']
model_list = ['LFTSAD']
model_list = ['CATCH']
dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']# 'UCR']


import subprocess

def run_baselines():
    for model in model_list:
        for dataset in dataset_list:
            if dataset == 'UCR':
                index_range = range(1, 251)
                for idx in index_range:
                    print(f'Running experiment for model: {model}, dataset: {dataset} with index: {idx}')
                    cmd = ['python', 'semisupervised.py', '--model_name', model, '--dataset_name', dataset, '--index', str(idx), 
                           '--task_name', 'semisupervised']
                    handle = subprocess.run(cmd)
                    if handle.returncode != 0:
                        raise RuntimeError(f'Error occurred while running {model} on {dataset} with index {idx}')
            else:
                print(f'Running experiment for model: {model}, dataset: {dataset}')
                cmd = ['python', 'semisupervised.py', '--model_name', model, '--dataset_name', dataset, '--task_name', 'semisupervised']
                handle = subprocess.run(cmd)
                if handle.returncode != 0:
                    raise RuntimeError(f'Error occurred while running {model} on {dataset}')


if __name__ == '__main__':
    run_baselines()