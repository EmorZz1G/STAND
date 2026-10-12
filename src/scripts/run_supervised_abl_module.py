import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

import subprocess

def run_stand_abl_module(model_name, dataset):
    cmd = ['python', 'supervised_abl_module.py', '--model_name', model_name, '--dataset_name', dataset, '--index', '1',
                       '--task_name', 'supervised', '--train_test_split', '0.5', '--epochs', '10', '--win_size', '32']

    handle = subprocess.run(cmd)    
    if handle.returncode != 0:
        raise RuntimeError(f'Error occurred while running {model_name} on {dataset} with index 1')

if __name__ == '__main__':
    dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
    models = ['STAND_Attn', 'STAND_GRU']
    # models = ['STAND']
    # dataset_list = ['PSM']
    for dataset in dataset_list:
        for model in models:
            run_stand_abl_module(model, dataset)