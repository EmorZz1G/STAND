import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

model_list = ['KNN', 'LR', 'RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']

import subprocess

def run_baselines(train_test_split=0.5):
    for model in model_list:
        for dataset in dataset_list:
            if dataset == 'UCR':
                index_range = range(1, 251)
                for idx in index_range:
                    print(f'Running experiment for model: {model}, dataset: {dataset} with index: {idx}')
                    cmd = ['python', 'supervised.py', '--model_name', model, '--dataset_name', dataset, '--index', str(idx), 
                           '--task_name', 'supervised', '--train_test_split', train_test_split]
                    handle = subprocess.run(cmd)
                    if handle.returncode != 0:
                        raise RuntimeError(f'Error occurred while running {model} on {dataset} with index {idx}')
            else:
                print(f'Running experiment for model: {model}, dataset: {dataset}')
                cmd = ['python', 'supervised.py', '--model_name', model, '--dataset_name', dataset, '--task_name', 'supervised',
                       '--train_test_split', train_test_split]
                handle = subprocess.run(cmd)
                if handle.returncode != 0:
                    raise RuntimeError(f'Error occurred while running {model} on {dataset}')

def run_stand(train_test_split=0.5, win_size=32):
    for dataset in dataset_list:
        if dataset == 'UCR':
            index_range = range(1, 251)
            for idx in index_range:
                print(f'Running experiment for model: STAND, dataset: {dataset} with index: {idx}')
                cmd = ['python', 'supervised.py', '--model_name', 'STAND', '--dataset_name', dataset, '--index', str(idx),
                       '--task_name', 'supervised', '--train_test_split', train_test_split, '--win_size', win_size]
                subprocess.run(cmd)
        else:
            print(f'Running experiment for model: STAND, dataset: {dataset}')
            cmd = ['python', 'supervised.py', '--model_name', 'STAND', '--dataset_name', dataset,
                   '--task_name', 'supervised',
                   '--train_test_split', train_test_split, '--win_size', win_size]
            subprocess.run(cmd)

if __name__ == '__main__':
    train_test_split_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for train_test_split in train_test_split_list:
        run_baselines(train_test_split)
        run_stand(train_test_split)