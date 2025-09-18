import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

# model_list = ['IForest', 'LOF', 'POLY', 'MatrixProfile', 'PCA', 'HBOS', 'KNN', 'KMeansAD', 'KShapeAD', 'Random']
# model_list = ['MatrixProfile', 'PCA', 'HBOS', 'KNN', 'KMeansAD', 'KShapeAD', 'Random']
model_list = ['PCA', 'HBOS', 'KNN', 'KMeansAD', 'KShapeAD', 'Random']
dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']

extra_config = {
    'POLY': {'win_size': 1},
}

import subprocess

def run_baselines():
    for model in model_list:
        for dataset in dataset_list:
            if dataset == 'UCR':
                index_range = range(1, 251)
                for idx in index_range:
                    print(f'Running experiment for model: {model}, dataset: {dataset} with index: {idx}')
                    cmd = ['python', 'unsupervised.py', '--model_name', model, '--dataset_name', dataset, '--index', str(idx), 
                           '--task_name', 'unsupervised']
                    handle = subprocess.run(cmd)
                    if handle.returncode != 0:
                        raise RuntimeError(f'Error occurred while running {model} on {dataset} with index {idx}')
            else:
                print(f'Running experiment for model: {model}, dataset: {dataset}')
                cmd = ['python', 'unsupervised.py', '--model_name', model, '--dataset_name', dataset, '--task_name', 'unsupervised']
                handle = subprocess.run(cmd)
                if handle.returncode != 0:
                    raise RuntimeError(f'Error occurred while running {model} on {dataset}')


if __name__ == '__main__':
    run_baselines()