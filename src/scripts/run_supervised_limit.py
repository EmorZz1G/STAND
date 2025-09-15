import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

model_list = ['KNN', 'LR', 'RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']

import subprocess

def run_baselines(anomaly_ratio=0.1):
    for model in model_list:
        for dataset in dataset_list:
            print(f'Running experiment for model: {model}, dataset: {dataset}')
            cmd = ['python', 'supervised.py', '--model_name', model, '--dataset_name', dataset, '--task_name', 'supervised_limit',
                    '--anomaly_ratio', anomaly_ratio]
            handle = subprocess.run(cmd)
            if handle.returncode != 0:
                raise RuntimeError(f'Error occurred while running {model} on {dataset}')

def run_stand(anomaly_ratio=0.1):
    for dataset in dataset_list:
        print(f'Running experiment for model: STAND, dataset: {dataset}')
        cmd = ['python', 'supervised.py', '--model_name', 'STAND', '--dataset_name', dataset, '--task_name', 'supervised_limit',
                '--anomaly_ratio', anomaly_ratio]
        subprocess.run(cmd)

if __name__ == '__main__':
    anomaly_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for anomaly_ratio in anomaly_ratio_list:
        run_baselines(anomaly_ratio)
        run_stand(anomaly_ratio)