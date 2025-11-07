import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

# model_list = ['KNN', 'LR', 'RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
model_list = ['RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
# dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']
# dataset_list = ['PSM']
dataset_list = ['NIPS_TS_Swan']

import subprocess

def run_baselines(anomaly_ratio=0.1):
    for model in model_list:
        if model == 'RF' and anomaly_ratio == 0.1: continue
        # if model == 'SVM' and anomaly_ratio == 0.1: continue
        for dataset in dataset_list:
            print(f'Running experiment for model: {model}, dataset: {dataset}')
            cmd = ['python', 'supervised_limit.py', '--model_name', model, '--dataset_name', dataset, '--task_name', 'supervised_limit',
                    '--anomaly_ratio', str(anomaly_ratio)]
            handle = subprocess.run(cmd)
            if handle.returncode != 0:
                raise RuntimeError(f'Error occurred while running {model} on {dataset}')

def run_stand(dataset_list, anomaly_ratio=0.1, win_size=32, d_model=32, num_layers=1, bidirectional=0):
    for dataset in dataset_list:
        print(f'Running experiment for model: STAND, dataset: {dataset}')
        cmd = ['python', 'supervised_limit.py', '--model_name', 'STAND', '--dataset_name', dataset, '--task_name', 'supervised_limit',
                '--anomaly_ratio', str(anomaly_ratio), '--win_size', str(win_size), '--d_model', str(d_model),
                '--num_layers', str(num_layers), '--bidirectional', str(bidirectional)]
        subprocess.run(cmd)

if __name__ == '__main__':
    anomaly_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    dataset_list = ['PSM']
    # train_test_split_list = [0.5]
    win_size_list = [32,64,128]
    d_model_list = [16,32,64,128,256]
    num_layers_list = [0,1,2,3]
    bidirectional_list = [0,1]
    from itertools import product
    task_list_generator = product(anomaly_ratio_list, win_size_list, d_model_list, num_layers_list, bidirectional_list)
    task_list_generator = list(task_list_generator)


    parallel_jobs = 10
    if parallel_jobs == 0:
        for anomaly_ratio, win_size, d_model, num_layers, bidirectional in task_list_generator:
            print(f'Running STAND with anomaly_ratio: {anomaly_ratio}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
            run_stand(dataset_list, anomaly_ratio, win_size, d_model, num_layers, bidirectional)
    elif parallel_jobs > 0:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = []
            for anomaly_ratio, win_size, d_model, num_layers, bidirectional in task_list_generator:
                print(f'Submitting STAND with anomaly_ratio: {anomaly_ratio}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
                futures.append(executor.submit(run_stand, dataset_list, anomaly_ratio, win_size, d_model, num_layers, bidirectional))
            total = len(futures)
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                print(f"\033[34m已完成任务数: {completed}/{total}\033[0m")
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred during execution: {e}")
    elif parallel_jobs == -1:
        anomaly_ratio, win_size, d_model, num_layers, bidirectional = 0.5, 32, 32, 1, 0
        print(f'Running STAND with anomaly_ratio: {anomaly_ratio}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
        run_stand(dataset_list, anomaly_ratio, win_size, d_model, num_layers, bidirectional)

    anomaly_ratio_list = [0.1, 0.2]#, 0.3, 0.4, 0.5]
    dataset_list = ['NIPS_TS_Swan']
    task_list_generator = product(anomaly_ratio_list, win_size_list, d_model_list, num_layers_list, bidirectional_list)
    task_list_generator = list(task_list_generator)
    if parallel_jobs == 0:
        for anomaly_ratio, win_size, d_model, num_layers, bidirectional in task_list_generator:
            print(f'Running STAND with anomaly_ratio: {anomaly_ratio}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
            run_stand(dataset_list, anomaly_ratio, win_size, d_model, num_layers, bidirectional)
    elif parallel_jobs > 0:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = []
            for anomaly_ratio, win_size, d_model, num_layers, bidirectional in task_list_generator:
                print(f'Submitting STAND with anomaly_ratio: {anomaly_ratio}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
                futures.append(executor.submit(run_stand, dataset_list, anomaly_ratio, win_size, d_model, num_layers, bidirectional))
            total = len(futures)
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                print(f"\033[34m已完成任务数: {completed}/{total}\033[0m")
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred during execution: {e}")
    elif parallel_jobs == -1:
        anomaly_ratio, win_size, d_model, num_layers, bidirectional = 0.5, 32, 32, 1, 0
        print(f'Running STAND with anomaly_ratio: {anomaly_ratio}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
        run_stand(dataset_list, anomaly_ratio, win_size, d_model, num_layers, bidirectional)
