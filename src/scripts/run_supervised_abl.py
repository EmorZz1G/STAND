import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

# model_list = ['RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
model_list = ['RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
# model_list = ['SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
# model_list = ['ExtraTrees', 'LightGBM']
# model_list = ['LightGBM']
dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
# dataset_list = ['SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
# dataset_list = ['WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']

import subprocess

extra_config = {
    'ExtraTrees': {'win_size': 1},
    'RF': {'win_size': 1},
}

def run_baselines(model_list, dataset_list, extra_config, train_test_split=0.5):
    for model in model_list:
        for dataset in dataset_list:
            if dataset == 'UCR':
                index_range = range(1, 251)
                for idx in index_range:
                    print(f'Running experiment for model: {model}, dataset: {dataset} with index: {idx}')
                    cmd = ['python', 'supervised.py', '--model_name', model, '--dataset_name', dataset, '--index', str(idx), 
                           '--task_name', 'supervised', '--train_test_split', str(train_test_split)]
                    if model in extra_config:
                        for key, value in extra_config[model].items():
                            cmd += [f'--{key}', str(value)]
                    handle = subprocess.run(cmd)
                    if handle.returncode != 0:
                        raise RuntimeError(f'Error occurred while running {model} on {dataset} with index {idx}')
            else:
                print(f'Running experiment for model: {model}, dataset: {dataset}')
                cmd = ['python', 'supervised.py', '--model_name', model, '--dataset_name', dataset, '--task_name', 'supervised',
                       '--train_test_split', str(train_test_split)]
                handle = subprocess.run(cmd)
                if handle.returncode != 0:
                    raise RuntimeError(f'Error occurred while running {model} on {dataset}')

def run_stand(dataset_list, train_test_split=0.5, win_size=32, d_model=32, num_layers=1, bidirectional=0):
    if isinstance(dataset_list, str):
        dataset_list = [dataset_list]
    elif isinstance(dataset_list, list):
        print(f'Running STAND on datasets: {dataset_list}')
    else:
        raise ValueError('dataset_list should be a string or a list of strings.')
    for dataset in dataset_list:
        if dataset == 'UCR':
            index_range = range(1, 251)
            for idx in index_range:
                print(f'Running experiment for model: STAND, dataset: {dataset} with index: {idx}')
                cmd = ['python', 'supervised.py', '--model_name', 'STAND', '--dataset_name', dataset, '--index', str(idx),
                       '--task_name', 'supervised', '--train_test_split', str(train_test_split), '--win_size', str(win_size),
                       '--d_model', str(d_model), '--num_layers', str(num_layers), '--bidirectional', str(bidirectional)]
                subprocess.run(cmd)
        else:
            print(f'Running experiment for model: STAND, dataset: {dataset}')
            cmd = ['python', 'supervised.py', '--model_name', 'STAND', '--dataset_name', dataset,
                   '--task_name', 'supervised',
                   '--train_test_split', str(train_test_split), '--win_size', str(win_size),
                   '--d_model', str(d_model), '--num_layers', str(num_layers), '--bidirectional', str(bidirectional)]
            subprocess.run(cmd)

if __name__ == '__main__':
    # train_test_split_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    # train_test_split_list = [0.2, 0.3, 0.4, 0.5]
    train_test_split_list = [0.5]
    win_size_list = [32,64,128]
    d_model_list = [16,32,64,128,256]
    num_layers_list = [0,1,2,3]
    bidirectional_list = [0,1]
    from itertools import product
    # task_list_generator = product(train_test_split_list, win_size_list, d_model_list, num_layers_list, bidirectional_list)
    task_list_generator = product(dataset_list, train_test_split_list, win_size_list, d_model_list, num_layers_list, bidirectional_list)
    task_list_generator = list(task_list_generator)
    # 打乱
    import random
    random.shuffle(task_list_generator)
    parallel_jobs = 3

    exp_log_pth = pathlib.Path(__file__).parent.parent.parent / 'logs' / 'supervised_exp.csv'
    if not exp_log_pth.exists():
        print(f'Log file {exp_log_pth} does not exist. Please run baselines first.')
        raise FileNotFoundError(f'Log file {exp_log_pth} does not exist. Please run baselines first.')
    import pandas as pd
    exp_log = pd.read_csv(exp_log_pth)
    completed_tasks = set()
    for _, row in exp_log.iterrows():
        if row['model_name'] == 'STAND':
            completed_tasks.add((row['dataset_name'], row['train_test_split'], row['win_size'], row['d_model'], row['num_layers'], row['bidirectional']))
    if parallel_jobs == 0:
        for dataset_name, train_test_split, win_size, d_model, num_layers, bidirectional in task_list_generator:
            if (dataset_name, train_test_split, win_size, d_model, num_layers, bidirectional) in completed_tasks:
                print(f"Skipping completed task: {train_test_split}, {win_size}, {d_model}, {num_layers}, {bidirectional}")
                continue
            print(f'Running STAND with train_test_split: {train_test_split}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
            # run_stand(dataset_list, train_test_split, win_size, d_model, num_layers, bidirectional)
            run_stand(dataset_name, train_test_split, win_size, d_model, num_layers, bidirectional)
    elif parallel_jobs > 0:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = []
            for dataset_name, train_test_split, win_size, d_model, num_layers, bidirectional in task_list_generator:
                if (dataset_name, train_test_split, win_size, d_model, num_layers, bidirectional) in completed_tasks:
                    print(f"Skipping completed task: {dataset_name}, {train_test_split}, {win_size}, {d_model}, {num_layers}, {bidirectional}")
                    continue
                print(f'Submitting STAND with train_test_split: {train_test_split}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
                # futures.append(executor.submit(run_stand, dataset_list, train_test_split, win_size, d_model, num_layers, bidirectional))
                futures.append(executor.submit(run_stand, dataset_name, train_test_split, win_size, d_model, num_layers, bidirectional))
            total = len(futures)
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                print(f"\033[34m已完成任务数: {completed}/{total}\033[0m")
                try:
                    future.result()
                except Exception as e:
                    print(f'Error occurred during execution: {e}')
    elif parallel_jobs == -1:
        train_test_split, win_size, d_model, num_layers, bidirectional = 0.5, 32, 32, 1, 0
        print(f'Running STAND with train_test_split: {train_test_split}, win_size: {win_size}, d_model: {d_model}, num_layers: {num_layers}, bidirectional: {bidirectional}')
        run_stand(dataset_list, train_test_split, win_size, d_model, num_layers, bidirectional)