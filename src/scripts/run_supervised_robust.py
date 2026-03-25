import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

# model_list = ['RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
model_list = ['RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM', 'STAND']
# model_list = ['SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
# model_list = ['ExtraTrees', 'LightGBM']
# model_list = ['LightGBM']
dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
# dataset_list = ['PSM', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
# dataset_list = ['SWAT']#, 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
# dataset_list = ['SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
# dataset_list = ['WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']

task_list = [
    # {
    #     'model_list': ["KNN"],
    #     'dataset_list': ['NIPS_TS_Water'],
    #     'extra_config': {'KNN': {'win_size': 1}}
    # },
    {
        'model_list': ["LR"],
        'dataset_list': ['NIPS_TS_Swan', 'NIPS_TS_Water'],
        'extra_config': {"LR": {'win_size': 2}, "RF": {'win_size': 1}}
    }
]

import subprocess

extra_config = {
    'ExtraTrees': {'win_size': 1},
    'RF': {'win_size': 1},
}
proj_pth = pathlib.Path(__file__).parent.parent.parent
logs_pth = proj_pth / 'logs'
logs_file_pth = logs_pth / 'supervised_robust_exp.csv'

def check_exist(model_name, dataset_name, noise_prob):
    if logs_file_pth.exists():
        import pandas as pd
        df = pd.read_csv(logs_file_pth)
        exist = ((df['model_name'] == model_name) & (df['dataset_name'] == dataset_name) & (df['noise_prob'] == noise_prob)).any()
        return exist
    else:
        return False
    

def run_baselines(model_list, dataset_list, extra_config, train_test_split=0.5, noise_prob=0.0):
    for model in model_list:
        for dataset in dataset_list:
            if dataset == 'UCR':
                index_range = range(1, 251)
                for idx in index_range:
                    print(f'Running experiment for model: {model}, dataset: {dataset} with index: {idx}')
                    cmd = ['python', 'supervised_robust.py', '--model_name', model, '--dataset_name', dataset, '--index', str(idx), 
                           '--task_name', 'supervised', '--train_test_split', str(train_test_split), '--noise_prob', str(noise_prob)]
                    if model in extra_config:
                        for key, value in extra_config[model].items():
                            cmd += [f'--{key}', str(value)]
                    handle = subprocess.run(cmd)
                    if handle.returncode != 0:
                        raise RuntimeError(f'Error occurred while running {model} on {dataset} with index {idx}')
            else:
                if check_exist(model, dataset, noise_prob):
                    print(f'Experiment for model: {model}, dataset: {dataset} with noise_prob: {noise_prob} already exists. Skipping...')
                    continue
                print(f'Running experiment for model: {model}, dataset: {dataset}')
                cmd = ['python', 'supervised_robust.py', '--model_name', model, '--dataset_name', dataset, '--task_name', 'supervised',
                       '--train_test_split', str(train_test_split), '--noise_prob', str(noise_prob)]
                handle = subprocess.run(cmd)
                if handle.returncode != 0:
                    raise RuntimeError(f'Error occurred while running {model} on {dataset}')

def run_stand(dataset_list, train_test_split=0.5, win_size=32, noise_prob=0.0):
    for dataset in dataset_list:
        if dataset == 'UCR':
            index_range = range(1, 251)
            for idx in index_range:
                print(f'Running experiment for model: STAND, dataset: {dataset} with index: {idx}')
                cmd = ['python', 'supervised_robust.py', '--model_name', 'STAND', '--dataset_name', dataset, '--index', str(idx),
                       '--task_name', 'supervised', '--train_test_split', str(train_test_split), '--win_size', str(win_size), '--noise_prob', str(noise_prob)]
                subprocess.run(cmd)
        else:
            if check_exist('STAND', dataset, noise_prob):
                print(f'Experiment for model: STAND, dataset: {dataset} with noise_prob: {noise_prob} already exists. Skipping...')
                continue
            print(f'Running experiment for model: STAND, dataset: {dataset}')
            cmd = ['python', 'supervised_robust.py', '--model_name', 'STAND', '--dataset_name', dataset,
                   '--task_name', 'supervised',
                   '--train_test_split', str(train_test_split), '--win_size', str(win_size), '--noise_prob', str(noise_prob)]
            subprocess.run(cmd)

if __name__ == '__main__':
    # 设置可用CUDA
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # train_test_split_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    # train_test_split_list = [0.2, 0.3, 0.4, 0.5]
    # train_test_split_list = [0.3, 0.4, 0.5]
    # train_test_split_list = [0.5]
    noise_prob_list = [0.01, 0.03, 0.05, 0.07, 0.1]
    # noise_prob_list = [0.05, 0.07, 0.1]
    # noise_prob_list = [0.01, 0.03, 0.05]
    # noise_prob_list = [0.01, 0.03, 0.05, 0.07, 0.1]
    from itertools import product
    task_list_generator = product(model_list, dataset_list, noise_prob_list)
    task_list_generator = list(task_list_generator)

    print(len(task_list_generator))
    #  nohup python src/scripts/run_supervised_robust.py 2>&1 | tee logs/log_file/robust2.txt &
    run_task = 0
    train_test_split = 0.5
    parallel_jobs = -1
    # random
    import numpy as np
    np.random.shuffle(task_list_generator)

    if parallel_jobs > 0:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = []
            for model, dataset, noise_prob in task_list_generator:
                model = [model]
                dataset = [dataset]
                if check_exist(model[0], dataset[0], noise_prob):
                    # print(f'Experiment for model: {model[0]}, dataset: {dataset[0]} with noise_prob: {noise_prob} already exists. Skipping...')
                    continue
                if model[0] == 'STAND':
                    futures.append(executor.submit(run_stand, dataset, train_test_split, noise_prob=noise_prob))
                else:
                    futures.append(executor.submit(run_baselines, model, dataset, extra_config, train_test_split, noise_prob=noise_prob))
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
        for model, dataset, noise_prob in task_list_generator:
            model = [model]
            dataset = [dataset]
            if check_exist(model[0], dataset[0], noise_prob):
                # print(f'Experiment for model: {model[0]}, dataset: {dataset[0]} with noise_prob: {noise_prob} already exists. Skipping...')
                continue
            else:
                print(f'Experiment for model: {model[0]}, dataset: {dataset[0]} with noise_prob: {noise_prob}')
            if model[0] == 'STAND':
                run_stand(dataset, train_test_split, noise_prob=noise_prob)
            else:
                run_baselines(model, dataset, extra_config, train_test_split, noise_prob=noise_prob)

                
    elif parallel_jobs==0:
        for noise_prob in noise_prob_list:
            if run_task == 0:
                run_baselines(model_list, dataset_list, extra_config, train_test_split, noise_prob=noise_prob)
                run_stand(dataset_list, train_test_split, noise_prob=noise_prob)
            else:
                # pass
                for task in task_list:
                    model_list_ = task['model_list']
                    dataset_list_ = task['dataset_list']
                    extra_config_ = task.get('extra_config', extra_config)
                    run_baselines(model_list_, dataset_list_, extra_config_, train_test_split, noise_prob=noise_prob)
                    run_stand(dataset_list_, train_test_split, noise_prob=noise_prob)