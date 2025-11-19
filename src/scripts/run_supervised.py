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

def run_stand(dataset_list, train_test_split=0.5, win_size=32):
    for dataset in dataset_list:
        if dataset == 'UCR':
            index_range = range(1, 251)
            for idx in index_range:
                print(f'Running experiment for model: STAND, dataset: {dataset} with index: {idx}')
                cmd = ['python', 'supervised.py', '--model_name', 'STAND', '--dataset_name', dataset, '--index', str(idx),
                       '--task_name', 'supervised', '--train_test_split', str(train_test_split), '--win_size', str(win_size)]
                subprocess.run(cmd)
        else:
            print(f'Running experiment for model: STAND, dataset: {dataset}')
            cmd = ['python', 'supervised.py', '--model_name', 'STAND', '--dataset_name', dataset,
                   '--task_name', 'supervised',
                   '--train_test_split', str(train_test_split), '--win_size', str(win_size)]
            subprocess.run(cmd)

if __name__ == '__main__':
    # train_test_split_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    # train_test_split_list = [0.2, 0.3, 0.4, 0.5]
    # train_test_split_list = [0.3, 0.4, 0.5]
    train_test_split_list = [0.5]
    run_task = 0
    for train_test_split in train_test_split_list:
        if run_task == 0:
            run_baselines(model_list, dataset_list, extra_config, train_test_split)
            # run_stand(dataset_list, train_test_split)
        else:
            for task in task_list:
                model_list_ = task['model_list']
                dataset_list_ = task['dataset_list']
                extra_config_ = task.get('extra_config', extra_config)
                run_baselines(model_list_, dataset_list_, extra_config_, train_test_split)
                # run_stand(dataset_list_, train_test_split)