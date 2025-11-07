import pathlib

cur_proj = pathlib.Path(__file__).parent.parent / 'exp'
print('Current project path: ', cur_proj)

import os
os.chdir(cur_proj)
print('Current working directory: ', os.getcwd())

model_list = ['STAND']
model_list = ['ExtraTrees','LightGBM']
dataset_list = ['SWAT']#, 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']

import subprocess

def run_model(model_name, dataset_list, train_test_split=0.5, win_size=32, num_layers=1, d_model=300, bidirectional=1):
    for dataset in dataset_list:
        if dataset == 'UCR':
            index_range = range(1, 251)
            for idx in index_range:
                print(f'Running experiment for model: {model_name}, dataset: {dataset} with index: {idx}')
                cmd = ['python', 'supervised.py', '--model_name', model_name, '--dataset_name', dataset, '--index', str(idx),
                       '--task_name', 'supervised', '--train_test_split', str(train_test_split), '--win_size', str(win_size),
                       '--num_layers', str(num_layers), '--d_model', str(d_model), '--bidirectional', str(bidirectional), '--if_save', '1',]
                subprocess.run(cmd)
        else:
            print(f'Running experiment for model: {model_name}, dataset: {dataset}')
            cmd = ['python', 'supervised.py', '--model_name', model_name, '--dataset_name', dataset,
                   '--task_name', 'supervised',
                   '--train_test_split', str(train_test_split), '--win_size', str(win_size),
                   '--num_layers', str(num_layers), '--d_model', str(d_model), '--bidirectional', str(bidirectional), '--if_save', '1',]
            subprocess.run(cmd)




if __name__ == '__main__':
    train_test_split_list = [0.5]
    for train_test_split in train_test_split_list:
        for model in model_list:
            for dataset in dataset_list:
                run_model(model, [dataset], train_test_split=train_test_split)