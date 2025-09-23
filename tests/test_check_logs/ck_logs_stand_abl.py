import pathlib
proj_pth = pathlib.Path(__file__).parent.parent.parent

logs_pth = proj_pth / 'logs'
import os

if not os.path.exists(logs_pth):
    print('Log path does not exist: ', logs_pth)
else:
    print('Log path exists: ', logs_pth)


supervise_log = logs_pth / 'supervised_exp.csv'
import pandas as pd
if not os.path.exists(supervise_log):
    print('Supervised log file does not exist: ', supervise_log)
else:
    print('Supervised log file exists: ', supervise_log)
    df = pd.read_csv(supervise_log)
    print('First few lines of the supervised log:')


df = df[df['model_name']=='STAND']
train_test_split_list = [0.5]
win_size_list = [32,64,128]
d_model_list = [16,32,64,128,256]
num_layers_list = [0,1,2,3]
bidirectional_list = [0,1]
dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
from itertools import product
task_list_generator = product(dataset_list, train_test_split_list, win_size_list, d_model_list, num_layers_list, bidirectional_list)
task_list_generator = list(task_list_generator)

print('Total number of tasks to check: ', len(task_list_generator))

cnt = 0 
fl_cnt = 0
suc_cnt = 0
for task in task_list_generator:
    tmp = df[(df['dataset_name']==task[0]) & (df['train_test_split']==task[1]) & (df['win_size']==task[2]) & (df['d_model']==task[3]) & (df['num_layers']==task[4]) & (df['bidirectional']==task[5])]
    if tmp.shape[0]==0:
        cnt+=1
    elif tmp['F1'] is None or tmp['F1'].isna().all():
        fl_cnt+=1
    else:
        suc_cnt +=1

print(f'Total tasks: {len(task_list_generator)}, Not run tasks: {cnt}, Failed tasks: {fl_cnt}, Successful tasks: {suc_cnt}')