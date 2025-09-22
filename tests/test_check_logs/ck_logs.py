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

model_list = ['RF', 'SVM', 'AdaBoost', 'ExtraTrees', 'LightGBM']
dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']#, 'UCR']
METRIC_LIST = ['CCE', 'F1', 'Aff-F1', 'UAff-F1', 'AUC-ROC', 'VUS-PR']
train_split_max_list = [0.1, 0.2, 0.3, 0.4, 0.5]

for model in model_list:
    model_df = df[df['model_name'] == model]
    for train_split in train_split_max_list:
        train_split_df = model_df[model_df['train_test_split'] == train_split]
        if train_split_df.empty:
            print(f'No entries found for model: {model}, train_test_split: {train_split}')
        else:
            for dataset in dataset_list:
                dataset_df = train_split_df[train_split_df['dataset_name'] == dataset]
                if dataset_df.empty:
                    print(f'No entries found for model: {model}, dataset: {dataset}, train_test_split: {train_split}')
                else:
                    for metric in METRIC_LIST:
                        # test if metric exists in columns, and the metric value is not NaN
                        if metric in dataset_df.columns:
                            metric_values = dataset_df[metric].dropna()
                            if metric_values.empty:
                                print(f'No valid values for metric: {metric} in model: {model}, dataset: {dataset}')


print('Log checking completed.')