from args import get_parse_args
from utils import dataset
config = get_parse_args()
print('Config: --------------------------------')
print(config)

assert config.task_name == 'semisupervised', 'This script only supports semisupervised task!'
import pathlib
import numpy as np

proj_pth = pathlib.Path(__file__).parent.parent.parent
logs_pth = proj_pth / 'logs'
logs_pth = logs_pth.resolve()
print('Log will be saved to: ', logs_pth)
logs_pth.mkdir(exist_ok=True)

logs_file_pth = logs_pth / 'semisupervised_exp.csv'
logs_file_pth3 = logs_pth /'semisupervised_exp_left_test.csv'

import os
import sys
sys.path.append(str(proj_pth))

from src.utils.model_wrapper import run_Semisupervise_AD, Semisupervise_AD_Pool
from src.data_utils.SimAD_data_loader2 import get_loader_segment, SupervisedDataset

dataset_ = get_loader_segment(config.index, config.dataset_path+'/'+config.dataset_name, config.batch_size, 
                             config.win_size, config.step_size, 'train', config.dataset_name, 0, True)

train_dataset = SupervisedDataset(dataset_.test, dataset_.test_labels, config.win_size, config.step_size, 'train', config.train_test_split)
train_y = train_dataset.train_y
test_y = train_dataset.all_test_y
test_labels = train_dataset.all_test_labels

test_y_for_stad = train_dataset.test
left_len = len(test_y_for_stad)

from cce import metrics
metricor = metrics.basic_metricor()
config_dict = vars(config)

result1 = config_dict.copy()
result3 = config_dict.copy()

assert config.model_name in Semisupervise_AD_Pool, f"Model {config.model_name} not in {Semisupervise_AD_Pool}"
config_dict_tmp = config_dict.copy()
config_dict_tmp.pop('model_name')
config_dict_tmp.pop('lr')
config_dict_tmp.pop('win_size')
test_score = run_Semisupervise_AD(config.model_name, train_y, test_y, **config_dict_tmp)
# 计算数组中非NaN值的均值
mean_value = np.nanmean(test_score)
# 用均值填充NaN值
test_score[np.isnan(test_score)] = mean_value


cnt = 0
while True:
    test_pred = metricor.get_pred(test_score, quantile=config.quantile - 0.05*cnt).astype(int)
    if sum(test_pred) > 0:
        break
    else:
        cnt += 1
        if config.quantile - 0.05*cnt < 0:
            raise ValueError(f'Quantile adjustment out of range!, model: {config.model_name}, dataset: {config.dataset_name}, index: {config.index}')
        result1['quantile'] = config.quantile - 0.05*cnt
        print('Adjust quantile to: ', config.quantile - 0.05*cnt)


test_left_score = test_score.copy()[-left_len:]
test_left_labels = test_labels.copy()[-left_len:]

cnt = 0
while True:
    test_left_pred = metricor.get_pred(test_left_score, quantile=config.quantile - 0.05*cnt)
    if sum(test_left_pred) > 0:
        break
    else:
        cnt += 1
        if config.quantile - 0.05*cnt < 0:
            raise ValueError(f'Quantile adjustment out of range!, model: {config.model_name}, dataset: {config.dataset_name}, index: {config.index}')
        result3['quantile'] = config.quantile - 0.05*cnt
        print('Adjust quantile to: ', config.quantile - 0.05*cnt)

for metric in config.metric_list:
    print(test_labels.shape, test_score.shape, test_pred.shape)
    val1 = metricor.metric_by_name(metric, test_labels, test_score, test_pred)
    val3 = metricor.metric_by_name(metric, test_left_labels, test_left_score, test_left_pred)
    result1[metric] = val1
    result3.update({metric: val3})
    print(f'Metric {metric}, All Test: {val1:.4f} Left Test: {val3:.4f}')


import pandas as pd
df = pd.DataFrame([result1])
df3 = pd.DataFrame([result3])
# if not logs_file_pth.exists():
#     df.to_csv(logs_file_pth, index=False)
# else:
#     df.to_csv(logs_file_pth, index=False, header=False, mode='a')


if not logs_file_pth3.exists():
    df3.to_csv(logs_file_pth3, index=False)
else:
    df3.to_csv(logs_file_pth3, index=False, header=False, mode='a')

print('Result saved to: ', logs_file_pth)