from args import get_parse_args
from utils import dataset
config = get_parse_args()
print('Config: --------------------------------')
print(config)

assert config.task_name == 'supervised_limit', 'This script only supports supervised_limit task!'
import pathlib

proj_pth = pathlib.Path(__file__).parent.parent.parent
logs_pth = proj_pth / 'logs'
logs_pth = logs_pth.resolve()
print('Log will be saved to: ', logs_pth)
logs_pth.mkdir(exist_ok=True)

logs_file_pth = logs_pth / 'supervised_limit_un_semiun_exp.csv'
logs_file_pth2 = logs_pth / 'supervised_limit_un_semiun_exp_all_test.csv'
import os
import sys
sys.path.append(str(proj_pth))

from src.utils.model_wrapper import run_Supervise_AD, Supervise_AD_Pool
from src.utils.model_wrapper import run_Semisupervise_AD, Semisupervise_AD_Pool
from src.utils.model_wrapper import run_Unsupervise_AD, Unsupervise_AD_Pool

from src.data_utils.SimAD_data_loader2 import get_loader_segment, RandomSupervisedDataset

dataset_ = get_loader_segment(config.index, config.dataset_path+'/'+config.dataset_name, config.batch_size, 
                             config.win_size, config.step_size, 'train', config.dataset_name, 0, True)

train_dataset = RandomSupervisedDataset(dataset_.test, dataset_.test_labels, config.win_size, config.step_size, 'train', 
                                        config.anomaly_ratio, config.train_split_max)
# test_dataset = SupervisedDataset(dataset_.test, dataset_.test_labels, config.win_size, config.step_size, 'test', config.train_test_split)
# all_dataset = SupervisedDataset(dataset_.test, dataset_.test_labels, config.win_size, config.step_size, 'all', config.train_test_split)

train_y = train_dataset.train_y
train_labels = train_dataset.train_labels
test_y = train_dataset.test_y
test_labels = train_dataset.test_labels
all_test_y = train_dataset.all_test_y
all_test_labels = train_dataset.all_test_labels


from cce import metrics
metricor = metrics.basic_metricor()
config_dict = vars(config)
result1 = config_dict.copy()
result2 = config_dict.copy()

# assert config.model_name in Supervise_AD_Pool, f"Model {config.model_name} not in {Supervise_AD_Pool}"
config_dict_tmp = config_dict.copy()
config_dict_tmp.pop('model_name')
# test_score, test_all_score = run_Supervise_AD(config.model_name, train_y, train_labels, test_y, all_test_y, **config_dict_tmp)
if config.model_name == 'PCA':
    config_dict_tmp = config_dict.copy()
    config_dict_tmp.pop('model_name')
    test_score = run_Unsupervise_AD(config.model_name, test_y, **config_dict_tmp)
elif config.model_name == 'CATCH':
    config_dict_tmp = config_dict.copy()
    config_dict_tmp.pop('model_name')
    config_dict_tmp.pop('lr')
    config_dict_tmp.pop('win_size')
    test_score = run_Semisupervise_AD(config.model_name, train_y, test_y, **config_dict_tmp)
    # 计算数组中非NaN值的均值
    import numpy as np
    mean_value = np.nanmean(test_score)
    # 用均值填充NaN值
    test_score[np.isnan(test_score)] = mean_value
elif config.model_name in Supervise_AD_Pool:
    config_dict_tmp = config_dict.copy()
    config_dict_tmp.pop('model_name')
    test_score, test_all_score = run_Supervise_AD(config.model_name, train_y, train_labels, test_y, all_test_y, **config_dict_tmp)
else:
    raise ValueError(f'Model Selection Error!')

cnt = 0
while True:
    test_pred = metricor.get_pred(test_score, quantile=config.quantile - 0.05*cnt)
    if sum(test_pred) > 0:
        break
    else:
        cnt += 1
        if config.quantile - 0.05*cnt < 0:
            raise ValueError(f'Quantile adjustment out of range!, model: {config.model_name}, dataset: {config.dataset_name}, index: {config.index}')
        result1['quantile'] = config.quantile - 0.05*cnt
        print('Adjust quantile to: ', config.quantile - 0.05*cnt)


for metric in config.metric_list:
    val1 = metricor.metric_by_name(metric, test_labels, test_score, test_pred)
    result1.update({metric: val1})
    print(f"Metric: {metric}, Test: {val1:.4f}")

import pandas as pd
df = pd.DataFrame([result1])
if not logs_file_pth.exists():
    df.to_csv(logs_file_pth, index=False)
else:
    df.to_csv(logs_file_pth, index=False, header=False, mode='a')

print('Result saved to: ', logs_file_pth)
        

