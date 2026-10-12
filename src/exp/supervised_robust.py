from args import get_parse_args
# from utils import dataset
config = get_parse_args()
print('Config: --------------------------------')
print(config)

assert config.task_name == 'supervised', 'This script only supports supervised task!'
import pathlib

proj_pth = pathlib.Path(__file__).parent.parent.parent
logs_pth = proj_pth / 'logs'
logs_pth = logs_pth.resolve()
print('Log will be saved to: ', logs_pth)
logs_pth.mkdir(exist_ok=True)

# noise_type=''
noise_type='_fp'

logs_file_pth = logs_pth / f'supervised_robust_exp{noise_type}.csv'
logs_file_pth2 = logs_pth / f'supervised_robust_exp_all_test{noise_type}.csv'
# logs_file_pth3 = logs_pth / 'supervised_robust_exp_left_test.csv'
import os
import sys
sys.path.append(str(proj_pth))

from src.utils.model_wrapper import run_Supervise_AD, Supervise_AD_Pool
from src.data_utils.SimAD_data_loader2 import get_loader_segment, SupervisedDataset

dataset_ = get_loader_segment(config.index, config.dataset_path+'/'+config.dataset_name, config.batch_size, 
                             config.win_size, config.step_size, 'train', config.dataset_name, 0, True)

train_dataset = SupervisedDataset(dataset_.test, dataset_.test_labels, config.win_size, config.step_size, 'train', config.train_test_split)
# test_dataset = SupervisedDataset(dataset_.test, dataset_.test_labels, config.win_size, config.step_size, 'test', config.train_test_split)
# all_dataset = SupervisedDataset(dataset_.test, dataset_.test_labels, config.win_size, config.step_size, 'all', config.train_test_split)

train_y = train_dataset.train
train_labels = train_dataset.train_labels
test_y = train_dataset.test
test_labels = train_dataset.test_labels
all_test_y = train_dataset.all_test_y
all_test_labels = train_dataset.all_test_labels
noise_prob = config.noise_prob

import numpy as np

def get_real_anomaly_lengths(labels):
    """
    从标签中提取所有连续异常(1)片段的长度，并返回最大和最小值。
    """
    labels = np.asarray(labels)
    
    # 在数组首尾各补一个 0。这样做是为了完美处理开头或结尾正好是 1 的边缘情况
    padded = np.pad(labels, (1, 1), 'constant', constant_values=(0, 0))
    
    # 计算相邻元素的差分
    # 从 0 变成 1，差分为 1 (片段开始)
    # 从 1 变成 0，差分为 -1 (片段结束)
    diffs = np.diff(padded)
    
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    # 结束索引减去开始索引，就是每个异常片段的真实长度
    segment_lengths = ends - starts
    
    if len(segment_lengths) == 0:
        print("警告：提供的标签中没有任何异常(1)。将使用默认长度。")
        return 10, 50 # 随便给个默认值防止报错
        
    min_len = np.min(segment_lengths)
    max_len = np.max(segment_lengths)
    mean_len = np.mean(segment_lengths) # 顺便看看平均长度也很有用
    
    print(f"发现 {len(segment_lengths)} 个真实异常片段。")
    print(f"最短: {min_len}, 最长: {max_len}, 平均: {mean_len:.1f}")
    
    return min_len, max_len

def inject_segment_noise(labels, noise_ratio=0.01, min_seg_len=20, max_seg_len=100):
    """
    向时序标签中注入“区块噪声”（将连续的正常 0 标记为异常 1）。
    
    参数:
        labels: 原始的真实标签，1D numpy array (0 为正常，1 为异常)
        noise_ratio: 想要污染的正常数据的比例 (默认 10%)
        min_seg_len: 注入的噪声区块的最小长度
        max_seg_len: 注入的噪声区块的最大长度
        
    返回:
        noisy_labels: 注入噪声后的新标签
    """
    # 拷贝一份，避免修改原数据
    noisy_labels = np.copy(labels)
    
    # 找出所有真实为正常 (0) 的索引
    normal_indices = np.where(labels == 0)[0]
    if len(normal_indices) == 0:
        return noisy_labels
        
    # 计算我们需要翻转的正常时间点总数
    target_noise_count = int(len(normal_indices) * noise_ratio)
    current_noise_count = 0
    
    # 开始随机注入区块，直到达到目标噪声比例
    while current_noise_count < target_noise_count:
        # 1. 随机选一个正常点作为区块的起点
        start_idx = np.random.choice(normal_indices)
        
        # 2. 随机决定这个噪声区块的长度
        seg_len = np.random.randint(min_seg_len, max_seg_len + 1)
        end_idx = min(start_idx + seg_len, len(labels)) # 防止越界
        
        # 3. 提取这个区间内的标签
        segment = noisy_labels[start_idx:end_idx]
        
        # 4. 我们只把区间里的 0 翻转成 1 (如果区间里本来就有真实的 1，就不动它)
        mask = (segment == 0)
        flipped_count = np.sum(mask)
        
        if flipped_count > 0:
            
            if current_noise_count + flipped_count > target_noise_count:
                break

            # 执行翻转：将这个区间内为 0 的位置赋值为 1
            noisy_labels[start_idx:end_idx][mask] = 1
            # 累计已注入的噪声数量
            current_noise_count += flipped_count
            
    return noisy_labels

if noise_prob > 0:
    import numpy as np
    np.random.seed(47)
    train_n = len(train_labels)

    if noise_type=='':
        noise_indices = np.random.choice(train_n, size=int(train_n * noise_prob), replace=False)
        train_labels[noise_indices] = 1 - train_labels[noise_indices]
    elif noise_type=='_fp':
        # 选择
        # 1. 先从测试集(或所有数据)的真实标签中，学习异常片段的长度分布
        min_real_len, max_real_len = get_real_anomaly_lengths(all_test_labels)

        # 2. 使用这两个真实学到的边界，去污染训练集
        # 假设你想把 train_labels 中 10% 的正常数据变成假异常
        noisy_train_labels = inject_segment_noise(
            labels=train_labels, 
            noise_ratio=noise_prob, 
            min_seg_len=20, 
            max_seg_len=400
        )
        train_labels = noisy_train_labels


from cce import metrics
metricor = metrics.basic_metricor()
config_dict = vars(config)
result1 = config_dict.copy()
result2 = config_dict.copy()
result3 = config_dict.copy()

assert config.model_name in Supervise_AD_Pool, f"Model {config.model_name} not in {Supervise_AD_Pool}"
config_dict_tmp = config_dict.copy()
config_dict_tmp.pop('model_name')
test_score, test_all_score = run_Supervise_AD(config.model_name, train_y, train_labels, test_y, all_test_y, **config_dict_tmp)

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

cnt = 0
while True:
    test_all_pred = metricor.get_pred(test_all_score, quantile=config.quantile - 0.05*cnt)
    if sum(test_all_pred) > 0:
        break
    else:
        cnt += 1
        if config.quantile - 0.05*cnt < 0:
            raise ValueError(f'Quantile adjustment out of range!, model: {config.model_name}, dataset: {config.dataset_name}, index: {config.index}')
        result2['quantile'] = config.quantile - 0.05*cnt
        print('Adjust quantile to: ', config.quantile - 0.05*cnt)

left_len = len(test_all_score) - len(test_score)
test_left_score = test_all_score.copy()[-left_len:]
test_left_labels = all_test_labels.copy()[-left_len:]
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
    val1 = metricor.metric_by_name(metric, test_labels, test_score, test_pred)
    # val2 = metricor.metric_by_name(metric, all_test_labels, test_all_score, test_all_pred)
    # val3 = metricor.metric_by_name(metric, test_left_labels, test_left_score, test_left_pred)
    result1.update({metric: val1})
    # result2.update({metric: val2})
    # result3.update({metric: val3})
    # print(f"Metric: {metric}, Test: {val1:.4f}, All Test: {val2:.4f}, Left Test: {val3:.4f}")

import pandas as pd
df = pd.DataFrame([result1])
df2 = pd.DataFrame([result2])
df3 = pd.DataFrame([result3])
if not logs_file_pth.exists():
    df.to_csv(logs_file_pth, index=False)
else:
    df.to_csv(logs_file_pth, index=False, header=False, mode='a')

# if not logs_file_pth2.exists():
#     df2.to_csv(logs_file_pth2, index=False)
# else:
#     df2.to_csv(logs_file_pth2, index=False, header=False, mode='a')

# if not logs_file_pth3.exists():
#     df3.to_csv(logs_file_pth3, index=False)
# else:
#     df3.to_csv(logs_file_pth3, index=False, header=False, mode='a')

print('Result saved to: ', logs_file_pth)
        

