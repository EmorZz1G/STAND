dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']
anomaly_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]

base_pth = r'/share/home/202220143416/project/FTSAD/datasets'

import pathlib
proj_pth = pathlib.Path(__file__).parent.parent.parent
import sys
sys.path.append(str(proj_pth))

from src.data_utils.SimAD_data_loader2 import get_loader_segment, RandomSupervisedDataset

results = []

for dataset_name in dataset_list:
    for anomaly_ratio in anomaly_ratio_list:
        print(f'Generating dataset details for dataset: {dataset_name} with anomaly ratio: {anomaly_ratio}')
        dataset_ = get_loader_segment(0, base_pth+'/'+dataset_name, 64, 32, 16, 'train', dataset_name, 0, True)
        train_dataset = RandomSupervisedDataset(dataset_.test, dataset_.test_labels, 32, 16, 'train', anomaly_ratio, 0.5)
        train_anomaly_ratio = train_dataset.train_anomaly_ratio
        test_anomaly_ratio = train_dataset.test_anomaly_ratio
        all_test_anomaly_ratio = sum(train_dataset.all_test_labels) / len(train_dataset.all_test_labels)
        result = {
            'dataset_name': dataset_name,
            'anomaly_ratio': anomaly_ratio,
            'train_size': len(train_dataset.train_y),
            'train_anomaly_ratio': train_anomaly_ratio,
            'test_size': len(train_dataset.test_y),
            'test_anomaly_ratio': test_anomaly_ratio,
            'all_test_size': len(train_dataset.all_test_y),
            'all_test_anomaly_ratio': all_test_anomaly_ratio
        }
        results.append(result)

import pandas as pd
df = pd.DataFrame(results)
df.to_csv(proj_pth / 'logs' / 'dataset_details.csv', index=False)
