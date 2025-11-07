dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']
anomaly_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]

base_pth = r'/public/home/202220143416/projects/FTSAD/datasets'

import pathlib
proj_pth = pathlib.Path(__file__).parent.parent.parent
import sys
sys.path.append(str(proj_pth))

from src.data_utils.SimAD_data_loader2 import get_loader_segment

results = []

for dataset_name in dataset_list:
    dataset = get_loader_segment(0, base_pth+'/'+dataset_name, 64, 32, 16, 'train', dataset_name, 0, True)
    train = dataset.train
    test = dataset.test
    labels = dataset.test_labels
    result = {
        'Dataset': dataset_name,
        "#Channel":train.shape[1],
        '#Train': train.shape[0],
        '#Test':test.shape[0],
        'Anomaly Rate': (sum(labels) / len(labels)).item()
    }
    results.append(result)

import pandas as pd
print(results)
df = pd.DataFrame(results)
df.to_csv(proj_pth / 'logs' / 'dataset_details_all.csv', index=False)
