dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']
# anomaly_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
anomaly_ratio_list = [0.1, 0.2, 0.4]
dataset_list = ['PSM']

from itertools import groupby
from operator import itemgetter

def convert_vector_to_events(vector = [0, 1, 1, 0, 0, 1, 0]):
    """
    Convert a binary vector (indicating 1 for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).
    
    :param vector: a list of elements belonging to {0, 1}
    :return: a list of couples, each couple representing the start and stop of
    each event
    """
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix : ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    
    # Consistent conversion in case of range anomalies (for indexes):
    # A positive index i is considered as the interval [i, i+1),
    # so the last index should be moved by 1
    events = [(x, y+1) for (x,y) in events]
        
    return (events)

base_pth = r'/home/pcl/zzj/projects/FTSAD/datasets'

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
        train_dataset = RandomSupervisedDataset(dataset_.test, dataset_.test_labels, 32, 16, 'train', anomaly_ratio, 0.3)
        train_anomaly_ratio = train_dataset.train_anomaly_ratio
        test_anomaly_ratio = train_dataset.test_anomaly_ratio
        all_test_anomaly_ratio = sum(train_dataset.all_test_labels) / len(train_dataset.all_test_labels)

        train_labels = train_dataset.train_labels
        test_labels = train_dataset.test_labels
        train_event = convert_vector_to_events(train_labels)
        train_event_N = len(train_event)
        test_labels = convert_vector_to_events(test_labels)
        test_labels_N = len(test_labels)
        print(dataset_name,anomaly_ratio,train_event_N,test_labels_N)

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
        # print(result)

# import pandas as pd
# df = pd.DataFrame(results)
# df.to_csv(proj_pth / 'logs' / 'dataset_details.csv', index=False)
