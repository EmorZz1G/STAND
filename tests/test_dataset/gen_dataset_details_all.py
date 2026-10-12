dataset_list = ['PSM', 'SWAT', 'WADI', 'NIPS_TS_Swan', 'NIPS_TS_Water']
anomaly_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]

base_pth = r'/public/home/202220143416/projects/FTSAD/datasets'
base_pth = r'/home/pcl/zzj/projects/FTSAD/datasets'
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
import pathlib
proj_pth = pathlib.Path(__file__).parent.parent.parent
import sys
sys.path.append(str(proj_pth))

from src.data_utils.SimAD_data_loader2 import get_loader_segment, SupervisedDataset

results = []

for dataset_name in dataset_list:
    dataset = get_loader_segment(0, base_pth+'/'+dataset_name, 64, 32, 16, 'train', dataset_name, 0, True)
    train_dataset = SupervisedDataset(dataset.test, dataset.test_labels, 32, 32, 'train', 0.5)
    train = dataset.train
    test = dataset.test
    labels = dataset.test_labels

    init_train = train_dataset.train_labels
    test_labels = train_dataset.test_labels

    train_event = convert_vector_to_events(init_train)
    train_event_N = len(train_event)
    train_anomaly_ratio = (sum(init_train)/len(init_train)).item()

    test_event = convert_vector_to_events(test_labels)
    test_event_N = len(test_event)
    test_anomaly_ratio = (sum(test_labels)/len(test_labels)).item()
    result = {
        'Dataset': dataset_name,
        "#Channel":train.shape[1],
        'Norm Len 用于半监督方法': train.shape[0],
        'init len': len(init_train),
        'init Anomaly Rate': train_anomaly_ratio,
        'init evnt': train_event_N,
        'test len': len(test_labels),
        'test Anomaly Rate': test_anomaly_ratio,
        'test event': test_event_N,
    }
    results.append(result)

import pandas as pd
print(results)
df = pd.DataFrame(results)
df.to_csv(proj_pth / 'logs' / 'dataset_details_all.csv', index=False)
