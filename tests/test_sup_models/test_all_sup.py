pth = '/home/zzj/projects/STAND'
data_pth = '/home/zzj/projects/FTSAD/datasets'
import sys
sys.path.append(pth)
from src.models.supervised import KNN, LR, RF, SVM, STAND, AdaBoost, ExtraTrees, LightGBM
from src.data_utils.SimAD_data_loader2 import SupervisedDataset
from src.data_utils.SimAD_data_loader2 import get_loader_segment
import numpy as np
from torch.utils.data import DataLoader
win_size = 32
batch_size = 128
dataset = get_loader_segment(0, data_pth+'/PSM', batch_size, win_size, 10, 'train', 'PSM', 0, True)
train_loader = SupervisedDataset(dataset.test, dataset.test_labels, 32, 1, 'train', 0.4)
test_loader = SupervisedDataset(dataset.test, dataset.test_labels, 32, 1, 'test', 0.4)
all_loader = SupervisedDataset(dataset.test, dataset.test_labels, 32, 1, 'all', 0.4)
train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False, num_workers=8)
all_loader = DataLoader(all_loader, batch_size=batch_size, shuffle=False, num_workers=8)
from sklearn.metrics import f1_score,precision_recall_curve
models = [RF, KNN,  RF, SVM, STAND, AdaBoost, ExtraTrees, LightGBM]
# models = [ LightGBM]
for model_ in models:
    print(model_.__name__)
    model = model_(2)
    model.fit(train_loader.dataset.train, train_loader.dataset.train_labels)
    try:
        pred = model.decision_function(test_loader.dataset.test)
    except:
        print('ok')
        # pred = model.predict_proba(test_loader.dataset.test)
        # if model_.__name__ in ['RandomForestClassifier','KNeighborsClassifier','ExtraTreesClassifier']:
        #     pred = np.mean(pred, 1)
        # print(model_.__name__)
    print(pred.shape,test_loader.dataset.test.shape)
    
    precision, recall, thresholds = precision_recall_curve(test_loader.dataset.test_labels, pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_thres = thresholds[np.argmax(f1)]
    best_f1 = np.max(f1)
    print('Best F1: ', best_f1)
    print('Best Threshold: ', best_thres)
    pred = (pred >= best_thres).astype(int)

    print('--------------------------------')