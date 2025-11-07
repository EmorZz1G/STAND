pth = '/home/zzj/projects/STAND'
data_pth = '/home/zzj/projects/FTSAD/datasets'
import sys
sys.path.append(pth)
from src.models.supervised.STAND import STAND
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


print(len(train_loader),train_loader.dataset.train.shape)
print(len(test_loader),test_loader.dataset.test.shape)

model = STAND(32, device='cuda', epochs=10, d_model=64, num_layers=1, debug=1)
model.fit(train_loader.dataset.train, train_loader.dataset.train_labels)
# model.fit_loader(train_loader)
pred = model.decision_function(all_loader.dataset.all_test_y)
labels = all_loader.dataset.all_test_labels
# pred, labels = model.decision_function_loader(all_loader)
print(pred)
print(labels)

# eval
from sklearn.metrics import precision_recall_curve, roc_curve, auc,accuracy_score, f1_score

acc = accuracy_score(labels, pred > 0.5)
print('acc',acc)
precision, recall, thresholds = precision_recall_curve(labels, pred)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1)]
best_f1 = np.max(f1)
print('best_threshold: ', best_threshold)
print('best_f1: ', best_f1)
pred2 = (pred>0.5).astype(float)
f1 = f1_score(labels, pred2)
print('f1 ',f1)

auc_score = auc(recall, precision)
print('auc_score: ', auc_score)