pth = '/home/zzj/projects/STAND'
data_pth = '/home/zzj/projects/FTSAD/datasets'
import sys
sys.path.append(pth)
from src.models.supervised.KNN import KNN
from src.data_utils.SimAD_data_loader2 import SupervisedDataset
from src.data_utils.SimAD_data_loader2 import get_loader_segment


dataset = get_loader_segment(0, data_pth+'/MSL', 128, 100, 1, 'train', 'MSL', 1, True)
train_loader = SupervisedDataset(dataset.test, dataset.test_labels, 100, 1, 'train', 0.8)
test_loader = SupervisedDataset(dataset.test, dataset.test_labels, 100, 1, 'test', 0.8)

model = KNN(2, n_jobs=8)
model.fit(train_loader.train, train_loader.train_labels)
pred = model.decision_function(test_loader.test)
print(pred)