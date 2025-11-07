import os
pth = r'/home/zzj/projects/FTSAD/datasets'
from sklearn.metrics import precision_recall_curve
name = 'SWAT'
# name = 'MSL'
# name = 'WADI'
# name = 'NIPS_TS_Swan'
name = 'PSM'
win_size = 32
batch_size = 128
pth = os.path.join(pth, name)

from SimAD_data_loader2 import get_loader_segment,SupervisedSegLoader,DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

dataset = get_loader_segment(0, pth, batch_size, win_size, 10, 'None', name, 0, True)
test_y = dataset.test
test_labels = dataset.test_labels
split = 0.4

dataset = SupervisedSegLoader(test_y, test_labels, win_size, 1, 'train', split)
train_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             sampler=None,
                             shuffle=True,
                             num_workers=8,
                             drop_last=False)
dataset = SupervisedSegLoader(test_y, test_labels, win_size, 1, 'all_test', split)
test_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             sampler=None,
                             shuffle=False,
                             num_workers=8,
                             drop_last=False)

print(len(train_loader),train_loader.dataset.train.shape)
print(len(test_loader),test_loader.dataset.test.shape)

def test_ml():
    # 训练ml模型，KNN，SVM，LR，RF，GBDT，XGBoost，LGBM，CatBoost
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    models = [KNeighborsClassifier(), SVC(), LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier()]
    for model in models:
        model.fit(train_loader.dataset.train, train_loader.dataset.train_labels)
        pred = model.predict(test_loader.dataset.test)
        print(f'{model.__class__.__name__}: {accuracy_score(test_loader.dataset.test_labels, pred)}')
        
# test_ml()

def test_dl():

    class SimCLF(nn.Module):
        def __init__(self, dim, win_size, d_model=32, bidirectional=False):
            super(SimCLF, self).__init__()
            self.emb = nn.Sequential(
                nn.Linear(dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.enc = nn.LSTM(d_model, d_model, 3, batch_first=True, bidirectional=bidirectional)
            if bidirectional:
                d_model = 2 * d_model
                self.clf = nn.Linear(d_model, 1)
            else:
                d_model = d_model
                self.clf = nn.Linear(d_model, 1)


        def forward(self, x):
            x = self.emb(x)
            x, _ = self.enc(x)
            x = self.clf(x).squeeze(-1)
            return x
        
    epochs = 10
    dim = dataset.test.shape[1]
    print(dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = SimCLF(dim, win_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    


    for epoch in tqdm(range(epochs)):
        acc = 0
        preds = []
        labels = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.sigmoid(pred)
            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            pred = (pred > 0.5).float()
            acc += (pred == y).float().mean().item()
        preds = np.concatenate(preds, axis=0).reshape(-1)
        labels = np.concatenate(labels, axis=0).reshape(-1)

        precision, recall, thresholds = precision_recall_curve(labels, preds)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1 = np.max(f1)
        best_threshold = thresholds[np.argmax(f1)]
        print(f'Best F1: {best_f1}, Best Threshold: {best_threshold}')

        print(f'Train Accuracy: {acc / len(train_loader)}')
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        
    acc = 0
    preds = []
    labels = []
    model.eval()
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = torch.sigmoid(pred)
        preds.append(pred.detach().cpu().numpy())
        pred = (pred > 0.5).float()
        acc += (pred == y).float().mean().item()
        labels.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1)
    labels = np.concatenate(labels, axis=0).reshape(-1)
    print(preds.shape)
    # preds有nan
    print(np.isnan(preds).sum())
    print(np.isnan(labels).sum())
    # 填充nan用平均值
    preds[np.isnan(preds)] = np.mean(preds[~np.isnan(preds)])
    labels[np.isnan(labels)] = np.mean(labels[~np.isnan(labels)])
    print(labels.shape)


    precision, recall, thresholds = precision_recall_curve(labels, preds)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_f1 = np.max(f1)
    best_threshold = thresholds[np.argmax(f1)]
    print(f'Best F1: {best_f1}, Best Threshold: {best_threshold}')
    print(f'Test Accuracy: {acc / len(test_loader)}')

    # plot
    import matplotlib.pyplot as plt
    plt.plot(preds)
    plt.savefig('/home/zzj/projects/FF-BLS/preds.png')
    plt.show()


test_dl()


