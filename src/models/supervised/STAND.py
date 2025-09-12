"""
This function is implemented by EmorZz1G.
"""

from __future__ import division
from __future__ import print_function

import math
import numpy as np

from scipy.optimize._lsq import bvls
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.utils import check_array

from ..base import BaseDetector
from ..feature import Window
from ...utils.utility import zscore
from sklearn.metrics import precision_recall_curve

class _StandNet(nn.Module):
    def __init__(self, input_dim, d_model=128, num_layers=2, bidirectional=True):
        super(_StandNet, self).__init__()
        self.emb = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.enc = nn.LSTM(d_model, d_model, num_layers, batch_first=True, bidirectional=bidirectional)
        proj_in = d_model * (2 if bidirectional else 1)
        self.clf = nn.Linear(proj_in, 1)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.enc(x)
        x = self.clf(x).squeeze(-1)
        return x


class STAND(BaseDetector):
    """Supervised time series anomaly detector (PyTorch), following sliding-window API.

    Parameters
    ----------
    slidingWindow : int
        Window length for sliding conversion.
    normalize : bool
        Whether to z-score normalize each window.
    epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate.
    optimizer : str
        One of ['adam', 'sgd', 'adamw']. Default 'adam'.
    d_model : int
        Hidden size for MLP/LSTM.
    num_layers : int
        Number of LSTM layers.
    bidirectional : bool
        Whether LSTM is bidirectional.
    device : str
        'cuda' or 'cpu'. If None, auto-detect.
    """

    def __init__(self, slidingWindow=32, sub=True, contamination=0.1, normalize=False,
                 epochs=10, batch_size=128, lr=1e-3, optimizer='adam',
                 d_model=64, num_layers=1, bidirectional=False, device=None,
                 debug=0,
                 **kwargs):

        self.slidingWindow = slidingWindow
        self.sub = sub
        self.normalize = normalize

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_name = optimizer.lower()
        self.d_model = d_model
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self._model = None
        self.debug=debug
        self.fit_on_loader=0

    def _build_model(self, input_dim):
        model = _StandNet(input_dim=input_dim, d_model=self.d_model,
                          num_layers=self.num_layers, bidirectional=self.bidirectional)
        return model.to(self.device)

    def _make_optimizer(self, model):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=self.lr)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def fit_loader(self, loader):
        self.fit_on_loader=1
        # Peek one batch to get input dimension and validate shapes
        it = iter(loader)
        try:
            xb0, yb0 = next(it)
        except StopIteration:
            return self

        win = xb0.shape[1]
        feature_dim = xb0.shape[2]

        # Build model and training tools
        self._model = self._build_model(input_dim=feature_dim)
        optimizer = self._make_optimizer(self._model)
        criterion = nn.BCEWithLogitsLoss()

        from tqdm import tqdm
        self._model.train()
        bar = tqdm(range(self.epochs), desc='Training STAND')
        last_acc=0
        last_f1=0
        for _ in bar:
            # reuse the first peeked batch, then the rest
            preds = []
            labels = []
            acc=0
            for i, (xb, yb) in enumerate(loader):
                optimizer.zero_grad()
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                if self.debug:
                    bar.set_postfix(loss=loss.item(),f1=last_f1,acc=last_acc)
                    pred = torch.sigmoid(pred)
                    preds.append(pred.detach().cpu().numpy())
                    labels.append(yb.detach().cpu().numpy())
                    pred = (pred > 0.5).float()
                    acc += (pred == yb).float().mean().item()
                else:
                    bar.set_postfix(loss=loss.item())
            
            
            if self.debug:
                preds = np.concatenate(preds, axis=0).reshape(-1)
                labels = np.concatenate(labels, axis=0).reshape(-1)
                precision, recall, thresholds = precision_recall_curve(labels, preds)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                best_f1 = np.max(f1)
                last_f1 = best_f1
                last_acc = acc/len(loader)
            # print(f'Best F1: {best_f1}, Best Threshold: {best_threshold}')

            # print(f'Train Accuracy: {acc / len(train_loader)}')
                


        return self
    
    def decision_function_loader(self, loader):
        if self.fit_on_loader==0:
            print('STAND: fit_on_loader is 0, please call fit_loader first')
            return None
        elif self.fit_on_loader==2:
            print('STAND has been fitted by "fit" function, please call decision_function function')
            return None

        self._model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logit = self._model(xb)
                prob = torch.sigmoid(logit)  # [B, win]
                preds.append(prob.cpu().numpy())
                labels.append(yb.cpu().numpy())
        if len(preds) == 0:
            return np.array([], dtype=float)
        scores = np.concatenate(preds, axis=0).reshape(-1)
        labels = np.concatenate(labels, axis=0).reshape(-1)
        return scores, labels
    
    def fit(self, X, y):
        self.fit_on_loader=2
        n_samples, feature_dim = X.shape

        Xw = Window(window=self.slidingWindow).convert(X)
        yw = Window(window=self.slidingWindow).convert(y)
        if self.normalize:
            Xw = zscore(Xw, axis=1, ddof=1)

        Xw = check_array(Xw)

        # In STAND, reshape features back to [N, win, dim] assuming original dim=1 per time step.
        win = self.slidingWindow
        X_seq = Xw.reshape(-1, win, feature_dim)

        # Ensure labels are float in [0,1] with shape [N, win]
        y_seq = yw.reshape(-1, win)
        print(X_seq.shape,y_seq.shape)

        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

        self._model = self._build_model(input_dim=feature_dim)
        optimizer = self._make_optimizer(self._model)
        criterion = nn.BCEWithLogitsLoss()

        self._model.train()
        from tqdm import tqdm
        last_acc=0
        last_f1=0
        bar = tqdm(range(self.epochs), desc='Training STAND')
        for _ in bar:
            preds = []
            labels = []
            acc=0
            for xb, yb in loader:
                optimizer.zero_grad()
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                
                if self.debug:
                    bar.set_postfix(loss=loss.item(),f1=last_f1,acc=last_acc)
                    pred = torch.sigmoid(pred)
                    preds.append(pred.detach().cpu().numpy())
                    labels.append(yb.detach().cpu().numpy())
                    pred = (pred > 0.5).float()
                    acc += (pred == yb).float().mean().item()
                else:
                    bar.set_postfix(loss=loss.item())
                    
            if self.debug:
                preds = np.concatenate(preds, axis=0).reshape(-1)
                labels = np.concatenate(labels, axis=0).reshape(-1)
                precision, recall, thresholds = precision_recall_curve(labels, preds)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                best_f1 = np.max(f1)
                last_f1 = best_f1
                last_acc = acc/len(loader)

        # # compute training scores
        # self._model.eval()
        # with torch.no_grad():
        #     preds = []
        #     for xb, _ in DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
        #         xb = xb.to(self.device)
        #         logit = self._model(xb)
        #         prob = torch.sigmoid(logit)
        #         preds.append(prob.cpu().numpy())
        # scores = np.concatenate(preds, axis=0).reshape(-1)

        # self.decision_scores_ = scores
        # if self.decision_scores_.shape[0] < n_samples:
        #     self.decision_scores_ = np.array(
        #         [self.decision_scores_[0]] * math.ceil((self.slidingWindow - 1) / 2)
        #         + list(self.decision_scores_)
        #         + [self.decision_scores_[-1]] * ((self.slidingWindow - 1) // 2)
        #     )

        return self

    def decision_function(self, X):
        if self.fit_on_loader==0:
            print('STAND: fit_on_loader is 0, please call fit first')
            return None
        elif self.fit_on_loader==1:
            print('STAND has been fitted by "fit_loader" function, please call decision_function_loader function')
            return None

        """Window-level inference: one anomaly score per window.

        Uses non-overlapping windows (stride=win). The model outputs per-step
        logits over the window; we take the sigmoid and then average across
        the window to get a single score per window.
        """
        win = self.slidingWindow

        # Non-overlapping windows for window-level scores
        Xw = Window(window=win, stride=win).convert(X)
        
        if self.normalize:
            Xw = zscore(Xw, axis=1, ddof=1)

        Xw = check_array(Xw)

        # reshape to [num_windows, win, feature_dim]
        feature_dim = Xw.shape[1] // win
        X_seq = Xw.reshape(-1, win, feature_dim)

        num_windows = X_seq.shape[0]
        if num_windows == 0:
            return np.array([], dtype=float)

        self._model.eval()
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.zeros(num_windows, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        window_scores = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logit = self._model(xb)              # [B, win]
                score = torch.sigmoid(logit)           # [B, win]
                window_scores.append(score.cpu().numpy())

        window_scores = np.concatenate(window_scores, axis=0).reshape(-1)
        if len(window_scores) < X.shape[0]:
            window_scores = np.concatenate([window_scores, [window_scores[-1]] * (X.shape[0] - len(window_scores))])
        return window_scores


