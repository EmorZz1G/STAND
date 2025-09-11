"""
This function is implemented by EmorZz1G.
"""

from __future__ import division
from __future__ import print_function

import math
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import check_array

from ..base import BaseDetector
from ..feature import Window
from ...utils.utility import zscore


class LightGBM(BaseDetector):
    """Gradient Boosting classifier for supervised time series anomaly detection.
    """

    def __init__(self, slidingWindow=100, sub=True, contamination=0.1, normalize=True,
                 loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0,
                 criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                 max_depth=3, random_state=None, **kwargs):

        self.slidingWindow = slidingWindow
        self.sub = sub
        self.normalize = normalize

        self.clf_ = HistGradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y):
        n_samples, _ = X.shape

        Xw = Window(window=self.slidingWindow).convert(X)
        yw = Window(window=self.slidingWindow).convert(y)
        if self.normalize:
            Xw = zscore(Xw, axis=1, ddof=1)

        Xw = check_array(Xw)

        self.clf_.fit(Xw, yw)

        if hasattr(self.clf_, 'predict_proba'):
            scores = self.clf_.predict_proba(Xw)[:, 1]
        else:
            raw = self.clf_.decision_function(Xw)
            raw_min = np.min(raw)
            raw_max = np.max(raw)
            denom = (raw_max - raw_min) if raw_max != raw_min else 1.0
            scores = (raw - raw_min) / denom

        self.decision_scores_ = scores
        if self.decision_scores_.shape[0] < n_samples:
            self.decision_scores_ = np.array(
                [self.decision_scores_[0]] * math.ceil((self.slidingWindow - 1) / 2)
                + list(self.decision_scores_)
                + [self.decision_scores_[-1]] * ((self.slidingWindow - 1) // 2)
            )

        return self

    def decision_function(self, X):
        n_samples = X.shape[0]
        Xw = Window(window=self.slidingWindow).convert(X)
        if self.normalize:
            Xw = zscore(Xw, axis=1, ddof=1)

        Xw = check_array(Xw)

        if hasattr(self.clf_, 'predict_proba'):
            pred_scores = self.clf_.predict_proba(Xw)[:, 1]
        else:
            raw = self.clf_.decision_function(Xw)
            raw_min = np.min(raw)
            raw_max = np.max(raw)
            denom = (raw_max - raw_min) if raw_max != raw_min else 1.0
            pred_scores = (raw - raw_min) / denom

        if pred_scores.shape[0] < n_samples:
            pred_scores = np.array(
                [pred_scores[0]] * math.ceil((self.slidingWindow - 1) / 2)
                + list(pred_scores)
                + [pred_scores[-1]] * ((self.slidingWindow - 1) // 2)
            )

        return pred_scores

