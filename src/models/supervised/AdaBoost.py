"""
This function is implemented by EmorZz1G.
"""

from __future__ import division
from __future__ import print_function

import math
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import check_array

from ..base import BaseDetector
from ..feature import Window
from ...utils.utility import zscore


class AdaBoost(BaseDetector):
    """AdaBoost classifier for supervised time series anomaly detection.
    """

    def __init__(self, slidingWindow=100, sub=True, contamination=0.1, normalize=True,
                 n_estimators=50, learning_rate=1.0, algorithm='SAMME.R',
                 random_state=None, **kwargs):

        self.slidingWindow = slidingWindow
        self.sub = sub
        self.normalize = normalize

        self.clf_ = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
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

