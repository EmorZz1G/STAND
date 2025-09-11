"""
This function is implemented by EmorZz1G.
"""

from __future__ import division
from __future__ import print_function

import math
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array

from ..base import BaseDetector
from ..feature import Window
from ...utils.utility import zscore


class LR(BaseDetector):
    """Logistic Regression classifier used for supervised time series anomaly detection.

    Converts time series into windowed samples, optionally normalizes windows, fits a
    LogisticRegression classifier, and uses the predicted probability of the anomaly
    class as the anomaly score.
    """

    def __init__(self, slidingWindow=100, sub=True, contamination=0.1, normalize=True,
                 penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, n_jobs=None,
                 class_weight=None, **kwargs):

        self.slidingWindow = slidingWindow
        self.sub = sub
        self.normalize = normalize

        # underlying classifier
        self.clf_ = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            n_jobs=n_jobs,
            class_weight=class_weight,
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

        # Use training proba as decision scores for consistency
        if hasattr(self.clf_, 'predict_proba'):
            scores = self.clf_.predict_proba(Xw)[:, 1]
        elif hasattr(self.clf_, 'decision_function'):
            raw = self.clf_.decision_function(Xw)
            # Min-max normalize to [0,1]
            raw_min = np.min(raw)
            raw_max = np.max(raw)
            denom = (raw_max - raw_min) if raw_max != raw_min else 1.0
            scores = (raw - raw_min) / denom
        else:
            # Fallback to predicted label
            scores = self.clf_.predict(Xw)

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
        elif hasattr(self.clf_, 'decision_function'):
            raw = self.clf_.decision_function(Xw)
            raw_min = np.min(raw)
            raw_max = np.max(raw)
            denom = (raw_max - raw_min) if raw_max != raw_min else 1.0
            pred_scores = (raw - raw_min) / denom
        else:
            pred_scores = self.clf_.predict(Xw)

        if pred_scores.shape[0] < n_samples:
            pred_scores = np.array(
                [pred_scores[0]] * math.ceil((self.slidingWindow - 1) / 2)
                + list(pred_scores)
                + [pred_scores[-1]] * ((self.slidingWindow - 1) // 2)
            )

        return pred_scores

