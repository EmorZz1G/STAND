from sklearn.neighbors import KNeighborsClassifier as KNN_
from sklearn.svm import SVC as SVM_
from sklearn.linear_model import LogisticRegression as LR_
from sklearn.ensemble import RandomForestClassifier as RF_, GradientBoostingClassifier as GBM_
from sklearn.ensemble import HistGradientBoostingClassifier as LightGBM_
from sklearn.ensemble import AdaBoostClassifier as AdaBoost_
from sklearn.ensemble import ExtraTreesClassifier as ExtraTrees_

from ..feature import Window
import numpy as np

class SupervisedModel:
    def fit(self, X, y):
        print(X.shape, y.shape)
        X_windows = Window(self.win_size).convert(X)
        y = y[self.win_size - 1:]
        print(X_windows.shape, y.shape)
        self.detector_.fit(X_windows, y)
        return self
    
    def decision_function(self, X):
        # 在X前面填充win_size-1个0
        X = np.concatenate([np.zeros((self.win_size - 1, X.shape[1])), X], axis=0)
        X_windows = Window(self.win_size).convert(X)
        return self.decision_function_(X_windows)

class KNN(SupervisedModel):
    def __init__(self, win_size=3,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=-1):
        self.detector_ = KNN_(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        self.win_size = win_size
    
    def decision_function_(self, X):
        return self.detector_.predict_proba(X)[:, 1]
    
class SVM(SupervisedModel):
    def __init__(self, win_size=3, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0, shrinking=True, probability=True, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape="ovr", break_ties=False, random_state=None):
        self.detector_ = SVM_(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state)
        self.win_size = win_size
    def decision_function_(self, X):
        return self.detector_.decision_function(X)
    
class LR(SupervisedModel):
    def __init__(self, win_size=3, penalty="l2", dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None):
        self.detector_ = LR_(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)
        self.win_size = win_size
    def decision_function_(self, X):
        return self.detector_.decision_function(X)

class RF(SupervisedModel):
    def __init__(self, win_size=3, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.detector_ = RF_(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        self.win_size = win_size

    def decision_function_(self, X):
        return self.detector_.predict_proba(X)[:, 1]

class AdaBoost(SupervisedModel):
    def __init__(self, win_size=3,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        algorithm="SAMME.R",
        random_state=None):
        self.detector_ = AdaBoost_(estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)
        self.win_size = win_size

    def decision_function_(self, X):
        return self.detector_.predict_proba(X)[:, 1]

class ExtraTrees(SupervisedModel):
    def __init__(self, win_size=3, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.detector_ = ExtraTrees_(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        self.win_size = win_size

    def decision_function_(self, X):
        return self.detector_.predict_proba(X)[:, 1]

class LightGBM(SupervisedModel):
    def __init__(self, win_size=3, 
        loss="log_loss",
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features="warn",
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        class_weight=None):
        self.detector_ = LightGBM_(loss=loss, learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization, max_features=max_features, max_bins=max_bins, categorical_features=categorical_features, monotonic_cst=monotonic_cst, interaction_cst=interaction_cst, warm_start=warm_start, early_stopping=early_stopping, scoring=scoring, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, verbose=verbose, random_state=random_state, class_weight=class_weight)
        self.win_size = win_size

    def decision_function_(self, X):
        return self.detector_.predict_proba(X)[:, 1]