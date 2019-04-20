from lifelines import AalenAdditiveFitter
from lifelines import CoxPHFitter
from lifelines import utils
from sklearn.base import BaseEstimator

class CoxPHFitterModel(BaseEstimator):
    def __init__(self, duration_column=None, event_col=None, initial_point=None, strata=None, alpha=0.95, tie_method='Efron', penalizer=0.0, **kwargs):
        self.alpha = alpha
        self.tie_method = tie_method
        self.penalizer = penalizer

        self.duration_column = duration_column
        self.event_col = event_col

        self.initial_point = initial_point
        self.strata = strata


    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        X_[self.duration_column]=y[self.duration_column]
        if self.event_col is not None:
            X_[self.event_col] = y[self.event_col]

        est = CoxPHFitter(alpha=self.alpha, tie_method=self.tie_method, penalizer=self.penalizer)

        est.fit(X_, duration_col=self.duration_column, event_col=self.event_col, initial_point=self.initial_point, strata=self.strata, **fit_params)
        self.estimator = est
        return self

    def predict(self, X):
        return self.estimator.predict_expectation(X)

    def score(self, X, y):
        """Calculate score based on concordance index."""
        partial_hazard = self.estimator.predict_partial_hazard(X)[0]
        if self.event_col is None:
            return utils.concordance_index(y[self.duration_column], -partial_hazard)
        else:
            return utils.concordance_index(y[self.duration_column], -partial_hazard, y[self.event_col])


class AalenAdditiveFitterModel(BaseEstimator):

    def __init__(self, duration_column=None, event_col=None, weights_col=None, show_progress=None, fit_intercept=True, alpha=0.95, coef_penalizer=0.5, smoothing_penalizer=0.0,**kwargs):
        self.fit_intercept=fit_intercept
        self.alpha=alpha
        self.coef_penalizer=coef_penalizer
        self.smoothing_penalizer=smoothing_penalizer

        self.duration_column = duration_column
        self.event_col = event_col
        self.weights_col = weights_col
        self.show_progress = show_progress

    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        X_[self.duration_column]=y[self.duration_column]
        if self.event_col is not None:
            X_[self.event_col] = y[self.event_col]

        est = AalenAdditiveFitter(fit_intercept=self.fit_intercept, alpha=self.alpha, coef_penalizer=self.coef_penalizer,
                                  smoothing_penalizer=self.smoothing_penalizer)
        est.fit(X_, duration_col=self.duration_column, event_col=self.event_col, weights_col=self.weights_col,
                show_progress=self.show_progress, **fit_params)
        self.estimator = est
        return self

    def predict(self, X):
        return self.estimator.predict_expectation(X)

    def score(self, X, y):
        """Calculate score based on concordance index."""
        predicted_hazards = self.estimator.predict_cumulative_hazard(X).iloc[-1]
        if self.event_col is None:
            return utils.concordance_index(y[self.duration_column], -predicted_hazards)
        else:
            return utils.concordance_index(y[self.duration_column], -predicted_hazards, y[self.event_col])
