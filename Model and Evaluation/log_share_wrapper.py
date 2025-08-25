import numpy as np

class LogShareWrapper:
    """
    Wrapper for regression models that predicts log-transformed values
    and converts them back to the original scale (exponential of predicted values).
    """

    def __init__(self, reg_model):
        self.reg_model = reg_model

    def predict(self, X):
        log_preds = self.reg_model.predict(X)
        return np.expm1(np.clip(log_preds, 0, None))

    def fit(self, X, y):
        return self.reg_model.fit(X, y)

    def get_params(self, deep=True):
        return {"reg_model": self.reg_model}

    def set_params(self, **params):
        if "reg_model" in params:
            self.reg_model = params["reg_model"]
        return self
