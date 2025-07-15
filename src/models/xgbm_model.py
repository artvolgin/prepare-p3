from __future__ import annotations
import xgboost as xgb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel


class XgbmModel(BaseModel):
    '''
    XGBoost regressor with Optuna search.
    '''

    def __init__(self, n_splits: int = 5, random_state: int = 42, fair_col: str | None = None, verbose: bool = False):
        super().__init__(n_splits, random_state, fair_col)
        self.verbose = verbose

    def _mk_model(self, params=None):
        '''
        Create an XGBoost regressor with the specified parameters.

        Parameters
        ----------
        params : dict, optional
            Dictionary of parameters to override the default values.
        '''
        cfg = dict(
            random_state=self.random_state,
            n_estimators=100,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=1 if self.verbose else 0,  # 0 = silent, 1 = warning, 2 = info, 3 = debug
            eval_metric="rmse"
        )
        cfg.update(params or {})
        return xgb.XGBRegressor(**cfg)

    def optimize_params(self, X, y, n_trials: int = 5):
        '''
        Optimize the parameters of the XGBoost regressor using Optuna.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series
            The target variable.
        n_trials : int, optional
            The number of trials to run.
        '''
        
        def objective(trial):
            param = {
                "max_depth": trial.suggest_int("max_depth", 4, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
                "verbosity": 1 if self.verbose else 0,
                "eval_metric": "rmse",
                "early_stopping_rounds": 50,
                "random_state": self.random_state  # Fix random seed for each trial
            }
            scores = []
            for tr, va in self.kf.split(X):
                mdl = self._mk_model(param)
                mdl.fit(X.iloc[tr], y.iloc[tr],
                        eval_set=[(X.iloc[va], y.iloc[va])],
                        verbose=self.verbose)
                preds = mdl.predict(X.iloc[va])
                scores.append(mean_squared_error(y.iloc[va], preds))
            return np.mean(scores)

        # Create study with fixed random seed
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        if self.verbose:
            study.optimize(objective, n_trials=n_trials)
        else:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        return self.best_params

    def fit(self, X, y, params=None):
        '''
        Override fit to control verbosity during training.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series
            The target variable.
        params : dict, optional
        '''
        params = params or self.best_params or {}
        # Ensure random_state is always set
        params = {**params, "random_state": self.random_state}
        
        X_fit = self._prepare_X(X)
        self._feat_cols = X_fit.columns.tolist()

        sw = (
            self._build_sample_weight(X[self.fair_col]) if self.fair_col else None
        )
        self.model = self._mk_model(params)
        
        self.model.fit(
            X_fit, y, 
            sample_weight=sw,
            eval_set=[(X_fit, y)],
            verbose=self.verbose
        )

        self._X_train, self._y_train = X.copy(), y.copy()
        return self
