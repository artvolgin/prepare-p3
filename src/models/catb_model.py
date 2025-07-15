from __future__ import annotations
import catboost as cb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel


class CatbModel(BaseModel):
    '''
    CatBoost regressor with Optuna search.
    '''

    def __init__(self, n_splits: int = 5, random_state: int = 42, fair_col: str | None = None, verbose: bool = False):
        super().__init__(n_splits, random_state, fair_col)
        self.verbose = verbose

    def _mk_model(self, params=None):
        cfg = dict(
            random_state=self.random_state,
            iterations=100,
            loss_function="RMSE",
            verbose=self.verbose,
            allow_writing_files=False,  # Prevent log files
            train_dir=None  # Prevent creating training directories
        )
        cfg.update(params or {})
        return cb.CatBoostRegressor(**cfg)

    def optimize_params(self, X, y, n_trials: int = 5):
        '''
        Optimize the parameters of the CatBoost regressor using Optuna.

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
                "depth": trial.suggest_int("depth", 4, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 0.5),
                "early_stopping_rounds": 50,
                "verbose": self.verbose,
                "allow_writing_files": False,
                "train_dir": None,
                "random_state": self.random_state  # Fix random seed for each trial
            }
            scores = []
            for tr, va in self.kf.split(X):
                mdl = self._mk_model(param)
                mdl.fit(X.iloc[tr], y.iloc[tr],
                        eval_set=(X.iloc[va], y.iloc[va]),
                        early_stopping_rounds=50,
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
            early_stopping_rounds=50,
            verbose=self.verbose
        )

        self._X_train, self._y_train = X.copy(), y.copy()
        return self
