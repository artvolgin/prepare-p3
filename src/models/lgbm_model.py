from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

from .base_model import BaseModel


class LgbmModel(BaseModel):
    '''
    LightGBM regressor with Optuna search (identical logic to your notebook).
    '''

    def __init__(self, n_splits: int = 5, random_state: int = 42, fair_col: str | None = None, verbose: bool = False):
        super().__init__(n_splits, random_state, fair_col)
        self.verbose = verbose

    # --------------------------- model factory -------------------------- #
    def _mk_model(self, params: dict | None = None) -> lgb.LGBMRegressor:
        cfg = dict(
            random_state=self.random_state, 
            n_estimators=100,
            verbosity=-1,  # Suppress all output
            objective="regression",
            metric="rmse",
            boosting_type="gbdt"
        )
        cfg.update(params or {})
        return lgb.LGBMRegressor(**cfg)

    # ----------------------- hyper-parameter search --------------------- #
    def optimize_params(
        self, X, y, n_trials: int = 5
    ) -> dict:
        '''
        Optimize the parameters of the LightGBM regressor using Optuna.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series
            The target variable.
        n_trials : int, optional
            The number of trials to run.
        '''
        def objective(trial: optuna.Trial) -> float:
            param = {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 128),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 3),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "early_stopping_rounds": 50,
                "random_state": self.random_state,  # Fix random seed for each trial
                "feature_fraction_seed": self.random_state,  # Fix feature sampling seed
                "bagging_seed": self.random_state,  # Fix bagging seed
                "data_random_seed": self.random_state  # Fix data sampling seed
            }

            scores = []
            for tr_idx, va_idx in self.kf.split(X):
                mdl = self._mk_model(param)
                
                # Set up callbacks for silent training
                callbacks = [lgb.early_stopping(50)]
                if not self.verbose:
                    callbacks.append(lgb.log_evaluation(0))
                
                mdl.fit(
                    X.iloc[tr_idx],
                    y.iloc[tr_idx],
                    eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                    eval_metric="rmse",
                    callbacks=callbacks,
                )
                preds = mdl.predict(X.iloc[va_idx])
                scores.append(mean_squared_error(y.iloc[va_idx], preds))
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
        # Ensure random_state is always set with all LightGBM seed parameters
        params = {
            **params, 
            "random_state": self.random_state,
            "feature_fraction_seed": self.random_state,
            "bagging_seed": self.random_state,
            "data_random_seed": self.random_state
        }
        
        X_fit = self._prepare_X(X)
        self._feat_cols = X_fit.columns.tolist()

        sw = (
            self._build_sample_weight(X[self.fair_col]) if self.fair_col else None
        )
        self.model = self._mk_model(params)
        
        # Set up callbacks for silent training
        callbacks = [lgb.early_stopping(50)]
        if not self.verbose:
            callbacks.append(lgb.log_evaluation(0))
        
        self.model.fit(
            X_fit, y, 
            sample_weight=sw,
            eval_set=[(X_fit, y)],
            eval_metric="rmse",
            callbacks=callbacks
        )

        self._X_train, self._y_train = X.copy(), y.copy()
        return self 