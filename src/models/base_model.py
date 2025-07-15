from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class BaseModel(ABC):
    '''
    Parent class that implements all bookkeeping, data handling, CV helpers,
    SHAP utilities and Jackknife+ prediction intervals.  Sub-classes only need
    to implement two things:

        • _mk_model()          – factory that returns an *unfitted* regressor
        • optimize_params()    – (optional) Optuna hyper-parameter search
    '''

    # ------------------------------------------------------------------ #
    #                           constructor                              #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        fair_col: str | None = None,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.fair_col = fair_col

        # reusable splitter → makes CV folds identical everywhere
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # will be filled by .fit()
        self.model = None
        self._feat_cols: list[str] | None = None
        self._X_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None

        # will be filled by .optimize_params()
        self.best_params: dict | None = None

    # ------------------------------------------------------------------ #
    #                         ─ common helpers ─                          #
    # ------------------------------------------------------------------ #
    def _prepare_X(self, X: pd.DataFrame, *, align: bool = False) -> pd.DataFrame:
        '''
        Drop fairness column and optionally align to training feature order.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        align : bool, optional
            Whether to align the input features to the training feature order.

        Returns
        '''
        X_out = X.drop(columns=[self.fair_col], errors="ignore")
        if align and self._feat_cols is not None:
            X_out = X_out.reindex(columns=self._feat_cols, fill_value=0.0)
        return X_out

    @staticmethod
    def _build_sample_weight(cat_series: pd.Series | None) -> np.ndarray | None:
        '''
        Inverse-frequency weights so each category contributes equally.
        Handles NaNs by treating them as a separate “__MISSING__” category.

        Parameters
        ----------
        cat_series : pd.Series
            The input features.

        Returns
        '''
        if cat_series is None:
            return None

        # Treat NaN as its own category
        filled = cat_series.fillna("__MISSING__")

        # Frequency of each category (including the missing bucket)
        counts = filled.value_counts()

        # Weight = 1 / freq
        return filled.map(lambda c: 1.0 / counts[c]).to_numpy()

    # ---- SHAP helper ---------------------------------------------------
    @staticmethod
    def _compute_shap_dict(model, X: pd.DataFrame) -> pd.Series:
        '''
        Compute SHAP values for the given model and input features.

        Parameters
        ----------
        model : object
            The model to compute SHAP values for.
        X : pd.DataFrame
            The input features.

        Returns
        -------
        pd.Series
            SHAP values for the given model and input features
        '''
        import shap

        explainer = shap.Explainer(model)
        shap_values = explainer(X).values
        dicts = [dict(zip(X.columns, row)) for row in shap_values]
        return pd.Series(dicts, index=X.index, name="shap")

    # ------------------------------------------------------------------ #
    #                    ─ to be supplied by subclasses ─                 #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _mk_model(self, params: dict | None = None):
        '''
        Return *unfitted* regressor configured with `params`.

        Parameters
        ----------
        params : dict, optional
            Dictionary of parameters to override the default values.

        Returns
        -------
        object
            *unfitted* regressor configured with `params`
        '''
        raise NotImplementedError

    def optimize_params(  # optional – raise by default
        self, X: pd.DataFrame, y: pd.Series, n_trials: int = 20
    ) -> dict:
        '''
        Optimize the parameters of the model using Optuna.

        Parameters
        '''
        raise NotImplementedError(
            "This model has no built-in Optuna search. "
            "Either override in subclass or pass params directly to .fit()"
        )

    # ------------------------------------------------------------------ #
    #                               fit                                  #
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series, params: dict | None = None):
        '''
        Train on full data and remember everything for later prediction.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series
            The target variable.
        params : dict, optional
        '''
        params = params or self.best_params or {}
        X_fit = self._prepare_X(X)
        self._feat_cols = X_fit.columns.tolist()

        sw = (
            self._build_sample_weight(X[self.fair_col]) if self.fair_col else None
        )
        self.model = self._mk_model(params)
        self.model.fit(X_fit, y, sample_weight=sw)

        self._X_train, self._y_train = X.copy(), y.copy()
        return self

    # ------------------------------------------------------------------ #
    #                     out-of-fold prediction helper                  #
    # ------------------------------------------------------------------ #
    def _fit_one(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, params: dict
    ):
        '''
        Fit a single model on the training data.

        Parameters
        ----------
        X_tr : pd.DataFrame
            The input features.
        y_tr : pd.Series
            The target variable.
        params : dict, optional
        '''
        sw = (
            self._build_sample_weight(X_tr[self.fair_col])
            if self.fair_col
            else None
        )
        mdl = self._mk_model(params)
        mdl.fit(self._prepare_X(X_tr), y_tr, sample_weight=sw)
        return mdl

    def cross_val_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict | None = None,
        *,
        return_shap: bool = False,
    ) -> pd.DataFrame:
        '''
        Return DF with OOF preds (+ SHAP dicts if requested).

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series
            The target variable.
        params : dict, optional
        return_shap : bool, optional
            Whether to return SHAP values.

        Returns
        -------
        pd.DataFrame
            DataFrame with OOF predictions and SHAP values
        '''
        params = params or self.best_params or {}
        oof = np.zeros(len(X))
        shap_series = pd.Series([None] * len(X), index=X.index)

        for tr_idx, va_idx in self.kf.split(X):
            mdl = self._fit_one(X.iloc[tr_idx], y.iloc[tr_idx], params)
            X_val_prep = self._prepare_X(X.iloc[va_idx], align=True)
            oof[va_idx] = mdl.predict(X_val_prep)

            if return_shap:
                shap_series.iloc[va_idx] = self._compute_shap_dict(mdl, X_val_prep)

        out = pd.DataFrame({"y": oof}, index=X.index)
        if return_shap:
            out["shap"] = shap_series
        return out

    # ------------------------------------------------------------------ #
    #                             predict                                #
    # ------------------------------------------------------------------ #
    def predict(
        self,
        X: pd.DataFrame,
        *,
        return_shap: bool = False,
        return_ci: bool = False,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        '''
        Predict on new data, optionally with SHAP values and Jackknife+ CIs.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        return_shap : bool, optional
            Whether to return SHAP values.
        return_ci : bool, optional
            Whether to return Jackknife+ CIs.
        alpha : float, optional
            The significance level for the confidence intervals.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions, SHAP values, and Jackknife+ CIs
        '''
        if self.model is None:
            raise ValueError("Call fit() before predict().")

        X_in = self._prepare_X(X, align=True)
        preds = self.model.predict(X_in)
        out = pd.DataFrame({"y": preds}, index=X.index)

        # ---- SHAP -------------------------------------------------------
        if return_shap:
            out["shap"] = self._compute_shap_dict(self.model, X_in)

        # ---- Jackknife+ -------------------------------------------------
        if return_ci:
            if self._X_train is None:
                raise ValueError("Fit must be called before predict(return_ci=True).")

            residuals = np.empty(len(self._X_train))
            fold_models = []

            for tr_idx, va_idx in self.kf.split(self._X_train):
                mdl = self._fit_one(
                    self._X_train.iloc[tr_idx],
                    self._y_train.iloc[tr_idx],
                    self.best_params or {},
                )
                fold_models.append(mdl)

                X_val_prep = self._prepare_X(self._X_train.iloc[va_idx], align=True)
                residuals[va_idx] = (
                    self._y_train.iloc[va_idx] - mdl.predict(X_val_prep)
                )

            # prediction matrix (k_folds × n_test)
            preds_matrix = np.vstack([m.predict(X_in) for m in fold_models])
            resid = residuals.reshape(1, -1)  # (1, n_train)

            lower_all = np.concatenate(preds_matrix[:, :, None] - resid, axis=1)
            upper_all = np.concatenate(preds_matrix[:, :, None] + resid, axis=1)

            out["y_lower"] = np.quantile(lower_all, alpha / 2, axis=1)
            out["y_upper"] = np.quantile(upper_all, 1 - alpha / 2, axis=1)

        return out 