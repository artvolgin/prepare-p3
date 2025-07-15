import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.special import softmax
from collections import defaultdict
from typing import List, Optional, Sequence


class EnsbModel:
    '''
    Age-aware soft-max blending of three base regressors.

    Parameters
    ----------
    age_col : str
        Name of the column in X that contains the age-group label.
    n_splits : int
        Number of folds for cross_val_predict.
    random_state : int
        Seed for KFold shuffling (only used in cross_val_predict).
    '''

    def __init__(self, age_col: str = "age_group",
                 n_splits: int = 5,
                 random_state: int = 42):
        self.age_col = age_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.weights_ = None               # dict[age_group] -> np.array shape (3,)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rmse(y_true, y_pred) -> float:
        '''
        Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

        Parameters
        ----------
        y_true : pd.Series
            The true values.
        y_pred : pd.Series
            The predicted values.

        Returns
        -------
        float
            The RMSE between true and predicted values
        '''
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def _compute_weights(self, y_true, preds, age_groups):
        '''
        Same as before but robust to NaNs / empty slices.

        Parameters
        ----------
        y_true : pd.Series
            The true values.
        preds : list of pd.Series
            The predicted values.
        age_groups : pd.Series
            The age groups.

        Returns
        -------
        dict
            A dictionary with age groups as keys and weights as values
        '''
        weight_table = {}
        
        # Ensure all inputs are aligned to y_true.index
        age_groups_aligned = age_groups.reindex(y_true.index)
        
        for g in age_groups_aligned.unique():
            # Skip NaN age groups
            if pd.isna(g):
                continue
                
            mask = age_groups_aligned == g
            rmses = []
            for p in preds:
                # align index & drop NaNs
                y_pred_g = p.reindex(y_true.index)[mask].dropna()
                y_true_g = y_true[mask].loc[y_pred_g.index]

                if len(y_pred_g) == 0:              # no usable rows
                    rmses.append(np.inf)            # weight → 0
                else:
                    rmses.append(self._rmse(y_true_g, y_pred_g))

            weight_table[g] = softmax(-np.array(rmses))
        return weight_table

    def _blend(self,
               preds_matrix: np.ndarray,  # shape (n_rows, 3)
               age_groups: pd.Series) -> pd.Series:
        '''
        Blend row by row using stored weights_.

        Parameters
        ----------
        preds_matrix : np.ndarray
            The predicted values.
        age_groups : pd.Series
            The age groups.

        Returns
        -------
        pd.Series
        Returns a Series aligned with age_groups.
        '''
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")

        out = np.empty(len(age_groups))
        for i, g in enumerate(age_groups):
            w = self.weights_.get(g)
            if w is None:               # unseen age group → uniform weights
                w = np.full(3, 1/3)
            out[i] = preds_matrix[i] @ w
        return pd.Series(out, index=age_groups.index, name="ensemble_pred")

    # ------------------------------------------------------------------
    # helper to blend ONE row (point, shap-dict, CI)
    # ------------------------------------------------------------------
    @staticmethod
    def _blend_row(
        w: np.ndarray,
        ys: Sequence[float],
        shap_dicts: Optional[List[dict]] = None,
        lowers: Optional[Sequence[float]] = None,
        uppers: Optional[Sequence[float]] = None,
    ):
        '''
        Weighted blend for a single observation.

        Parameters
        ----------
        w : np.ndarray
            The weights.
        ys : Sequence[float]
            The true values.
        shap_dicts : Optional[List[dict]], optional
            The SHAP values.
        lowers : Optional[Sequence[float]], optional
            The lower confidence intervals.
        uppers : Optional[Sequence[float]], optional
            The upper confidence intervals.

        Returns
        -------
        tuple
            The blended prediction, SHAP values, lower confidence interval, and upper confidence interval
        '''
        y_blend = float(np.dot(w, ys))

        # --- CI -------------------------------------------------------
        lower_blend = upper_blend = None
        if lowers is not None and uppers is not None:
            lower_blend = float(np.dot(w, lowers))
            upper_blend = float(np.dot(w, uppers))

        # --- SHAP -----------------------------------------------------
        shap_blend = None
        if shap_dicts is not None:
            shap_agg = defaultdict(float)
            for w_m, d in zip(w, shap_dicts):
                for k, v in d.items():
                    shap_agg[k] += w_m * v
            shap_blend = dict(shap_agg)

        return y_blend, shap_blend, lower_blend, upper_blend

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self,
            X_train: pd.DataFrame,
            y_xgbm_train: pd.Series,
            y_lgbm_train: pd.Series,
            y_catb_train: pd.Series,
            y_train: pd.Series) -> "EnsembleModel":
        '''
        Fit the ensemble model.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training features.
        y_xgbm_train : pd.Series
            The XGBoost predictions.
        y_lgbm_train : pd.Series
            The LightGBM predictions.
        y_catb_train : pd.Series
            The CatBoost predictions.
        y_train : pd.Series
            The true target values.

        Returns
        -------
        EnsbModel
            The fitted ensemble model
        '''
        
        # Align all inputs to y_train.index for consistency
        preds = [
            y_xgbm_train.reindex(y_train.index),
            y_lgbm_train.reindex(y_train.index),
            y_catb_train.reindex(y_train.index),
        ]
        
        # Ensure age_groups is also aligned to y_train.index
        age_groups_aligned = X_train[self.age_col].reindex(y_train.index)
        
        self.weights_ = self._compute_weights(
            y_train, preds, age_groups_aligned
        )
        return self

    def cross_val_predict(self,
                        X_train: pd.DataFrame,
                        xgb_df: pd.DataFrame,
                        lgb_df: pd.DataFrame,
                        cat_df: pd.DataFrame,
                        y_train: pd.Series,
                        groups: pd.Series | None = None) -> pd.DataFrame:
        '''
        Out-of-fold blended predictions incl. SHAP, without CI.
        
        Parameters
        ----------
        X_train : DataFrame
            Training features incl. age column.
        *_df : DataFrame
            Must include columns 'y' and 'shap'.
        y_train : Series
            True target values.
        groups : Series, optional
            Groups for grouped CV.
        
        Returns
        -------
        DataFrame
            ensemble_y, ensemble_shap
            Aligned to training index.
        '''
        # Detect whether SHAP is available
        has_shap = all("shap" in df.columns for df in [xgb_df, lgb_df, cat_df])

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        y_oof, shap_oof = [], []

        y_all = np.column_stack([xgb_df["y"], lgb_df["y"], cat_df["y"]])
        if has_shap:
            shap_dfs = [xgb_df["shap"], lgb_df["shap"], cat_df["shap"]]

        idx = np.arange(len(X_train))

        for tr_idx, val_idx in kf.split(idx, groups=groups):
            preds_tr = [s.iloc[tr_idx] for s in [xgb_df["y"], lgb_df["y"], cat_df["y"]]]
            age_tr = X_train.iloc[tr_idx][self.age_col]
            y_tr = y_train.iloc[tr_idx]
            weights_fold = self._compute_weights(y_tr, preds_tr, age_tr)

            age_val = X_train.iloc[val_idx][self.age_col]
            for i, g in zip(val_idx, age_val):
                w = weights_fold.get(g, np.full(3, 1/3))
                ys = y_all[i, :]

                y_blend = float(np.dot(w, ys))
                y_oof.append((i, y_blend))

                if has_shap:
                    shap_dicts = [df.iloc[i] for df in shap_dfs]
                    shap_agg = defaultdict(float)
                    for w_m, d in zip(w, shap_dicts):
                        for k, v in d.items():
                            shap_agg[k] += w_m * v
                    shap_oof.append((i, dict(shap_agg)))

        cols = ["y"]
        if has_shap:
            cols.append("shap")
        df_out = pd.DataFrame(index=X_train.index, columns=cols)

        for i, yb in y_oof:
            df_out.at[df_out.index[i], "y"] = yb
        if has_shap:
            for i, shapb in shap_oof:
                df_out.at[df_out.index[i], "shap"] = shapb

        return df_out

    def predict(self,
        X_test: pd.DataFrame,
        xgb_df: pd.DataFrame,
        lgb_df: pd.DataFrame,
        cat_df: pd.DataFrame,
    ) -> pd.DataFrame:
        '''
        Blend point predictions always.
        SHAP and CI are blended only if present in **all** model data-frames.

        Parameters
        ----------
        X_test : pd.DataFrame
            The test features.
        xgb_df : pd.DataFrame
            The XGBoost predictions.
        lgb_df : pd.DataFrame
            The LightGBM predictions.
        cat_df : pd.DataFrame
            The CatBoost predictions.

        Returns
        -------
        pd.DataFrame
            The blended predictions
        '''
        # What columns are available?
        has_shap = all("shap" in df.columns for df in [xgb_df, lgb_df, cat_df])
        has_ci   = all({"y_lower", "y_upper"}.issubset(df.columns) for df in [xgb_df, lgb_df, cat_df])

        age_groups = X_test[self.age_col]
        y_out, shap_out, low_out, up_out = [], [], [], []

        for idx, g in age_groups.items():
            w = self.weights_.get(g, np.full(3, 1/3))
            ys = [df.at[idx, "y"] for df in [xgb_df, lgb_df, cat_df]]

            shap_dicts = None
            if has_shap:
                shap_dicts = [df.at[idx, "shap"] for df in [xgb_df, lgb_df, cat_df]]

            lowers = uppers = None
            if has_ci:
                lowers = [df.at[idx, "y_lower"] for df in [xgb_df, lgb_df, cat_df]]
                uppers = [df.at[idx, "y_upper"] for df in [xgb_df, lgb_df, cat_df]]

            y_b, shap_b, lo_b, up_b = self._blend_row(w, ys, shap_dicts, lowers, uppers)
            y_out.append(y_b)
            if has_shap:
                shap_out.append(shap_b)
            if has_ci:
                low_out.append(lo_b)
                up_out.append(up_b)

        data = {"y": y_out}
        if has_shap:
            data["shap"] = shap_out
        if has_ci:
            data["y_lower"] = low_out
            data["y_upper"] = up_out

        return pd.DataFrame(data, index=X_test.index)