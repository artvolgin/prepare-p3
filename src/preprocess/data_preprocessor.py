import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from pandas.errors import PerformanceWarning
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.model_selection import KFold
# Add imports for advanced statistics
from scipy.stats import entropy, hmean, gmean
from scipy.signal import find_peaks

warnings.simplefilter(action="ignore", category=PerformanceWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


class DataPreprocessor:
    def __init__(self, columns_cat=None, columns_ord=None, columns_num=None, group_vars=None, target_col='fgcp',
                 add_logs=True, add_group_aggregations=True):
        """
        Parameters
        ----------
        columns_cat : list of str
            Names of categorical columns to one-hot encode.
        columns_ord : list of str
            Names of ordinal columns to convert to numeric.
        columns_num : list of str
            Names of numeric columns to convert and log-transform.
        group_vars : list of str
            List of columns to group by for advanced statistics.
        target_col : str
            Name of the target column for advanced statistics.
        add_logs_and_squared : bool, default=True
            Whether to add log and squared transformations of numerical features.
        add_group_aggregations : bool, default=True
            Whether to add group-level aggregation features.
        """
        self.columns_cat = columns_cat or []
        self.columns_ord = columns_ord or []
        self.columns_num = columns_num or []
        # self.group_vars = ['country', 'ragender', 'raeducl', 'age_group']
        self.group_vars = ['country', 'age_group']
        self.target_col = target_col
        self.add_logs = add_logs
        self.add_group_aggregations = add_group_aggregations
        self.dummies_encoder = None
        self.grouped_stats_ = None  # Store fitted grouped statistics

    def fit(self, X, y=None):
        '''
        Fit encoders on categorical columns and calculate grouped statistics.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series, optional
            The target variable.
        '''
        # 0. CREATE engineered columns needed for grouping
        # (age_group  &  integer country)
        X_fit = X.copy()

        if "r5agey" in X_fit.columns:
            X_fit["age_group"] = pd.cut(
                X_fit["r5agey"],
                bins=[float("-inf"), 60, 70, 80, float("inf")],
                labels=False,          # → integers 0-3
            )

        if "country" in X_fit.columns:
            X_fit["country"] = pd.Categorical(X_fit["country"]).codes + 1

        # 1. Apply ordinal transformations to ensure consistent dtypes
        for col in self.columns_ord:
            if col in X_fit.columns and not np.issubdtype(X_fit[col].dtype, np.floating):
                if X_fit[col].dtype == "object":  # string → extract digits
                    X_fit[col] = (
                        X_fit[col].astype(str)
                                  .str.extract(r"^(\d+)")[0]
                                  .astype(float)
                    )
                else:                              # already numeric
                    X_fit[col] = X_fit[col].astype(float)

        # 2. fit one-hot encoder on (possibly cleaned) categorical cols
        if self.columns_cat:
            for col in self.columns_cat:
                if X_fit[col].dtype == "object":  # string → keep only leading digits
                    X_fit[col] = (
                        X_fit[col]
                        .str.extract(r"^(\d+)")[0]
                        .astype(float)
                        .astype(str)
                    )
                else:                              # already numeric
                    X_fit[col] = X_fit[col].astype(float).astype(str)
            
            self.dummies_encoder = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            )
            self.dummies_encoder.fit(X_fit[self.columns_cat])
        
        # 3. GROUP-LEVEL statistics (age_group now exists)
        if self.add_group_aggregations and self.group_vars and y is not None:
            df_with_target = X_fit.copy()
            df_with_target[self.target_col] = y
            self.grouped_stats_ = self._calculate_advanced_stats(
                df_with_target, self.group_vars
            )
        
        return self

    def transform(self, X):
        '''
        Transform dataset with additional age group and country encodings

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        '''
        X = X.copy()

        # Add age groups as integers (0: <60, 1: 60-70, 2: 70-80, 3: 80+)
        if 'r5agey' in X.columns:
            X['age_group'] = pd.cut(
                X['r5agey'],
                bins=[float('-inf'), 60, 70, 80, float('inf')],
                labels=False  # This will create 0-based integer labels
            )

        # Transform country to integers (1-based)
        if 'country' in X.columns:
            # Create a categorical encoding and convert to 1-based integers
            X['country'] = pd.Categorical(X['country']).codes + 1

        # 1. basic numeric handling with optional log and squared features
        for col in self.columns_num:
            X[col] = X[col].astype(float)
            
            # Add log and squared features if enabled
            if self.add_logs:
                X[f'log_{col}'] = np.log1p(X[col])
                # X[f'squared_{col}'] = X[col] ** 2

        # 2. categorical / ordinal cleanup (digits extraction)
        for col in self.columns_cat:
            if X[col].dtype == 'object':  # Only apply string operations if column is string type
                X[col] = (
                    X[col].str.extract(r'^(\d+)')[0]
                           .astype(float)
                           .astype(str)
                )
            else:  # If column is already numeric
                X[col] = X[col].astype(float).astype(str)

        for col in self.columns_ord:
            if col in X.columns and not np.issubdtype(X[col].dtype, np.floating):
                if X[col].dtype == 'object':  # Only apply string operations if column is string type
                    X[col] = (
                        X[col].astype(str)
                              .str.extract(r'^(\d+)')[0]
                              .astype(float)
                    )
                else:  # If column is already numeric
                    X[col] = X[col].astype(float)

        # 3. add advanced-stats BEFORE we drop group_vars (if enabled)
        if self.add_group_aggregations and self.grouped_stats_ is not None and self.group_vars:
            X = self._add_advanced_stats_features(X)

        # 4. one-hot encode categorical columns (then drop originals)
        if self.columns_cat and self.dummies_encoder is not None:
            dummies = self.dummies_encoder.transform(X[self.columns_cat])
            dummy_cols = self.dummies_encoder.get_feature_names_out(self.columns_cat)
            X = pd.concat(
                [X.drop(columns=self.columns_cat),
                 pd.DataFrame(dummies, columns=dummy_cols, index=X.index)],
                axis=1
            )

        # 5. flags for "present in wave"
        for wave in range(1, 6):
            col = f"r{wave}agey"
            if col in X.columns:
                X[f"r{wave}present"] = X[col].notna().astype(int)
    
        # 6. clean feature names to be compatible with models like LightGBM
        X = self._clean_feature_names(X)

        return X

    def fit_transform(self, X, y=None):
        '''
        Fit then transform.

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series, optional
            The target variable.
        '''
        self.fit(X, y)
        return self.transform(X)
    
    def _clean_feature_names(self, df):
        '''
        Clean feature names to be compatible with LightGBM and other models.
        Replaces special characters with underscores.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with potentially problematic column names.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with cleaned column names.
        '''
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        return df_clean

    def _calculate_advanced_stats(self, df, group_vars):
        '''
        Calculate advanced statistical features for the given DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data.
        group_vars : list
            List of columns to group by.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with advanced statistical features.
        '''
        # Calculate standard aggregations separately
        grouped_stats = df.groupby(group_vars)[self.target_col].agg([
            'min', 'max', 'mean', 'std', 'skew', 'median'
        ]).reset_index()
        
        # Custom aggregations - apply separately
        grouped_stats['range'] = df.groupby(group_vars)[self.target_col].apply(lambda x: x.max() - x.min()).values
        grouped_stats['iqr'] = df.groupby(group_vars)[self.target_col].apply(lambda x: x.quantile(0.75) - x.quantile(0.25)).values
        grouped_stats['mad'] = df.groupby(group_vars)[self.target_col].apply(lambda x: (x - x.mean()).abs().mean()).values  # Manual calculation of MAD
        grouped_stats['cv'] = df.groupby(group_vars)[self.target_col].apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0).values
        grouped_stats['gini'] = df.groupby(group_vars)[self.target_col].apply(lambda x: sum([abs(i - j) for i in x for j in x]) / (2 * len(x) * sum(x)) if sum(x) != 0 else 0).values
        grouped_stats['entropy'] = df.groupby(group_vars)[self.target_col].apply(lambda x: entropy(pd.Series(x).value_counts(normalize=True))).values
        grouped_stats['harmonic_mean'] = df.groupby(group_vars)[self.target_col].apply(lambda x: hmean(x) if all(x > 0) else 0).values
        grouped_stats['geometric_mean'] = df.groupby(group_vars)[self.target_col].apply(lambda x: gmean(x) if all(x > 0) else 0).values
        grouped_stats['peaks'] = df.groupby(group_vars)[self.target_col].apply(lambda x: len(find_peaks(x)[0])).values
        grouped_stats['troughs'] = df.groupby(group_vars)[self.target_col].apply(lambda x: len(find_peaks(-x)[0])).values

        # Calculate quintiles separately
        quintile_labels = ['q1_20', 'q2_40', 'q3_60', 'q4_80']
        quintiles_df = df.groupby(group_vars)[self.target_col].apply(
            lambda x: pd.Series(x.quantile([0.2, 0.4, 0.6, 0.8]).values)
        ).unstack()
        quintiles_df.columns = quintile_labels
        quintiles_df.reset_index(inplace=True)

        # Merge quintiles with grouped stats
        grouped_stats = grouped_stats.merge(quintiles_df, on=group_vars, how='left')
        
        return grouped_stats

    def _add_advanced_stats_features(self, X):
        '''
        Add advanced statistics features to the DataFrame based on fitted grouped_stats_.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to add features to.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added advanced statistics features.
        '''
        # Merge with grouped statistics
        X_with_stats = X.merge(self.grouped_stats_, on=self.group_vars, how='left')
        
        # Add prefix to advanced stats columns to distinguish them
        stats_columns = [col for col in X_with_stats.columns if col not in X.columns]
        X_with_stats = X_with_stats.rename(columns={col: f'stats_{col}' for col in stats_columns})
        
        return X_with_stats

    def advanced_stats(self, df, group_vars):
        '''
        Calculate advanced statistical features for the given DataFrame.
        This is a standalone method for external use.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data.
        group_vars : list
            List of columns to group by.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with advanced statistical features.
        '''
        return self._calculate_advanced_stats(df, group_vars)
