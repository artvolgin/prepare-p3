"""
PipelineRunner: A comprehensive class that encapsulates the entire ML pipeline
from data preprocessing to ensemble model training.

This class is based on the logic from the pipeline_runner.ipynb notebook and
provides a clean, reusable interface for running the complete pipeline.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split

# Local imports
from src.preprocess import DataPreprocessor
from src.models import LgbmModel, XgbmModel, CatbModel, EnsbModel
from src.utils import get_columns_type


class PipelineRunner:
    """
    A comprehensive pipeline runner that handles the entire ML workflow.
    
    This class encapsulates:
    1. Data loading and splitting
    2. Data preprocessing with DataPreprocessor
    3. Individual model training (LGBM, CatBoost, XGBoost)
    4. Ensemble model training
    5. Prediction generation
    6. Model and data persistence
    """
    
    def __init__(
        self,
        run_name: str = "default",
        data_dir: str = "data",
        models_dir: str = "models",
        df_combined_path: str = None,
        variables_type_path: str = None,
        random_state: int = 42,
        n_splits: int = 5,
        test_size: float = 0.2,
        country_holdout: Optional[str] = None,
        holdout_split_mode: str = "random",
        add_logs: bool = False,
        add_group_aggregations: bool = False,
        return_shap: bool = False,
        return_ci: bool = False,
        fair_col: Optional[str] = None
    ):
        '''
        Initialize the PipelineRunner.
        
        Parameters
        ----------
        run_name : str
            Name for this run - used in file names for data and models
        data_dir : str
            Directory containing data files
        models_dir : str
            Directory to save models
        df_combined_path : str, optional
            Exact path to df_combined.pkl file
        variables_type_path : str, optional
            Exact path to table_variables_type.xlsx file
        random_state : int
            Random state for reproducibility
        n_splits : int
            Number of CV splits for models
        test_size : float
            Test set size for train/test split
        country_holdout : str, optional
            Country to hold out for testing (e.g., 'mexico')
        holdout_split_mode : str
            How to split when using country holdout:
            - 'full_holdout': entire country in test, others in train
            - 'partial_holdout': test_size% of holdout country in test, 
                                rest of holdout + all others in train
            - 'random': ignore country, do random split
        add_logs : bool, default=True
            Whether to add log transformations of numerical features
        add_group_aggregations : bool, default=True
            Whether to add group-level aggregation features
        return_shap : bool, default=True
            Whether to return SHAP values for model interpretability
        return_ci : bool, default=True
            Whether to return confidence intervals for predictions
        fair_col : str, optional
            Column name to use for fairness-aware training. If provided, models will
            use inverse-frequency weights to ensure each category contributes equally.
        '''
        self.run_name = run_name
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.df_combined_path = df_combined_path
        self.variables_type_path = variables_type_path
        self.random_state = random_state
        self.n_splits = n_splits
        self.test_size = test_size
        self.country_holdout = country_holdout
        self.holdout_split_mode = holdout_split_mode
        self.add_logs = add_logs
        self.add_group_aggregations = add_group_aggregations
        self.return_shap = return_shap
        self.return_ci = return_ci
        self.fair_col = fair_col
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'lgbm': {'n_trials': 10, 'optimize': True},
            'catb': {'n_trials': 10, 'optimize': True},
            'xgbm': {'n_trials': 10, 'optimize': True}
        }
        
        # Storage for components
        self.data_preprocessor = None
        self.models = {}
        self.ensemble_model = None
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    # ================================================================
    # Data Loading and Splitting
    # ================================================================
    
    def load_and_split_data(
        self, 
        combined_data_path: str = None,
        variables_type_path: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        '''
        Load and split data into train/test sets.
        
        Parameters
        ----------
        combined_data_path : str, optional
            Path to combined dataframe pickle file
        variables_type_path : str, optional
            Path to variables type Excel file
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        '''
        # Use provided paths or fall back to instance/default paths
        if combined_data_path is None:
            combined_data_path = (self.df_combined_path or 
                                self.data_dir / "processed" / "df_combined.pkl")
        if variables_type_path is None:
            variables_type_path = (self.variables_type_path or 
                                 self.data_dir / "processed" / "table_variables_type.xlsx")
            
        print("Loading data...")
        
        # Load data
        df_combined = pd.read_pickle(combined_data_path)
        self.table_variables_type = pd.read_excel(variables_type_path)
        
        # Split features and target
        X = df_combined.drop(columns=['fgcp'])
        y = df_combined['fgcp']
        
        # Split data based on mode
        if self.holdout_split_mode == 'random' or not self.country_holdout:
            print(f"Using random {self.test_size} split...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
        elif self.holdout_split_mode == 'full_holdout':
            print(f"Using full leave-{self.country_holdout}-out split...")
            mask = X['country'] == self.country_holdout
            self.X_train = X[~mask]
            self.y_train = y[~mask]
            self.X_test = X[mask]
            self.y_test = y[mask]
            
        elif self.holdout_split_mode == 'partial_holdout':
            print(f"Using partial {self.country_holdout} holdout split ({self.test_size*100:.0f}% test)...")
            
            # Split holdout country
            holdout_mask = X['country'] == self.country_holdout
            X_holdout = X[holdout_mask]
            y_holdout = y[holdout_mask]
            
            # Split holdout country into train/test
            X_holdout_train, X_holdout_test, y_holdout_train, y_holdout_test = train_test_split(
                X_holdout, y_holdout, test_size=self.test_size, random_state=self.random_state
            )
            
            # Get other countries for training
            X_others = X[~holdout_mask]
            y_others = y[~holdout_mask]
            
            # Combine for final splits
            self.X_train = pd.concat([X_others, X_holdout_train], axis=0)
            self.y_train = pd.concat([y_others, y_holdout_train], axis=0)
            self.X_test = X_holdout_test
            self.y_test = y_holdout_test
            
        else:
            raise ValueError(f"Unknown holdout_split_mode: {self.holdout_split_mode}")
        
        # Save raw splits
        self._save_data(f'X_train_{self.run_name}.pkl', self.X_train)
        self._save_data(f'X_test_{self.run_name}.pkl', self.X_test)
        self._save_data(f'y_train_{self.run_name}.pkl', self.y_train)
        self._save_data(f'y_test_{self.run_name}.pkl', self.y_test)
        
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # ================================================================
    # Data Preprocessing
    # ================================================================
    
    def run_preprocessing(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Run data preprocessing pipeline.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Preprocessed X_train, X_test
        '''
        if self.X_train is None:
            raise ValueError("Data must be loaded first. Call load_and_split_data().")
            
        print("Running data preprocessing...")
        
        # Get column types
        columns_cat, columns_ord, columns_num = get_columns_type(
            self.table_variables_type, 
            pd.concat([self.X_train, self.X_test])
        )
        
        # Initialize preprocessor with specification parameters
        self.data_preprocessor = DataPreprocessor(
            columns_cat=columns_cat,
            columns_ord=columns_ord,
            columns_num=columns_num,
            # group_vars=['country', 'ragender', 'raeducl', 'age_group'],
            target_col='fgcp',
            add_logs=self.add_logs,
            add_group_aggregations=self.add_group_aggregations
        )
        
        # Fit and transform training data - preserve index
        self.X_train = self.data_preprocessor.fit_transform(self.X_train, self.y_train)
        
        # Save preprocessor
        self._save_model(f'data_preprocessor_{self.run_name}.pkl', self.data_preprocessor)
        
        # Transform test data - preserve index
        self.X_test = self.data_preprocessor.transform(self.X_test)
        
        # Save preprocessed data
        self._save_data(f'X_pp_train_{self.run_name}.pkl', self.X_train)
        self._save_data(f'X_pp_test_{self.run_name}.pkl', self.X_test)
        
        feature_info = []
        if self.add_logs:
            feature_info.append("logs")
        if self.add_group_aggregations:
            feature_info.append("group aggregations")
        
        feature_desc = f" (with {', '.join(feature_info)})" if feature_info else ""
        print(f"Preprocessing complete. Features: {self.X_train.shape[1]}{feature_desc}")
        
        return self.X_train, self.X_test
    
    # ================================================================
    # Individual Model Training
    # ================================================================
    
    def train_individual_models(
        self, 
        models_to_train: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        '''
        Train individual models.
        
        Parameters
        ----------
        models_to_train : List[str], optional
            List of models to train. Defaults to ['lgbm', 'catb', 'xgbm']
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with model predictions for train and test sets
        '''
        if self.X_train is None:
            raise ValueError("Data must be prepared first.")
            
        if models_to_train is None:
            models_to_train = ['lgbm', 'catb', 'xgbm']
            
        predictions = {}
        
        for model_name in models_to_train:
            print(f"Training {model_name.upper()} model...")
            predictions.update(self._train_single_model(model_name))
            
        return predictions
    
    def _train_single_model(self, model_name: str) -> Dict[str, pd.DataFrame]:
        '''
        Train a single model and return predictions.

        Parameters
        ----------
        model_name : str
            Name of the model to train

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with model predictions for train and test sets
        '''
        model_classes = {
            'lgbm': LgbmModel,
            'catb': CatbModel,
            'xgbm': XgbmModel
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Initialize model
        model = model_classes[model_name](
            n_splits=self.n_splits,
            fair_col=self.fair_col
        )
        
        config = self.model_configs[model_name]
        best_params = {}
        
        # Optimize hyperparameters if needed
        if config['optimize']:
            print(f"  Optimizing {model_name} hyperparameters...")
            best_params = model.optimize_params(
                self.X_train, self.y_train, n_trials=config['n_trials']
            )
            print("  Parameters optimized")
        
        # Fit model
        print(f"  Fitting {model_name} model...")
        if best_params:
            model.fit(self.X_train, self.y_train, best_params)
        else:
            model.fit(self.X_train, self.y_train)
        
        # Save model
        self._save_model(f'{model_name}_model_{self.run_name}.pkl', model)
        print(f"  {model_name} model saved")
        
        # Generate predictions
        print(f"  Generating {model_name} predictions...")
        
        # OOF predictions - preserve index
        if best_params:
            y_train_pred = model.cross_val_predict(
                self.X_train, self.y_train, best_params, return_shap=self.return_shap
            )
        else:
            y_train_pred = model.cross_val_predict(
                self.X_train, self.y_train, return_shap=self.return_shap
            )
        
        # Test predictions - preserve index
        y_test_pred = model.predict(
            self.X_test,
            return_shap=self.return_shap,
            return_ci=self.return_ci,
            alpha=0.05
        )
        
        # Save predictions
        self._save_data(f'y_{model_name}_train_{self.run_name}.pkl', y_train_pred)
        self._save_data(f'y_{model_name}_test_{self.run_name}.pkl', y_test_pred)
        
        print(f"  {model_name} predictions saved")
        
        # Store model
        self.models[model_name] = model
        
        return {
            f'y_{model_name}_train': y_train_pred,
            f'y_{model_name}_test': y_test_pred
        }
    
    # ================================================================
    # Ensemble Training
    # ================================================================
    
    def train_ensemble(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Train ensemble model.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Ensemble predictions for train and test sets
        '''
        print("Training ensemble model...")
        
        # Load individual model predictions
        predictions = {}
        for model_name in ['lgbm', 'catb', 'xgbm']:
            predictions[f'y_{model_name}_train'] = self._load_data(f'y_{model_name}_train_{self.run_name}.pkl')
            predictions[f'y_{model_name}_test'] = self._load_data(f'y_{model_name}_test_{self.run_name}.pkl')
        
        # Check if we have required columns for ensemble
        required_cols = ['y']
        if self.return_shap:
            required_cols.append('shap')
        if self.return_ci:
            required_cols.extend(['y_lower', 'y_upper'])
        
        # Check if all models have required columns
        test_models = ['xgbm', 'lgbm', 'catb']
        missing_cols = []
        for model in test_models:
            model_df = predictions[f'y_{model}_test']
            missing = [col for col in required_cols if col not in model_df.columns]
            if missing:
                missing_cols.extend(missing)
        
        if missing_cols:
            print(f"Warning: Ensemble training skipped. Missing columns: {set(missing_cols)}")
            print("Ensemble requires all individual models to have the same output format.")
            
            # Return empty DataFrames with proper structure
            empty_cols = ['y']
            if self.return_shap:
                empty_cols.append('shap')
            if self.return_ci:
                empty_cols.extend(['y_lower', 'y_upper'])
                
            empty_train = pd.DataFrame(columns=empty_cols, index=self.X_train.index)
            empty_test = pd.DataFrame(columns=empty_cols, index=self.X_test.index)
            
            return empty_train, empty_test
        
        # Initialize ensemble model
        self.ensemble_model = EnsbModel(age_col="age_group")
        
        # Fit ensemble
        self.ensemble_model.fit(
            self.X_train,
            predictions['y_xgbm_train']["y"],
            predictions['y_lgbm_train']["y"],
            predictions['y_catb_train']["y"],
            self.y_train
        )
        
        # Save ensemble model
        self._save_model(f'ensemble_model_{self.run_name}.pkl', self.ensemble_model)
        
        # Generate ensemble predictions - preserve index
        y_ensb_test = self.ensemble_model.predict(
            self.X_test,
            xgb_df=predictions['y_xgbm_test'],
            lgb_df=predictions['y_lgbm_test'],
            cat_df=predictions['y_catb_test']
        )
        
        y_ensb_train = self.ensemble_model.cross_val_predict(
            self.X_train,
            xgb_df=predictions['y_xgbm_train'],
            lgb_df=predictions['y_lgbm_train'],
            cat_df=predictions['y_catb_train'],
            y_train=self.y_train
        )
        
        # Save ensemble predictions
        self._save_data(f'y_ensb_train_{self.run_name}.pkl', y_ensb_train)
        self._save_data(f'y_ensb_test_{self.run_name}.pkl', y_ensb_test)
        
        print("Ensemble training complete")
        
        return y_ensb_train, y_ensb_test
    
    # ================================================================
    # Full Pipeline Runner
    # ================================================================
    
    def run_full_pipeline(
        self,
        combined_data_path: str = None,
        variables_type_path: str = None,
        models_to_train: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        '''
        Run the complete pipeline from start to finish.
        
        Parameters
        ----------
        combined_data_path : str, optional
            Path to combined dataframe pickle file
        variables_type_path : str, optional
            Path to variables type Excel file
        models_to_train : List[str], optional
            List of models to train
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            All predictions from individual models and ensemble
        '''
        print("=== Starting Full Pipeline ===")
        
        # Step 1: Load and split data
        self.load_and_split_data(combined_data_path, variables_type_path)
        
        # Step 2: Preprocessing
        self.run_preprocessing()
        
        # Step 3: Train individual models
        predictions = self.train_individual_models(models_to_train)
        
        # Step 4: Train ensemble
        y_ensb_train, y_ensb_test = self.train_ensemble()
        predictions['y_ensb_train'] = y_ensb_train
        predictions['y_ensb_test'] = y_ensb_test
        
        # Step 5: Save final predictions dictionary
        self._save_predictions_dict(predictions)
        
        print("=== Pipeline Complete ===")
        
        return predictions
    
    # ================================================================
    # Prediction Methods
    # ================================================================
    
    def predict_new_data(self, X_new: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        '''
        Make predictions on new data using all trained models.
        
        Parameters
        ----------
        X_new : pd.DataFrame
            New data to predict on
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Predictions from all models
        '''
        if self.data_preprocessor is None:
            raise ValueError("Pipeline must be trained first.")
            
        # Preprocess new data - preserve index
        X_processed = self.data_preprocessor.transform(X_new)
        
        predictions = {}
        
        # Individual model predictions - preserve index
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(
                X_processed, 
                return_shap=self.return_shap, 
                return_ci=self.return_ci
            )
        
        # Ensemble prediction - preserve index
        if self.ensemble_model is not None and len(self.models) >= 3:
            # Check if we have required columns for ensemble
            required_cols = ['y']
            if self.return_shap:
                required_cols.append('shap')
            if self.return_ci:
                required_cols.extend(['y_lower', 'y_upper'])
            
            # Only proceed if all models have required columns
            if all(all(col in predictions[model].columns for col in required_cols) 
                   for model in ['xgbm', 'lgbm', 'catb']):
                
                ensemble_pred = self.ensemble_model.predict(
                    X_processed,
                    xgb_df=predictions.get('xgbm', predictions.get('xgb')),
                    lgb_df=predictions.get('lgbm', predictions.get('lgb')),
                    cat_df=predictions.get('catb', predictions.get('cat'))
                )
                predictions['ensemble'] = ensemble_pred
            else:
                print("Warning: Ensemble prediction skipped due to missing SHAP/CI columns")
        
        return predictions
    
    # ================================================================
    # Utility Methods
    # ================================================================
    
    def _save_predictions_dict(self, predictions: Dict[str, pd.DataFrame]):
        '''
        Save the complete predictions dictionary to a pickle file.

        Parameters
        ----------
        predictions : Dict[str, pd.DataFrame]
            Dictionary with model predictions for train and test sets
        '''
        filename = f'predictions_{self.run_name}.pkl'
        filepath = self.data_dir / "predictions" / filename
        
        # Save the predictions dictionary
        with open(filepath, 'wb') as f:
            joblib.dump(predictions, f)
        
        print(f"Final predictions saved to: {filename}")
    
    def _save_data(self, filename: str, data: Union[pd.DataFrame, pd.Series]):
        '''
        Save data to pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to save
        data : Union[pd.DataFrame, pd.Series]
            Data to save
        '''
        filepath = self.data_dir / "processed" / filename
        data.to_pickle(filepath)
    
    def _load_data(self, filename: str) -> Union[pd.DataFrame, pd.Series]:
        '''
        Load data from pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to load

        Returns
        -------
        Union[pd.DataFrame, pd.Series]
            Loaded data
        '''
        filepath = self.data_dir / "processed" / filename
        return pd.read_pickle(filepath)
    
    def _save_model(self, filename: str, model):
        '''
        Save model to pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to load
        model : object
            Model to save
        '''
        filepath = self.models_dir / filename
        joblib.dump(model, filepath)
    
    def _load_model(self, filename: str):
        '''
        Load model from pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to load

        Returns
        -------
        object
            Loaded model
        '''
        filepath = self.models_dir / filename
        return joblib.load(filepath)
