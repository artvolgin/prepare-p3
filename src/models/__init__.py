"""
Model wrappers with unified API for regression tasks.

All models support:
• optimize_params() - Optuna hyperparameter search (where applicable)
• fit() - Train on full dataset
• cross_val_predict() - Out-of-fold predictions
• predict() - Predictions with optional SHAP values and confidence intervals
"""

from .base_model import BaseModel
from .lgbm_model import LgbmModel
from .xgbm_model import XgbmModel
from .catb_model import CatbModel
from .ensb_model import EnsbModel

__all__ = [
    "BaseModel",
    "LgbmModel", 
    "XgbmModel",
    "CatbModel",
    "EnsbModel"
]
