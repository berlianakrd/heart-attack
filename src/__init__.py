"""
Indonesia Heart Attack Prediction
Source code package initialization
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

# Import modules untuk memudahkan akses
from . import data_preprocessing
from . import feature_engineering
from . import model_training
from . import model_evaluation

__all__ = [
    'data_preprocessing',
    'feature_engineering',
    'model_training',
    'model_evaluation'
]