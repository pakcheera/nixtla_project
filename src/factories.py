# Model Factory
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from statsforecast.models import AutoARIMA


class ModelFactory:
    """Factory for creating ML and statistical models with default configurations."""
    
    DEFAULT_PARAMS = {
        'random_state': 42,
    }
    
    MODEL_DEFAULTS = {
        'LGBMRegressor': {'verbose': -1},
        'XGBRegressor': {},
        'RandomForestRegressor': {'n_jobs': -1},
        'AutoARIMA': {},
    }
    
    MODEL_REGISTRY = {
        'LGBMRegressor': LGBMRegressor,
        'XGBRegressor': XGBRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'AutoARIMA': AutoARIMA,
    }
    
    @classmethod
    def get_model(cls, model_type: str, **kwargs):
        """Get a model instance with merged defaults."""
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_type}. Available: {list(cls.MODEL_REGISTRY.keys())}")
        
        # Merge parameters: defaults → model-specific defaults → user kwargs
        params = {**cls.DEFAULT_PARAMS}
        params.update(cls.MODEL_DEFAULTS.get(model_type, {}))
        params.update(kwargs)
        
        # Remove non-applicable params for AutoARIMA
        if model_type == 'AutoARIMA':
            params = {}
        
        return cls.MODEL_REGISTRY[model_type](**params)
