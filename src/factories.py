# ML Imports
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Stats Imports
from statsforecast.models import AutoARIMA, SeasonalNaive, Naive

class ModelFactory:
    def get_model(model_type: str, **kwargs):
        if 'random_state' not in kwargs:
            kwargs['random_state'] = 42
        if model_type == "LGBMRegressor":
            if 'verbose' not in kwargs: kwargs['verbose'] = -1
            return LGBMRegressor(**kwargs)
        elif model_type == "XGBRegressor":
            return XGBRegressor(**kwargs)
        elif model_type == "RandomForestRegressor":
            if 'n_jobs' not in kwargs: kwargs['n_jobs'] = -1
            return RandomForestRegressor(**kwargs)
        elif model_type == "AutoARIMA":
            return AutoARIMA()
        else:
            raise ValueError(f"Unknown ML model: {model_type}")
