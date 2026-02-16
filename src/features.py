import numpy as np
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd, RollingMin, RollingMax


class FeatureEngineer:
    """Feature engineering for time series forecasting."""
    
    TRANSFORM_MAPPER = {
        "rolling_mean": RollingMean,
        "rolling_std": RollingStd,
        "rolling_min": RollingMin,
        "rolling_max": RollingMax,
    }
    
    def __init__(self, config):
        self.config = config['features']
    
    def _parse_lag_ops(self, ops: list) -> list:
        """Parse lag operations into transform instances."""
        tf_list = []
        for i in range(0, len(ops), 2):
            if ops[i] in self.TRANSFORM_MAPPER:
                transform_class = self.TRANSFORM_MAPPER[ops[i]]
                window_size = ops[i + 1]
                tf_list.append(transform_class(window_size=window_size))
        return tf_list
    
    def get_lag_transforms(self) -> dict:
        """Get lag transforms grouped by lag."""
        transforms = {}
        for lag, ops in self.config['lag_transforms'].items():
            lag = int(lag)
            tf_list = self._parse_lag_ops(ops)
            if tf_list:
                transforms[lag] = tf_list
        return transforms

