import numpy as np
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd, RollingMin, RollingMax

class FeatureEngineer:
    def __init__(self, config):
        self.config = config['features']

    def get_lag_transforms(self):
        transforms = {}
        mapper = {
            "rolling_mean": RollingMean, "rolling_std": RollingStd, 
            "rolling_min": RollingMin, "rolling_max": RollingMax
        }
        for lag, ops in self.config['lag_transforms'].items():
            lag = int(lag)
            tf_list = []
            for i in range(0, len(ops), 2):
                if ops[i] in mapper:
                    tf_list.append(mapper[ops[i]](window_size=ops[i+1]))
            if tf_list:
                transforms[lag] = tf_list
        return transforms

