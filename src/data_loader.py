import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.paths = config['paths']
        self.static_feats = config['features']['static_features']
        self.one_hot_encode = config["features"]["one_hot_encode"]
        

    def optimize_floats(self, df):
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        return df

    def load_data(self):

        df_raw = pd.read_csv(self.paths['bookings'])
        df_raw = df_raw.rename(columns={'Cost Center': 'unique_id', 'Day': 'ds', 'Bookings': 'y'})
        df_raw['ds'] = pd.to_datetime(df_raw['ds'])

        

        all_ids = df_raw['unique_id'].unique()
        min_date = df_raw['ds'].min()
        max_date = df_raw['ds'].max()
        full_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        idx = pd.MultiIndex.from_product([all_ids, full_dates], names=['unique_id', 'ds'])
        grid = pd.DataFrame(index=idx).reset_index()
        df = grid.merge(df_raw, on=['unique_id', 'ds'], how='left')
        df['y'] = df['y'].fillna(0)

     
        oc = pd.read_csv(self.paths['oc_actual']).rename(columns={'Cost Center': 'unique_id', 'Day': 'ds', 'Operational Capacity': 'capacity'})
        oc['ds'] = pd.to_datetime(oc['ds'])


        rules = pd.read_csv(self.paths['rules_actual']).rename(columns={'Cost Centre': 'unique_id', 'Day': 'ds'})
        rules['ds'] = pd.to_datetime(rules['ds'])
        


  
        df = df.merge(oc, on=['unique_id', 'ds',"Booking Type"], how='left')
        df = df.merge(rules, on=['unique_id', 'ds'], how='left')
        for col in self.one_hot_encode:
            df[col] = df[col].fillna("Unknown")
        df = pd.get_dummies(df, columns=self.one_hot_encode, prefix=self.one_hot_encode, dtype=np.float32, drop_first=True)
        

        self.df = df

    def get_ml_forecast(self):
        return self.df.copy().reset_index(drop=True)
    
    def get_stats_forecast(self):
        return self.df[["unique_id", "ds", "y"]].copy().reset_index(drop=True)
    
    def get_planning_data(self):
        oc = pd.read_csv(self.paths['oc_plan']).rename(columns={'Cost Center': 'unique_id', 'GeneratedDate': 'ds', 'Operational Capacity': 'capacity'})
        oc['ds'] = pd.to_datetime(oc['ds'])
        rules = pd.read_csv(self.paths['rules_plan']).rename(columns={'Cost Centre': 'unique_id', 'Day': 'ds'})
        rules['ds'] = pd.to_datetime(rules['ds'])
        df_plan = oc.merge(rules, on=['unique_id', 'ds'], how='left')
        for col in self.one_hot_encode:
            df_plan[col] = df_plan[col].fillna("Unknown")
        df_plan = pd.get_dummies(df_plan, columns=self.one_hot_encode, prefix=self.one_hot_encode, dtype=np.float32, drop_first=True)
        df_columns = self.df.columns
        for col in df_columns:
            if col not in df_plan.columns:
                df_plan[col] = 0.0
        df_plan = df_plan[df_columns]
        return df_plan