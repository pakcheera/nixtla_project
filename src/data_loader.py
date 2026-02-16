import pandas as pd
import numpy as np


class DataLoader:
    """Load and preprocess forecast data."""
    
    def __init__(self, config):
        self.paths = config['paths']
        self.static_feats = config['features']['static_features']
        self.one_hot_encode = config["features"]["one_hot_encode"]
        self.df = None
    
    def _read_csv_with_rename(self, path: str, rename_map: dict) -> pd.DataFrame:
        """Read CSV and rename columns in one step."""
        df = pd.read_csv(path)
        return df.rename(columns=rename_map)
    
    def _ensure_datetime(self, df: pd.DataFrame, date_col: str = 'ds') -> pd.DataFrame:
        """Convert date column to datetime."""
        df[date_col] = pd.to_datetime(df[date_col])
        return df
    
    def _fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing dates with 0 for target column."""
        all_ids = df['unique_id'].unique()
        min_date = df['ds'].min()
        max_date = df['ds'].max()
        full_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        idx = pd.MultiIndex.from_product([all_ids, full_dates], names=['unique_id', 'ds'])
        grid = pd.DataFrame(index=idx).reset_index()
        df = grid.merge(df, on=['unique_id', 'ds'], how='left')
        df['y'] = df['y'].fillna(0)
        return df
    
    def _apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values and apply one-hot encoding."""
        for col in self.one_hot_encode:
            df[col] = df[col].fillna("Unknown")
        df = pd.get_dummies(df, columns=self.one_hot_encode, prefix=self.one_hot_encode, dtype=np.float32, drop_first=True)
        return df
    
    def _align_columns(self, df: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
        """Align columns with template dataframe."""
        template_cols = template_df.columns
        for col in template_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[template_cols]
    
    def load_data(self):
        """Load and process booking, operational capacity, and rules data."""
        # Load bookings
        df_raw = self._read_csv_with_rename(
            self.paths['bookings'],
            {'Cost Center': 'unique_id', 'Day': 'ds', 'Bookings': 'y'}
        )
        df_raw = self._ensure_datetime(df_raw)
        df = self._fill_missing_dates(df_raw)
        
        # Load operational capacity
        oc = self._read_csv_with_rename(
            self.paths['oc_actual'],
            {'Cost Center': 'unique_id', 'Day': 'ds', 'Operational Capacity': 'capacity'}
        )
        oc = self._ensure_datetime(oc)
        
        # Load rules
        rules = self._read_csv_with_rename(
            self.paths['rules_actual'],
            {'Cost Centre': 'unique_id', 'Day': 'ds'}
        )
        rules = self._ensure_datetime(rules)
        
        # Merge all data
        df = df.merge(oc, on=['unique_id', 'ds', "Booking Type"], how='left')
        df = df.merge(rules, on=['unique_id', 'ds'], how='left')
        df = self._apply_one_hot_encoding(df)
        df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")
        
        self.df = df
    
    def get_ml_forecast(self) -> pd.DataFrame:
        """Get data for ML forecasting."""
        return self.df.copy().reset_index(drop=True)
    
    def get_stats_forecast(self) -> pd.DataFrame:
        """Get minimal data for statistical forecasting."""
        return self.df[["unique_id", "ds", "y"]].copy().reset_index(drop=True)
    
    def get_planning_data(self) -> pd.DataFrame:
        """Get planning data with one-hot encoded features aligned to training data."""
        # Load OC and rules for planning
        oc = self._read_csv_with_rename(
            self.paths['oc_plan'],
            {'Cost Center': 'unique_id', 'GeneratedDate': 'ds', 'Operational Capacity': 'capacity'}
        )
        oc = self._ensure_datetime(oc)
        
        rules = self._read_csv_with_rename(
            self.paths['rules_plan'],
            {'Cost Centre': 'unique_id', 'Day': 'ds'}
        )
        rules = self._ensure_datetime(rules)
        
        # Merge and encode
        df_plan = oc.merge(rules, on=['unique_id', 'ds'], how='left')
        df_plan = self._apply_one_hot_encoding(df_plan)
        
        # Align with training data structure
        return self._align_columns(df_plan, self.df)