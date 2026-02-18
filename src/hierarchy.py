import pandas as pd
import numpy as np

from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut
from hierarchicalforecast.utils import aggregate
from hierarchy_spec import get_hierarchy_spec


class Hierarchy:
    def __init__(self, config, data_df=None, freq="D"):
        """
        Initialize Hierarchy using hierarchicalforecast.utils.aggregate method.
        
        Args:
            config: Configuration dictionary
            data_df: DataFrame with time series data
            freq: Frequency of the data (default 'D' for daily)
        """
        self.config = config
        self.freq = freq
        self.data_df = data_df.copy() if data_df is not None else None
        
        # Get hierarchy specification (now dynamic from CSV)
        self.spec = get_hierarchy_spec()
        if not self.spec:
            raise ValueError("Failed to get hierarchy_spec")
        
        # Initialize aggregation results
        self.Y_df = None
        self.S_df = None
        self.tags = None
        self.bottom_ids = None
        
        if data_df is not None:
            self._build_hierarchy_from_data(data_df)
        
        # Reconcilers config
        self.reconcilers_cf = config.get("reconcilers", [])
        
        # Reconciler registry
        self.RECONCILER_REGISTRY = {
            "BottomUp": BottomUp,
            "TopDown": TopDown,
            "MiddleOut": MiddleOut,
        }

    def _build_hierarchy_from_data(self, data_df):
        """
        Build hierarchy using hierarchicalforecast.utils.aggregate().
        
        This creates:
        - Y_df: Aggregated data with all hierarchy levels
        - S_df: Aggregation matrix
        - tags: Dictionary mapping level names to node arrays
        - bottom_ids: List of bottom-level series IDs
        """
        # Apply aggregate to get hierarchical structure
        self.Y_df, self.S_df, self.tags = aggregate(data_df, self.spec)
        
        # Extract bottom-level series (leaf nodes)
        # These are the series that appear in Y_df but not as aggregation ancestors
        all_unique_ids = self.Y_df['unique_id'].unique()
        
        # Bottom series are typically the most detailed aggregation level
        # We can identify them from the tags dictionary - they're in the deepest level
        if self.tags:
            # Get the deepest level (last key when sorted)
            sorted_keys = sorted(self.tags.keys())
            deepest_level = sorted_keys[-1] if sorted_keys else None
            if deepest_level:
                self.bottom_ids = list(self.tags[deepest_level])
            else:
                self.bottom_ids = list(all_unique_ids)
        else:
            self.bottom_ids = list(all_unique_ids)

    def get_hierarchy_data(self):
        """
        Get the aggregated hierarchy data with hierarchy columns.
        
        Returns:
            dict with keys: Y_df, S_df, tags, bottom_ids
            Y_df includes hierarchy columns (Root, Manager, CostCentre) for bottom-level series
        """
        Y_df = self.Y_df.copy()
        
        # For bottom-level series, merge back hierarchy columns from original data
        if self.data_df is not None and 'Root' in self.data_df.columns:
            # Get mapping from original unique_ids (cost centres) to hierarchy columns
            centre_hier = self.data_df[['Root', 'Manager', 'CostCentre']].drop_duplicates()
            
            # Extract cost centre code from the bottom-level unique_id
            # For bottom nodes, unique_id is like "Kylie Van Der Stok/Amanda Judd/E14005"
            def extract_centre_from_uid(uid):
                parts = uid.split('/')
                # Only leaf nodes (cost centre codes) have 3 parts
                if len(parts) == 3:
                    return parts[-1]  # Return the cost centre code
                return None
            
            Y_df['centre_code'] = Y_df['unique_id'].apply(extract_centre_from_uid)
            
            # Merge hierarchy info for bottom-level series only
            Y_df = Y_df.merge(
                centre_hier.rename(columns={'CostCentre': 'centre_code'}),
                on='centre_code',
                how='left'
            )
            # Keep the cost centre code
            Y_df = Y_df.rename(columns={'centre_code': 'CostCentre'})
        
        return {
            "Y_df": Y_df,
            "S_df": self.S_df,
            "tags": self.tags,
            "bottom_ids": self.bottom_ids,
        }

    def build_reconcilers(self):
        """
        Build reconcilers from configuration.
        
        Returns:
            List of reconciler instances
        """
        reconcilers = []
        for i, item in enumerate(self.reconcilers_cf):
            if "type" not in item:
                raise ValueError(f"reconcilers[{i}] missing 'type'")

            r_type = item["type"]
            params = item.get("params", {}) or {}

            if r_type not in self.RECONCILER_REGISTRY:
                raise ValueError(
                    f"Unknown reconciler type '{r_type}'. "
                    f"Allowed: {list(self.RECONCILER_REGISTRY.keys())}"
                )

            cls = self.RECONCILER_REGISTRY[r_type]
            reconcilers.append(cls(**params))

        return reconcilers
