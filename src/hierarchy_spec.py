import pandas as pd


def get_hierarchy_spec():
    """Get the aggregation spec for hierarchicalforecast.utils.aggregate()
    
    Returns:
        list of lists defining aggregation levels
    """
    return [
        ["Root"],
        ["Root", "Manager"],
        ["Root", "Manager", "CostCentre"],
    ]


def create_hierarchy_columns(df, csv_path):
    """Add hierarchy columns to dataframe by parsing parent-child CSV.
    
    Args:
        df: DataFrame with 'unique_id' column (Cost Centre code)
        csv_path: Path to hierarchy CSV file with :parent, :child columns
        
    Returns:
        DataFrame with Root, Manager, CostCentre columns added
    """
    df = df.copy()
    
    # Load the hierarchy CSV with parent-child structure
    hierarchy_df = pd.read_csv(csv_path)
    
    # Build mappings from the parent-child structure
    # Format: :parent, :child
    centre_to_manager = {}  # E14005 -> Amanda Judd
    
    for _, row in hierarchy_df.iterrows():
        parent = str(row[':parent']).strip() if pd.notna(row[':parent']) else ""
        child = str(row[':child']).strip()
        
        # Level 2: Manager -> Cost Centre (leaf nodes are codes like E14005)
        if parent and len(child) >= 5 and child[0] in 'EKREWZ':
            # child is a cost centre code
            centre_to_manager[child] = parent
    
    # Add hierarchy columns
    df["Root"] = "Kylie Van Der Stok"
    df["Manager"] = df["unique_id"].map(centre_to_manager)
    df["CostCentre"] = df["unique_id"]
    
    # Fill unmapped values
    df["Manager"] = df["Manager"].fillna("Unknown")
    
    return df
