
from pathlib import Path
import warnings
from data_loader import DataLoader
from features import FeatureEngineer
from utils import *
from mlforecast import MLForecast
from hierarchy import Hierarchy
warnings.filterwarnings("ignore")


def main():
    config = load_config()

    IMG_DIR = Path("images")
    IMG_DIR.mkdir(exist_ok=True, parents=True)

    loader = DataLoader(config)
    loader.load_data()

    fe = FeatureEngineer(config)
    ml_models = build_models(config["model_ml"])
    ml_model_names = list(config["model_ml"].keys())

    # Get raw data for hierarchy aggregation
    raw_data = loader.get_ml_forecast()
    
    # Build hierarchy using aggregate() method
    H = Hierarchy(config, data_df=raw_data)
    hierarchy_data = H.get_hierarchy_data()
    Y_df = hierarchy_data["Y_df"]
    S_df = hierarchy_data["S_df"]
    tags = hierarchy_data["tags"]
    IDS = hierarchy_data["bottom_ids"]

    # Prepare data for MLForecast - keep hierarchy columns but drop them for model training
    # We'll use the hierarchy-enriched data for reconciliation later
    ml_data_full = Y_df.query("unique_id in @IDS").copy()
    
    # For MLForecast, we need only numeric columns + unique_id, ds, y
    # Drop hierarchy columns for the model
    hierarchy_cols = ['Root', 'Manager', 'CostCentre']
    ml_data_for_model = ml_data_full.drop(
        columns=[c for c in hierarchy_cols if c in ml_data_full.columns]
    ).copy()

    ml_forecast = MLForecast(
        models=ml_models,
        freq="D",
        lags=config["features"]["lags"],
        lag_transforms=fe.get_lag_transforms(),
        date_features=config["features"]["date_features"],
        num_threads=config["forecast"]["num_threads"],
    )

    cv_ml = run_cv_and_plot(
        model=ml_forecast,
        data=ml_data_for_model,
        model_names=ml_model_names,
        forecast_cfg=config["forecast"],
        img_prefix="cv_ml",
        img_dir=IMG_DIR,
        ncols=3,
        max_ids=None,  # set e.g. 12 if too many series
        static_features=config["features"].get("static_features"),
    )


    rmse_dict = mean_rmse(cv_ml, ml_model_names)
    best_model_name = min(rmse_dict, key=rmse_dict.get)
    print("\nBest ML model:", best_model_name, "\n")

    CV_OUT_DIR = Path("cv_outputs")
    cv_rec_all, metrics_df = run_cv_reconcile_save_each_window(
        cv_df=cv_ml,
        ml_data=ml_data_full,  # Use full data with hierarchy columns for reconciliation
        best_model_name=best_model_name,
        H=H,
        S_df=S_df,
        tags=tags,
        IDS=IDS,
        out_dir=CV_OUT_DIR,
        img_dir=IMG_DIR,
        save_plots=True,
        save_diagnostics=False,
        print_diagnostics_head=False,
        ncols=3,
        max_ids=None,
        config=config,
        fe=fe
        )



if __name__ == "__main__":
    main()
