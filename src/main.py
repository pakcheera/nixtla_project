# main.py
import json
from pathlib import Path
import warnings

from data_loader import DataLoader
from features import FeatureEngineer
from factories import ModelFactory

from statsforecast import StatsForecast
from mlforecast import MLForecast
from utilsforecast.losses import rmse
import matplotlib.pyplot as plt

from utilsforecast.plotting import plot_series
import warnings
warnings.filterwarnings('ignore')


def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_models(model_cfg: dict):
    return [
        ModelFactory.get_model(name, **params)
        for name, params in model_cfg.items()
    ]


def run_cv_and_plot(
    model,
    data,
    model_names,
    forecast_cfg,
    img_prefix,
    img_dir,
    **cv_kwargs,
):
    cv_df = model.cross_validation(
        df=data,
        h=forecast_cfg["horizon"],
        n_windows=forecast_cfg["n_windows"],
        step_size=forecast_cfg["step_size"],
        **cv_kwargs,
    )

    for k, cutoff in enumerate(cv_df["cutoff"].unique()):
        cv = (
            cv_df[cv_df["cutoff"] == cutoff]
            .sort_values(["unique_id", "ds"])
            .drop(columns="cutoff")
        )

        plot = plot_series(
            cv,
            cv.drop(columns="y"),
            models=model_names,
            plot_random=False,
        )
        plot.savefig(img_dir / f"{img_prefix}_{k}.png", bbox_inches="tight")

    return cv_df


def mean_rmse(cv_df, model_names):
    return dict(
        rmse(cv_df, models=model_names, target_col="y")
        .groupby(["cutoff", "unique_id"])
        .mean()
        .reset_index(drop=True)
        .mean()
        .items()
    )



def main():
    config = load_config()

    IDS = ["Z34417", "Z34422"]
    IMG_DIR = Path("images")
    IMG_DIR.mkdir(exist_ok=True)
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)

    loader = DataLoader(config)
    loader.load_data()

    fe = FeatureEngineer(config)

    rmse_dict = {}
    stat_models = build_models(config["model_stat"])
    stat_model_names = list(config["model_stat"].keys())

    stat_data = (
        loader.get_stats_forecast()
        .query("unique_id in @IDS")
    )

    stat_forecast = StatsForecast(
        models=stat_models,
        freq="MS",
        n_jobs=config["forecast"]["num_threads"],
    )

    cv_stat = run_cv_and_plot(
        model=stat_forecast,
        data=stat_data,
        model_names=stat_model_names,
        forecast_cfg=config["forecast"],
        img_prefix="cv_stat",
        img_dir=IMG_DIR,
    )

    rmse_dict.update(mean_rmse(cv_stat, stat_model_names))
    ml_models = build_models(config["model_ml"])
    ml_model_names = list(config["model_ml"].keys())

    ml_data = (
        loader.get_ml_forecast()
        .query("unique_id in @IDS")
    )

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
        data=ml_data,
        model_names=ml_model_names,
        forecast_cfg=config["forecast"],
        img_prefix="cv_ml",
        img_dir=IMG_DIR,
        static_features=config["features"]["static_features"],
    )

    rmse_dict.update(mean_rmse(cv_ml, ml_model_names))
    print("\nRMSE for each model:")
    for name, value in rmse_dict.items():
        print(f"{name}: {value:.4f}")

    print("\nBest model:", min(rmse_dict, key=rmse_dict.get), "\n")
    
    #train best model on all data and save forecasts
    best_model_name = min(rmse_dict, key=rmse_dict.get)
    if best_model_name in stat_model_names:
        best_model = stat_models[stat_model_names.index(best_model_name)]
        best_forecast = StatsForecast(models=[best_model], freq="MS", n_jobs=config["forecast"]["num_threads"])
        best_forecast.fit(stat_data)
        forecast_df = best_forecast.predict(h=config["forecast"]["horizon"])
    else:
        best_model = ml_models[ml_model_names.index(best_model_name)]
        best_forecast = MLForecast(
            models=[best_model],
            freq="D",
            lags=config["features"]["lags"],
            lag_transforms=fe.get_lag_transforms(),
            date_features=config["features"]["date_features"],
            num_threads=config["forecast"]["num_threads"],
        )
        best_forecast.fit(ml_data, static_features=config["features"]["static_features"])
        best_forecast.save(MODEL_DIR / f"best_model_{best_model_name}")
        x_data=loader.get_planning_data()
        x_data =x_data[x_data["unique_id"].isin(IDS)]
        forecast_df = best_forecast.predict(h=config["forecast"]["horizon"],X_df=x_data)
        save_path = MODEL_DIR / f"best_forecast_{best_model_name}.csv"
        forecast_df.to_csv(save_path, index=False)
        
        #plot forecasts for each unique_id
        # y axis angle
        for uid in IDS:
            uid_data = forecast_df[forecast_df["unique_id"] == uid]
            plt.plot(uid_data["ds"], uid_data[best_model_name], label=f"Forecast {uid}")
            plt.title(f"Best Forecast: {best_model_name}")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.xticks(rotation=30)
            plt.legend()
            plt.savefig(IMG_DIR / f"best_forecast_{best_model_name}_{uid}.png", bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    main()
