
# utils.py - Hierarchical Forecasting Utilities
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from factories import ModelFactory
from utilsforecast.losses import rmse
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.utils import aggregate
from mlforecast import MLForecast
from hierarchicalforecast.evaluation import evaluate


# ==================== CONFIG & MODELS ====================

def load_config(path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_models(model_cfg: dict):
    """Build ML models from config dictionary."""
    return [ModelFactory.get_model(name, **params) for name, params in model_cfg.items()]


# ==================== DATA UTILITIES ====================

def flatten_spec(spec: list[list[str]]) -> list[str]:
    """Extract unique hierarchy column names from spec."""
    cols = []
    for lvl in spec:
        for c in lvl:
            if c not in cols:
                cols.append(c)
    return cols


def build_eval_tags(tags: dict) -> dict:
    """Auto-create evaluation groups by hierarchy level."""
    keys = list(tags.keys())
    
    def depth(k: str) -> int:
        if k.startswith("level_") and k.split("_")[-1].isdigit():
            return int(k.split("_")[-1])
        return k.count("/")
    
    keys_sorted = sorted(keys, key=depth)
    total_key = keys_sorted[0]
    bottom_key = keys_sorted[-1]
    
    eval_tags = {"total": tags[total_key], "bottom": tags[bottom_key]}
    if len(keys_sorted) >= 2:
        eval_tags["mid"] = tags[keys_sorted[1]]
    return eval_tags


def shorten_reconciler_names(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    """Shorten long reconciler column names for readability."""
    rename_map = {}
    for col in df.columns:
        if col.startswith(f"{base_col}/"):
            parts = col.split("/")[1:]
            method = parts[0]
            
            if method == "BottomUp":
                short_name = f"{base_col}/BottomUp"
            elif method.startswith("TopDown"):
                if "forecast_proportions" in method:
                    short_name = f"{base_col}/TD_FP"
                elif "proportion_averages" in method:
                    short_name = f"{base_col}/TD_PA"
                else:
                    short_name = f"{base_col}/TD"
            elif method.startswith("MiddleOut"):
                if "forecast_proportions" in method:
                    short_name = f"{base_col}/MO_FP"
                elif "proportion_averages" in method:
                    short_name = f"{base_col}/MO_PA"
                else:
                    short_name = f"{base_col}/MO"
            else:
                short_name = col
            
            if short_name in rename_map.values():
                counter = 2
                while f"{short_name}_{counter}" in rename_map.values():
                    counter += 1
                short_name = f"{short_name}_{counter}"
            
            rename_map[col] = short_name
    
    return df.rename(columns=rename_map)


# ==================== PLOTTING ====================

def ensure_hier_cols(df: pd.DataFrame, hier_cols: list[str], H=None) -> pd.DataFrame:
    """Ensure df has hierarchy columns; if not available, raise error."""
    missing_cols = [c for c in hier_cols if c not in df.columns]
    
    if not missing_cols:
        return df
    
    # If hierarchy columns are missing, we can't proceed with spec-based aggregation
    raise ValueError(
        f"Missing hierarchy columns in data: {missing_cols}. "
        f"Ensure data_loader merges hierarchy columns before aggregation."
    )


def get_fitted_values_or_none(ml_forecast: MLForecast, best_model_name: str) -> pd.DataFrame | None:
    """Try to obtain in-sample fitted values for MinTrace/ERM."""
    if hasattr(ml_forecast, "predict_in_sample"):
        fitted = ml_forecast.predict_in_sample()
        pred_col = None
        for c in ["y_hat", "y_pred", "y", best_model_name]:
            if c in fitted.columns and c not in ["unique_id", "ds"]:
                pred_col = c
                break
        if pred_col is None:
            num_cols = [c for c in fitted.columns if c not in ["unique_id", "ds"]]
            pred_col = num_cols[-1]
        fitted = fitted.rename(columns={pred_col: best_model_name})
        return fitted[["unique_id", "ds", best_model_name]]
    return None


def filter_reconcilers_if_no_fitted(reconcilers: list, has_fitted: bool) -> list:
    """Filter reconcilers that need residual covariance if no fitted values available."""
    if has_fitted:
        return reconcilers
    bad_names = {"MinTrace", "ERM"}
    return [r for r in reconcilers if r.__class__.__name__ not in bad_names]


def debug_identity(Y_rec_df: pd.DataFrame, base_col: str):
    """Warn if reconciliation columns equal base forecast."""
    rec_cols = [c for c in Y_rec_df.columns if c.startswith(f"{base_col}/")]
    if not rec_cols:
        return
    base = Y_rec_df[base_col]
    for c in rec_cols:
        diff = (Y_rec_df[c] - base).abs().sum()
        if diff == 0:
            print(f"[DEBUG] {c} is IDENTICAL to base -> reconciliation did not change forecasts.")


def load_config(path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_models(model_cfg: dict):
    """Build ML models from config dictionary."""
    return [ModelFactory.get_model(name, **params) for name, params in model_cfg.items()]


def plot_cutoff_subplots(
    df_window: pd.DataFrame,   # must have: unique_id, ds, y + pred cols
    pred_cols: list[str],      # columns to plot besides y
    title: str,
    out_path: Path,
    ncols: int = 3,
    max_ids: int | None = None,
):
    out_path.parent.mkdir(exist_ok=True, parents=True)

    df = df_window.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    ids = sorted(df["unique_id"].unique())
    if max_ids is not None:
        ids = ids[:max_ids]

    n = len(ids)
    if n == 0:
        return

    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 3 * nrows),
        sharex=False,
    )
    axes = np.array(axes).reshape(-1)

    for i, uid in enumerate(ids):
        ax = axes[i]
        u = df[df["unique_id"] == uid].sort_values("ds")

        ax.plot(u["ds"], u["y"], label="y")
        for c in pred_cols:
            if c in u.columns:
                ax.plot(u["ds"], u[c], label=c)

        ax.set_title(str(uid))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=30)

        # legend only on first subplot
        if i == 0:
            ax.legend(fontsize=8)
        else:
            lg = ax.get_legend()
            if lg:
                lg.remove()

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# CV + Plot (one image per cutoff; all ids as subplots)
# -------------------------
def run_cv_and_plot(
    model,
    data: pd.DataFrame,
    model_names: list[str],
    forecast_cfg: dict,
    img_prefix: str,
    img_dir: Path,
    ncols: int = 3,
    max_ids: int | None = None,
    **cv_kwargs,
):
    img_dir.mkdir(exist_ok=True, parents=True)

    # Clean data to avoid duplicate id-time
    data = data.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data = (
        data.sort_values(["unique_id", "ds"])
        .drop_duplicates(subset=["unique_id", "ds"], keep="last")
    )

    cv_df = model.cross_validation(
        df=data,
        h=forecast_cfg["horizon"],
        n_windows=forecast_cfg["n_windows"],
        step_size=forecast_cfg["step_size"],
        **cv_kwargs,
    )
    cv_df = cv_df.copy()
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

    # One plot per cutoff, all ids subplots
    for cutoff in sorted(cv_df["cutoff"].unique()):
        w = cv_df[cv_df["cutoff"] == cutoff].copy().sort_values(["unique_id", "ds"])
        cutoff_str = pd.to_datetime(cutoff).strftime("%Y%m%d")

        w_plot = w[["unique_id", "ds", "y"] + model_names]

        plot_cutoff_subplots(
            df_window=w_plot,
            pred_cols=model_names,
            title=f"{img_prefix} | cutoff {cutoff_str}",
            out_path=img_dir / f"{img_prefix}_cutoff_{cutoff_str}.png",
            ncols=ncols,
            max_ids=max_ids,
        )

    return cv_df


def mean_rmse(cv_df: pd.DataFrame, model_names: list[str]) -> dict:
    return dict(
        rmse(cv_df, models=model_names, target_col="y")
        .groupby(["cutoff", "unique_id"])
        .mean()
        .reset_index(drop=True)
        .mean()
        .items()
    )



def run_cv_reconcile_save_each_window(
    cv_df: pd.DataFrame,
    ml_data: pd.DataFrame,
    best_model_name: str,
    H,
    S_df: pd.DataFrame = None,
    tags: dict = None,
    IDS: list[str] = None,
    out_dir: Path = None,
    img_dir: Path | None = None,
    save_plots: bool = True,
    save_diagnostics: bool = True,
    print_diagnostics_head: bool = False,
    ncols: int = 3,
    max_ids: int | None = None,
    config: dict | None = None,
    fe: object | None = None,
    save_evaluation: bool = True,
    debug: bool = False,
):
    out_dir.mkdir(exist_ok=True, parents=True)
    if img_dir is not None:
        img_dir.mkdir(exist_ok=True, parents=True)

    # -------------------------
    # Setup / clean
    # -------------------------
    if config is None:
        raise ValueError("config is required")
    if fe is None:
        raise ValueError("fe (FeatureEngineer) is required")
    
    # Extract information from H object
    # If S_df/tags not provided, they'll be recomputed in the CV loop for each window
    if IDS is None:
        if S_df is None:
            # If S_df not provided, try to get it from H
            try:
                hierarchy_data = H.get_hierarchy_data()
                IDS = hierarchy_data["bottom_ids"]
            except:
                IDS = H.bottom_ids

    reconcilers = H.build_reconcilers()
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)

    hier_cols = flatten_spec(H.spec)

    ml_data = ml_data.copy()
    ml_data["ds"] = pd.to_datetime(ml_data["ds"])
    ml_data = (
        ml_data.sort_values(["unique_id", "ds"])
        .drop_duplicates(subset=["unique_id", "ds"], keep="last")
    )

    cv_df = cv_df.copy()
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

    all_rec = []
    metrics_all = []

    seasonality = (
        config.get("evaluation", {}).get("seasonality", 7)  # daily default
        if isinstance(config, dict) else 7
    )

    # -------------------------
    # Loop cutoffs
    # -------------------------
    for cutoff in sorted(cv_df["cutoff"].unique()):
        cutoff = pd.to_datetime(cutoff)
        cutoff_str = cutoff.strftime("%Y%m%d")

        cv_w = cv_df[cv_df["cutoff"] == cutoff].copy().sort_values(["unique_id", "ds"])

        # horizon dates (from CV output)
        horizon_ds = sorted(cv_w["ds"].unique())

        # ---- 1) Build TRAIN raw (bottom only), attach hierarchy cols, then aggregate ALL levels
        train_raw_bottom = ml_data[
            ml_data["unique_id"].isin(IDS) & (ml_data["ds"] <= cutoff)
        ][["unique_id", "ds", "y"] + hier_cols].copy()

        train_raw_bottom = ensure_hier_cols(train_raw_bottom, hier_cols, H)

        # IMPORTANT: rename the original unique_id to something else because aggregate will create a new unique_id
        # from the hierarchy columns by concatenating them
        train_raw_bottom = train_raw_bottom.rename(columns={"unique_id": "cost_centre_id"})
        
        # Keep mapping of original cost_centre_id to hierarchy levels for later retrieval
        id_map = train_raw_bottom[["cost_centre_id"] + hier_cols].drop_duplicates().reset_index(drop=True)

        # aggregate to all levels (this creates Y_train_all + S_df + tags consistently)
        # aggregate will create a new unique_id by concatenating the hierarchy columns
        Y_train_all, S_df, tags = aggregate(df=train_raw_bottom, spec=H.spec)
        
        if debug:
            print(f"[DEBUG] S_df columns: {S_df.columns.tolist()}")
            print(f"[DEBUG] S_df shape: {S_df.shape}")
            print(f"[DEBUG] S_df head:\n{S_df.head()}")
        
        # Map tags keys to level names for the reconcilers
        # aggregate returns tags with spec-based keys like: 'Root', 'Root/Manager', 'Root/Manager/CostCentre'
        # We need to map these to standard level names: 'level_0', 'level_1', 'level_2'
        tags_mapped = {}
        spec_keys = sorted(tags.keys(), key=lambda x: len(x.split('/')))  # Sort by depth
        
        for i, spec_key in enumerate(spec_keys):
            level_name = f'level_{i}'
            tags_mapped[level_name] = tags[spec_key]
        
        tags = tags_mapped

        # ---- 2) Fit model on BOTTOM LEVEL ONLY (base forecasts, not aggregated)
        # After aggregation, unique_ids are hierarchical (e.g., "Kylie Van Der Stok/Amanda Judd/Amanda Judd")
        # We need to identify which aggregated IDs correspond to leaf nodes (original cost centres)
        # Build mapping from original cost_centre_id to aggregated unique_id
        cost_centre_to_agg_id = {}
        for _, row in id_map.iterrows():
            cc_id = row["cost_centre_id"]
            if cc_id in IDS:  # Only process leaf nodes
                hier_path = "/".join([str(row[col]) for col in hier_cols])
                cost_centre_to_agg_id[cc_id] = hier_path
        
        # Get the aggregated IDs corresponding to bottom level
        bottom_agg_ids = list(cost_centre_to_agg_id.values())
        Y_train_bottom = Y_train_all[Y_train_all["unique_id"].isin(bottom_agg_ids)].copy()
        
        if debug:
            print(f"[DEBUG] Y_train_all shape: {Y_train_all.shape}")
            print(f"[DEBUG] Cost centre to agg ID mapping: {len(cost_centre_to_agg_id)}")
            print(f"[DEBUG] Y_train_bottom shape (bottom level only): {Y_train_bottom.shape}")
            print(f"[DEBUG] Bottom level unique_ids: {Y_train_bottom['unique_id'].nunique()}")
        
        ml_forecast = MLForecast(
            models=[ModelFactory.get_model(best_model_name)],
            freq="D",
            lags=config["features"]["lags"],
            lag_transforms=fe.get_lag_transforms(),
            date_features=config["features"]["date_features"],
            num_threads=config["forecast"]["num_threads"],
        )

        # static features: only pass if present in Y_train_bottom
        static_feats = config["features"].get("static_features", [])
        static_feats = [c for c in static_feats if c in Y_train_bottom.columns]
        static_feats = static_feats if len(static_feats) > 0 else None

        ml_forecast.fit(
            Y_train_bottom,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            static_features=static_feats,
        )

        Y_hat_df = ml_forecast.predict(
            h=config["forecast"]["horizon"],
        ).rename(columns={"y": best_model_name})
        
        if debug:
            print(f"[DEBUG] Y_hat_df shape (bottom level only): {Y_hat_df.shape}")
            print(f"[DEBUG] Y_hat_df unique_ids: {Y_hat_df['unique_id'].nunique()}")

        # ---- Aggregate forecasts to all levels (for reconciliation input)
        # Y_hat_df currently has only bottom-level forecasts
        # We need to aggregate them to create forecasts for upper hierarchy levels
        # This mimics what would happen if we had trained on aggregated data
        
        # Convert Y_hat_df from aggregated unique_ids back to original cost_centre_ids
        Y_hat_bottom_original = Y_hat_df.copy()
        Y_hat_bottom_original["cost_centre_id"] = Y_hat_bottom_original["unique_id"].map(
            {v: k for k, v in cost_centre_to_agg_id.items()}
        )
        Y_hat_bottom_original = Y_hat_bottom_original.merge(id_map, on="cost_centre_id", how="left")
        
        # Now aggregate to all levels
        Y_hat_all_df, _, _ = aggregate(
            df=Y_hat_bottom_original[["cost_centre_id", "ds"] + hier_cols + [best_model_name]].rename(
                columns={best_model_name: "y"}
            ),
            spec=H.spec
        )
        Y_hat_all_df = Y_hat_all_df.rename(columns={"y": best_model_name})
        
        if debug:
            print(f"[DEBUG] Y_hat_all_df shape (all levels): {Y_hat_all_df.shape}")
            print(f"[DEBUG] Y_hat_all_df unique_ids: {Y_hat_all_df['unique_id'].nunique()}")

        # ---- Debug: coverage vs S_df
        if debug:
            nodes_S = set(S_df["unique_id"].unique())
            nodes_hat = set(Y_hat_all_df["unique_id"].unique())
            missing = sorted(list(nodes_S - nodes_hat))[:10]
            extra = sorted(list(nodes_hat - nodes_S))[:10]
            print(f"\n[DEBUG] cutoff {cutoff_str}")
            print(f"[DEBUG] nodes in S_df: {len(nodes_S)} | nodes in Y_hat_all_df: {len(nodes_hat)}")
            if missing:
                print(f"[DEBUG] missing in forecasts (first 10): {missing}")
            if extra:
                print(f"[DEBUG] extra in forecasts (first 10): {extra}")
            print(f"[DEBUG] Y_hat_df sample (first 5):")
            print(Y_hat_df.head())
            print(f"[DEBUG] Y_hat_df unique_ids count by level:")
            for uid in sorted(nodes_hat)[:3]:
                count = (Y_hat_df["unique_id"] == uid).sum()
                sample_val = Y_hat_df[Y_hat_df["unique_id"] == uid][best_model_name].iloc[0] if count > 0 else None
                print(f"  {uid}: {count} rows, sample value: {sample_val}")

        # ---- 3) Fitted values (needed for MinTrace/ERM). If not available, drop those reconcilers.
        Y_fitted_df = get_fitted_values_or_none(ml_forecast, best_model_name)
        has_fitted = Y_fitted_df is not None
        reconcilers_use = filter_reconcilers_if_no_fitted(reconcilers, has_fitted)
        hrec = HierarchicalReconciliation(reconcilers=reconcilers_use)

        if debug and not has_fitted:
            print("[DEBUG] MLForecast has no predict_in_sample(); dropping MinTrace/ERM for this run.")

        # ---- 4) Reconcile (ALL levels) + diagnostics
        # Use Y_hat_all_df (aggregated forecasts from bottom level) for reconciliation
        Y_rec_all = hrec.reconcile(
            Y_hat_df=Y_hat_all_df,
            Y_df=(Y_fitted_df if has_fitted else Y_train_all),
            S_df=S_df,
            tags=tags,
            diagnostics=True,
        )
        Y_rec_all["cutoff"] = cutoff
        
        # Reconstruct hierarchy columns from unique_id (which is concatenated with "/" separator)
        # Example: "Kylie Van Der Stok/Amanda Judd/Amanda Judd" -> [level_0, level_1, level_2]
        for i, col in enumerate(hier_cols):
            Y_rec_all[col] = Y_rec_all["unique_id"].str.split("/", expand=True)[i]
        
        all_rec.append(Y_rec_all)
        if debug:
            print(f"[DEBUG] Y_rec_all columns after reconstruction: {Y_rec_all.columns.tolist()}")
            print(f"[DEBUG] Y_rec_all shape: {Y_rec_all.shape}")
            print(f"[DEBUG] Y_rec_all sample (first 3):")
            print(Y_rec_all.head(3))
            # Check if bottom level values are still aggregated
            bottom_sample = Y_rec_all[Y_rec_all["unique_id"] == "Kylie Van Der Stok/Amanda Judd/Amanda Judd"][best_model_name].iloc[0] if any(Y_rec_all["unique_id"] == "Kylie Van Der Stok/Amanda Judd/Amanda Judd") else None
            print(f"[DEBUG] Sample bottom-level reconciled value: {bottom_sample}")
            debug_identity(Y_rec_all, best_model_name)
        Y_rec_all = shorten_reconciler_names(Y_rec_all, best_model_name)

        # diagnostics CSV
        if save_diagnostics:
            diag = hrec.diagnostics.copy()
            diag["cutoff"] = cutoff
            diag.to_csv(out_dir / f"diagnostics_cutoff_{cutoff_str}.csv", index=False)
            if print_diagnostics_head:
                print(f"\n=== Diagnostics head | cutoff {cutoff_str} ===")
                print(diag.head(30))

        # reconciled columns
        rec_cols = [c for c in Y_rec_all.columns if c.startswith(f"{best_model_name}/")]

        # ---- 5) Bottom-level comparison CSV (actual y + base + reconciled)
        # We need to extract only the LEAF nodes (actual cost centres) from Y_rec_all
        # These correspond to the original IDS that we trained on
        # Create a mapping from aggregated unique_id back to original cost_centre_ids
        
        # The aggregated unique_id is formed by concatenating hierarchy columns with "/"
        # So we need to find which aggregated unique_ids have a leaf node structure
        # A leaf node is one that matches a single cost_centre_id from the original data
        
        # Build a lookup: for each cost_centre_id, what's the full hierarchical unique_id?
        leaf_nodes_map = {}
        for _, row in id_map.iterrows():
            cc_id = row["cost_centre_id"]
            if cc_id in IDS:  # Only process leaf nodes (bottom level IDs)
                # Construct the aggregated unique_id by joining hierarchy columns
                hier_path = "/".join([str(row[col]) for col in hier_cols])
                leaf_nodes_map[cc_id] = hier_path
        
        # Filter reconciled results to only leaf nodes
        leaf_unique_ids = list(leaf_nodes_map.values())
        rec_bottom = Y_rec_all[Y_rec_all["unique_id"].isin(leaf_unique_ids)].copy()
        
        # Restore original cost_centre_id for merging with cv_w
        rec_bottom_list = []
        for cc_id, agg_id in leaf_nodes_map.items():
            temp = rec_bottom[rec_bottom["unique_id"] == agg_id].copy()
            temp["original_cost_centre_id"] = cc_id
            rec_bottom_list.append(temp)
        
        if rec_bottom_list:
            rec_bottom = pd.concat(rec_bottom_list, ignore_index=True)
        else:
            rec_bottom = pd.DataFrame()
        
        if debug:
            print(f"[DEBUG] Leaf nodes mapping created: {len(leaf_nodes_map)} cost centres")
            print(f"[DEBUG] rec_bottom shape: {rec_bottom.shape}")
            print(f"[DEBUG] rec_bottom unique aggregated IDs: {rec_bottom['unique_id'].nunique() if not rec_bottom.empty else 0}")
            print(f"[DEBUG] rec_cols found: {rec_cols}")
            if rec_cols and not rec_bottom.empty:
                print(f"[DEBUG] rec_bottom null counts for reconciliation:\n{rec_bottom[rec_cols + [best_model_name]].isnull().sum()}")
        
        # Select only the needed columns (use original_cost_centre_id as unique_id for merging)
        cols_to_keep = ["original_cost_centre_id", "ds", best_model_name] + rec_cols
        cols_to_keep = [c for c in cols_to_keep if c in rec_bottom.columns]
        if not rec_bottom.empty:
            rec_bottom = rec_bottom[cols_to_keep].rename(columns={"original_cost_centre_id": "unique_id"})
        
        
        # Merge with actual CV values
        comp_bottom = cv_w[["unique_id", "ds", "y", best_model_name]].merge(
            rec_bottom, on=["unique_id", "ds"], how="left", suffixes=("_actual", "")
        )
        comp_bottom["cutoff"] = cutoff

        comp_bottom.to_csv(
            out_dir / f"cv_bottom_comp_{best_model_name}_cutoff_{cutoff_str}.csv",
            index=False
        )
        Y_rec_all.to_csv(
            out_dir / f"cv_reconciled_all_levels_{best_model_name}_cutoff_{cutoff_str}.csv",
            index=False
        )

        # ---- 6) Evaluate like Nixtla: build TEST (ALL levels) by aggregating CV actuals
        if save_evaluation:
            test_raw_bottom = cv_w[["unique_id", "ds", "y"]].copy()
            
            # Merge hierarchy columns from ml_data
            test_raw_bottom = test_raw_bottom.merge(
                ml_data[["unique_id"] + hier_cols].drop_duplicates(),
                on="unique_id",
                how="left"
            )
            
            test_raw_bottom = ensure_hier_cols(test_raw_bottom, hier_cols, H)
            # Rename unique_id for aggregate, just like we did for train_raw_bottom
            test_raw_bottom = test_raw_bottom.rename(columns={"unique_id": "cost_centre_id"})
            Y_test_all, _, _ = aggregate(df=test_raw_bottom, spec=H.spec)
            
            # Reconstruct hierarchy columns for Y_test_all as well
            for i, col in enumerate(hier_cols):
                Y_test_all[col] = Y_test_all["unique_id"].str.split("/", expand=True)[i]

            # merge actuals into reconciled forecasts
            # Only merge on unique_id and ds, keep hierarchy cols from Y_rec_all
            Y_eval = Y_rec_all.merge(
                Y_test_all[["unique_id", "ds", "y"]], 
                on=["unique_id", "ds"], 
                how="left"
            )
            
            # Ensure all forecast columns are numeric
            forecast_cols = [c for c in Y_eval.columns if c.startswith(best_model_name)]
            for col in forecast_cols:
                Y_eval[col] = pd.to_numeric(Y_eval[col], errors='coerce')
            Y_eval["y"] = pd.to_numeric(Y_eval["y"], errors='coerce')

            eval_tags = build_eval_tags(tags)
            
            # For evaluation, we need to keep only numeric columns and valid IDs
            # Only select columns that won't cause issues with evaluate
            Y_eval_numeric = Y_eval[["unique_id", "ds", "y"] + forecast_cols + hier_cols].copy()
            
            try:
                # evaluate()
                evaluation = evaluate(
                    Y_eval_numeric,
                    metrics=[rmse],
                    tags=eval_tags,
                    train_df=Y_train_all,
                )
                evaluation["cutoff"] = cutoff
                metrics_all.append(evaluation)

                evaluation.to_csv(out_dir / f"evaluation_cutoff_{cutoff_str}.csv", index=False)
                
                print(f"\n=== Evaluation for cutoff {cutoff_str} ===")
                print(evaluation)
            except Exception as e:
                print(f"[WARNING] Evaluation failed for cutoff {cutoff_str}: {e}")
                if debug:
                    print(f"[DEBUG] Y_eval_numeric columns: {Y_eval_numeric.columns.tolist()}")
                    print(f"[DEBUG] Y_eval_numeric dtypes:\n{Y_eval_numeric.dtypes}")

        # ---- 7) Plots (unchanged)
        print(comp_bottom.head())
        if save_plots and img_dir is not None:
            plot_cutoff_subplots(
                df_window=comp_bottom,
                pred_cols=[best_model_name] + rec_cols,
                title=f"Reconciled CV | {best_model_name} | cutoff {cutoff_str}",
                out_path=img_dir / f"cv_reconciled_{best_model_name}_cutoff_{cutoff_str}.png",
                ncols=ncols,
                max_ids=max_ids,
            )

    rec_df = pd.concat(all_rec, ignore_index=True)

    metrics_df = pd.concat(metrics_all, ignore_index=True) if len(metrics_all) else pd.DataFrame()

    # also save a combined evaluation file
    if save_evaluation and len(metrics_df):
        metrics_df.to_csv(out_dir / f"evaluation_all_cutoffs_{best_model_name}.csv", index=False)
        print(f"\n=== Evaluation Results (all cutoffs) ===")
        print(metrics_df)

    return rec_df, metrics_df
