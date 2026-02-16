
# main.py
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

from functools import partial
from hierarchicalforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mase



def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)
def build_level_lookup(tags: dict) -> dict:
    """Map unique_id -> level name from H.tags."""
    uid_to_level = {}
    for level, uids in tags.items():
        for uid in uids:
            uid_to_level[uid] = level
    return uid_to_level


def rmse_grouped(df: pd.DataFrame, y_col: str, pred_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    """Compute RMSE for each pred col grouped by group_cols."""
    rows = []
    for col in pred_cols:
        tmp = df.dropna(subset=[y_col, col])
        if tmp.empty:
            continue
        g = tmp.groupby(group_cols).apply(lambda x: float(np.sqrt(np.mean((x[y_col] - x[col]) ** 2))))
        out = g.reset_index(name="rmse")
        out["pred_col"] = col
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=group_cols + ["rmse", "pred_col"])


def build_models(model_cfg: dict):
    return [ModelFactory.get_model(name, **params) for name, params in model_cfg.items()]


def build_leaf_node_map(parent_of: dict, leaves: list[str]) -> pd.DataFrame:
    """
    leaf -> all ancestors (including itself) mapping for aggregation.
    """
    rows = []
    for leaf in leaves:
        cur = leaf
        while cur != "":
            rows.append((leaf, cur))
            cur = parent_of.get(cur, "")
    return pd.DataFrame(rows, columns=["leaf", "node"]).drop_duplicates()


def aggregate_up(df_bottom: pd.DataFrame, leaf_node_map: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    df_bottom must have ['unique_id','ds'] + value_cols at bottom level only.
    Returns all levels aggregated by summation.
    """
    merged = df_bottom.merge(leaf_node_map, left_on="unique_id", right_on="leaf", how="inner")
    out = merged.groupby(["node", "ds"], as_index=False)[value_cols].sum()
    out = out.rename(columns={"node": "unique_id"})
    return out


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
    IDS: list[str],
    out_dir: Path,
    img_dir: Path | None = None,
    save_plots: bool = True,
    save_diagnostics: bool = True,
    print_diagnostics_head: bool = False,
    ncols: int = 3,
    max_ids: int | None = None,
    config: dict | None = None,
    fe: object | None = None,
    save_evaluation: bool = True,
    debug: bool = True,
):
    out_dir.mkdir(exist_ok=True, parents=True)
    if img_dir is not None:
        img_dir.mkdir(exist_ok=True, parents=True)

    # -------------------------
    # Helpers
    # -------------------------
    def flatten_spec(spec: list[list[str]]) -> list[str]:
        cols = []
        for lvl in spec:
            for c in lvl:
                if c not in cols:
                    cols.append(c)
        return cols

    def ensure_hier_cols(df: pd.DataFrame, hier_cols: list[str]) -> pd.DataFrame:
        """Ensure df has hierarchy columns; if not, try merging from H.paths_df."""
        if all(c in df.columns for c in hier_cols):
            return df

        if not hasattr(H, "paths_df"):
            raise ValueError(
                f"Missing hierarchy columns {hier_cols} in df, and H.paths_df not found.\n"
                f"Either add these columns to ml_data/cv_df, or set H.paths_df with mapping."
            )

        # Get unique_ids that need hierarchy mapping
        missing_ids = df["unique_id"].unique()
        
        # Create paths_df for these specific IDs (handles orphans)
        paths_df = H.create_paths_df_for_ids(missing_ids)

        need_cols = ["unique_id"] + hier_cols
        miss = [c for c in need_cols if c not in paths_df.columns]
        if miss:
            raise ValueError(f"H.paths_df exists but missing columns: {miss}")

        # Ensure we have unique mapping (drop duplicates)
        map_df = paths_df[need_cols].drop_duplicates("unique_id").reset_index(drop=True)
        
        # Check for any remaining duplicates
        if map_df.duplicated(subset=["unique_id"]).any():
            dups = map_df[map_df.duplicated(subset=["unique_id"], keep=False)]
            raise ValueError("Duplicate unique_ids found in hierarchy mapping")
        
        # Merge carefully to avoid duplicate columns
        out = df.merge(map_df, on="unique_id", how="left", validate="m:1")

        # sanity: any unmapped?
        if out[hier_cols].isna().any().any():
            bad = out.loc[out[hier_cols].isna().any(axis=1), "unique_id"].unique()[:10]
            raise ValueError(
                f"Some unique_id could not be mapped to hierarchy columns. Examples: {bad}\n"
                f"Fix your H.paths_df mapping for these ids."
            )
        return out

    def build_eval_tags(tags: dict) -> dict:
        """Auto-create evaluation groups similar to Nixtla example."""
        keys = list(tags.keys())

        def depth(k: str) -> int:
            if k.startswith("level_") and k.split("_")[-1].isdigit():
                return int(k.split("_")[-1])
            return k.count("/")  # 0 for total, bigger for deeper

        keys_sorted = sorted(keys, key=depth)
        total_key = keys_sorted[0]
        bottom_key = keys_sorted[-1]

        eval_tags = {"total": tags[total_key], "bottom": tags[bottom_key]}
        if len(keys_sorted) >= 2:
            eval_tags["mid"] = tags[keys_sorted[1]]
        return eval_tags

    def get_fitted_values_or_none(ml_forecast: MLForecast) -> pd.DataFrame | None:
        """
        Try to obtain in-sample fitted values for MinTrace/ERM.
        Works if your MLForecast version supports predict_in_sample().
        """
        if hasattr(ml_forecast, "predict_in_sample"):
            fitted = ml_forecast.predict_in_sample()
            # try to guess the prediction column
            pred_col = None
            for c in ["y_hat", "y_pred", "y", best_model_name]:
                if c in fitted.columns and c not in ["unique_id", "ds"]:
                    pred_col = c
                    break
            if pred_col is None:
                # fallback: last numeric column
                num_cols = [c for c in fitted.columns if c not in ["unique_id", "ds"]]
                pred_col = num_cols[-1]
            fitted = fitted.rename(columns={pred_col: best_model_name})
            return fitted[["unique_id", "ds", best_model_name]]
        return None

    def filter_reconcilers_if_no_fitted(reconcilers: list, has_fitted: bool) -> list:
        """If no fitted values, drop methods that typically need residual covariance."""
        if has_fitted:
            return reconcilers
        bad_names = {"MinTrace", "ERM"}
        kept = [r for r in reconcilers if r.__class__.__name__ not in bad_names]
        return kept

    def debug_identity(Y_rec_df: pd.DataFrame, base_col: str):
        """Warn if reconciliation columns equal base (common when only bottom was forecast)."""
        rec_cols = [c for c in Y_rec_df.columns if c.startswith(f"{base_col}/")]
        if not rec_cols:
            return
        base = Y_rec_df[base_col]
        for c in rec_cols:
            diff = (Y_rec_df[c] - base).abs().sum()
            if diff == 0:
                print(f"[DEBUG] {c} is IDENTICAL to base -> reconciliation did not change forecasts.")

    def shorten_reconciler_names(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
        """Shorten long reconciler column names for readability."""
        rename_map = {}
        for col in df.columns:
            if col.startswith(f"{base_col}/"):
                # Extract the reconciler method name
                parts = col.split("/")[1:]  # e.g., ['BottomUp'], ['TopDown_method-forecast_proportions'], etc.
                method = parts[0]
                
                # Create short name
                if method == "BottomUp":
                    short_name = f"{base_col}/BottomUp"
                elif method.startswith("TopDown"):
                    # TopDown_method-forecast_proportions -> TopDown_FP or TopDown_PA
                    if "forecast_proportions" in method:
                        short_name = f"{base_col}/TD_FP"
                    elif "proportion_averages" in method:
                        short_name = f"{base_col}/TD_PA"
                    else:
                        short_name = f"{base_col}/TD"
                elif method.startswith("MiddleOut"):
                    # MiddleOut_middle_level-level_1_top_down_method-forecast_proportions -> MO_FP or MO_PA
                    if "forecast_proportions" in method:
                        short_name = f"{base_col}/MO_FP"
                    elif "proportion_averages" in method:
                        short_name = f"{base_col}/MO_PA"
                    else:
                        short_name = f"{base_col}/MO"
                else:
                    short_name = col  # Keep original if not recognized
                
                # Handle duplicates by adding counter
                if short_name in rename_map.values():
                    counter = 2
                    while f"{short_name}_{counter}" in rename_map.values():
                        counter += 1
                    short_name = f"{short_name}_{counter}"
                
                rename_map[col] = short_name
        
        return df.rename(columns=rename_map)

    # -------------------------
    # Setup / clean
    # -------------------------
    if config is None:
        raise ValueError("config is required")
    if fe is None:
        raise ValueError("fe (FeatureEngineer) is required")

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
        ][["unique_id", "ds", "y"]].copy()

        train_raw_bottom = ensure_hier_cols(train_raw_bottom, hier_cols)

        # IMPORTANT: rename the original unique_id to something else because aggregate will create a new unique_id
        # from the hierarchy columns by concatenating them
        train_raw_bottom = train_raw_bottom.rename(columns={"unique_id": "cost_centre_id"})

        # aggregate to all levels (this creates Y_train_all + S_df + tags consistently)
        # aggregate will create a new unique_id by concatenating the hierarchy columns
        Y_train_all, S_df, tags = aggregate(df=train_raw_bottom, spec=H.spec)
        
        # Map tags keys to level names for the reconcilers
        # aggregate returns tags like {0: [...], '0/1': [...], '0/1/2': [...]}
        # We need to map these to {'level_0': [...], 'level_1': [...], 'level_2': [...]}
        level_names = sorted(H.tags.keys())  # e.g. ['level_0', 'level_1', 'level_2']
        tags_mapped = {}
        
        # The keys in tags from aggregate correspond to the spec levels
        # spec is [['level_0'], ['level_0', 'level_1'], ['level_0', 'level_1', 'level_2']]
        # So tags will have keys like: 0, '0/1', '0/1/2'
        spec_keys = list(tags.keys())
        
        # Map spec keys to level names
        for i, spec_key in enumerate(sorted(spec_keys, key=str)):
            if i < len(level_names):
                tags_mapped[level_names[i]] = tags[spec_key]
        
        tags = tags_mapped

        # ---- 2) Fit model on ALL levels (so reconciliation can actually change things)
        ml_forecast = MLForecast(
            models=[ModelFactory.get_model(best_model_name)],
            freq="D",
            lags=config["features"]["lags"],
            lag_transforms=fe.get_lag_transforms(),
            date_features=config["features"]["date_features"],
            num_threads=config["forecast"]["num_threads"],
        )

        # static features: only pass if present in Y_train_all (aggregate usually drops them)
        static_feats = config["features"].get("static_features", [])
        static_feats = [c for c in static_feats if c in Y_train_all.columns]
        static_feats = static_feats if len(static_feats) > 0 else None

        ml_forecast.fit(
            Y_train_all,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            static_features=static_feats,
        )

        Y_hat_df = ml_forecast.predict(
            h=config["forecast"]["horizon"],
        ).rename(columns={"y": best_model_name})

        # ---- Debug: coverage vs S_df
        if debug:
            nodes_S = set(S_df["unique_id"].unique())
            nodes_hat = set(Y_hat_df["unique_id"].unique())
            missing = sorted(list(nodes_S - nodes_hat))[:10]
            extra = sorted(list(nodes_hat - nodes_S))[:10]
            print(f"\n[DEBUG] cutoff {cutoff_str}")
            print(f"[DEBUG] nodes in S_df: {len(nodes_S)} | nodes in Y_hat_df: {len(nodes_hat)}")
            if missing:
                print(f"[DEBUG] missing in forecasts (first 10): {missing}")
            if extra:
                print(f"[DEBUG] extra in forecasts (first 10): {extra}")

        # ---- 3) Fitted values (needed for MinTrace/ERM). If not available, drop those reconcilers.
        Y_fitted_df = get_fitted_values_or_none(ml_forecast)
        has_fitted = Y_fitted_df is not None
        reconcilers_use = filter_reconcilers_if_no_fitted(reconcilers, has_fitted)
        hrec = HierarchicalReconciliation(reconcilers=reconcilers_use)

        if debug and not has_fitted:
            print("[DEBUG] MLForecast has no predict_in_sample(); dropping MinTrace/ERM for this run.")

        # ---- 4) Reconcile (ALL levels) + diagnostics
        Y_rec_all = hrec.reconcile(
            Y_hat_df=Y_hat_df,
            Y_df=(Y_fitted_df if has_fitted else Y_train_all),
            S_df=S_df,
            tags=tags,
            diagnostics=True,
        )
        Y_rec_all["cutoff"] = cutoff
        all_rec.append(Y_rec_all)

        if debug:
            debug_identity(Y_rec_all, best_model_name)
        
        # Shorten reconciler column names for readability
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
        rec_bottom = Y_rec_all.query("unique_id in @IDS")[["unique_id", "ds"] + rec_cols + [best_model_name]]
        comp_bottom = cv_w[["unique_id", "ds", "y", best_model_name]].merge(
            rec_bottom, on=["unique_id", "ds"], how="left", suffixes=("", "_all")
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
            test_raw_bottom = ensure_hier_cols(test_raw_bottom, hier_cols)
            # Rename unique_id for aggregate, just like we did for train_raw_bottom
            test_raw_bottom = test_raw_bottom.rename(columns={"unique_id": "cost_centre_id"})
            Y_test_all, _, _ = aggregate(df=test_raw_bottom, spec=H.spec)

            # merge actuals into reconciled forecasts
            Y_eval = Y_rec_all.merge(Y_test_all, on=["unique_id", "ds"], how="left", suffixes=("", "_true"))
            # after merge, actual column is "y" (from Y_test_all). If conflict, fix:
            if "y_true" in Y_eval.columns and "y" not in Y_eval.columns:
                Y_eval = Y_eval.rename(columns={"y_true": "y"})

            eval_tags = build_eval_tags(tags)

            # evaluate()
            evaluation = evaluate(
                Y_eval,
                metrics=[rmse],
                tags=eval_tags,
                train_df=Y_train_all,
            )
            evaluation["cutoff"] = cutoff
            metrics_all.append(evaluation)

            evaluation.to_csv(out_dir / f"evaluation_cutoff_{cutoff_str}.csv", index=False)
            
            print(f"\n=== Evaluation for cutoff {cutoff_str} ===")
            print(evaluation)

        # ---- 7) Plots (unchanged)
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
        print(f"\n=== Evaluation Summary ===")
        print(metrics_df)

    return rec_df, metrics_df
