import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error


ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
RESULT_DIR = ROOT / "result"
REFEREE_ALGOS = [
    "FSRS-6",
    "FSRS-5",
    "FSRS-4.5",
    "FSRSv4",
    "FSRSv3",
    "SM16",
    "SM17",
    "AVG",
    "FSRS-6-default",
    "MOVING-AVG",
]


@dataclass
class UserStream:
    user: str
    frame: pd.DataFrame


def get_bin(x: np.ndarray, bins: int = 10) -> np.ndarray:
    return np.round(x * bins) / bins


def clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


def load_raw_streams() -> list[UserStream]:
    streams = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        user = path.stem.split("_", 1)[1]
        streams.append(UserStream(user=user, frame=pd.read_csv(path)))
    return streams


def load_result_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULT_DIR.glob("*.json")):
        result = json.loads(path.read_text())
        row = {"user": result["user"], "n": result["size"]}
        for model in [
            "FSRS-6",
            "FSRS-5",
            "FSRS-4.5",
            "FSRSv4",
            "FSRSv3",
            "SM16",
            "SM17",
            "AVG",
            "MOVING-AVG",
            "FSRS-6-default",
        ]:
            row[f"{model}_ll"] = result[model]["LogLoss"]
            row[f"{model}_auc"] = result[model]["AUC"]
            row[f"{model}_rmse"] = result[model]["RMSE(bins)"]

        for model in [
            "FSRS-6",
            "FSRS-5",
            "FSRS-4.5",
            "FSRSv4",
            "FSRSv3",
            "SM16",
            "SM17",
            "AVG",
            "MOVING-AVG",
            "FSRS-6-default",
        ]:
            um_scores = [
                value
                for key, value in result["Universal_Metrics"].items()
                if key.startswith(f"{model}_evaluated_by_")
            ]
            if um_scores:
                row[f"{model}_avg_um"] = float(np.mean(um_scores))

            ump_scores = [
                value
                for key, value in result["Universal_Metrics+"].items()
                if key.startswith(f"{model}_evaluated_by_")
            ]
            if ump_scores:
                row[f"{model}_max_ump"] = float(np.max(ump_scores))
        rows.append(row)
    return pd.DataFrame(rows)


def load_result_pairs() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULT_DIR.glob("*.json")):
        result = json.loads(path.read_text())
        user = result["user"]
        n = result["size"]
        for key, value in result["Universal_Metrics"].items():
            model, referee = key.split("_evaluated_by_")
            rows.append(
                {
                    "user": user,
                    "n": n,
                    "model": model,
                    "referee": referee,
                    "metric": "avg_um",
                    "value": value,
                }
            )
        for key, value in result["Universal_Metrics+"].items():
            model, referee = key.split("_evaluated_by_")
            rows.append(
                {
                    "user": user,
                    "n": n,
                    "model": model,
                    "referee": referee,
                    "metric": "ump",
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values, weights=weights))


def moving_average_preds(y: np.ndarray, x0: float = 1.2, w: float = 0.3) -> np.ndarray:
    x = x0
    preds = np.empty_like(y, dtype=float)
    for i, value in enumerate(y):
        p = 1 / (1 + np.exp(-x))
        preds[i] = p
        x += w * (value - p)
    return preds


def online_average_preds(
    y: np.ndarray, prior_success: float = 0.9, prior_count: float = 1.0
) -> np.ndarray:
    total_success = prior_success
    total_count = prior_count
    preds = np.empty_like(y, dtype=float)
    for i, value in enumerate(y):
        preds[i] = total_success / total_count
        total_success += value
        total_count += 1.0
    return preds


def trailing_window_preds(
    y: np.ndarray, window: int, prior_success: float = 0.9
) -> np.ndarray:
    preds = np.empty_like(y, dtype=float)
    history: list[float] = []
    for i, value in enumerate(y):
        preds[i] = prior_success if not history else float(np.mean(history[-window:]))
        history.append(value)
    return preds


def average_um_against_referees(frame: pd.DataFrame, pred: np.ndarray) -> tuple[float, float]:
    y = frame["y"].to_numpy(dtype=float)
    ums = []
    um_plus = []
    for referee in REFEREE_ALGOS:
        referee_p = frame[f"R ({referee})"].to_numpy(dtype=float)
        grouped = (
            pd.DataFrame({"bin": get_bin(referee_p), "y": y, "p": pred})
            .groupby("bin")
            .agg(y=("y", "mean"), p=("p", "mean"), count=("p", "count"))
        )
        ums.append(
            root_mean_squared_error(
                grouped["y"], grouped["p"], sample_weight=grouped["count"]
            )
        )

        grouped_plus = (
            pd.DataFrame(
                {
                    "bin": np.round((pred - referee_p) * 20) / 20,
                    "y": y,
                    "p": pred,
                }
            )
            .groupby("bin")
            .agg(y=("y", "mean"), p=("p", "mean"), count=("p", "count"))
        )
        um_plus.append(
            root_mean_squared_error(
                grouped_plus["y"],
                grouped_plus["p"],
                sample_weight=grouped_plus["count"],
            )
        )
    return float(np.mean(ums)), float(np.max(um_plus))


def pair_metrics_against_referees(frame: pd.DataFrame, pred: np.ndarray) -> list[dict]:
    y = frame["y"].to_numpy(dtype=float)
    rows = []
    for referee in REFEREE_ALGOS:
        referee_p = frame[f"R ({referee})"].to_numpy(dtype=float)
        grouped = (
            pd.DataFrame({"bin": get_bin(referee_p), "y": y, "p": pred})
            .groupby("bin")
            .agg(y=("y", "mean"), p=("p", "mean"), count=("p", "count"))
        )
        um = root_mean_squared_error(
            grouped["y"], grouped["p"], sample_weight=grouped["count"]
        )

        grouped_plus = (
            pd.DataFrame(
                {
                    "bin": np.round((pred - referee_p) * 20) / 20,
                    "y": y,
                    "p": pred,
                }
            )
            .groupby("bin")
            .agg(y=("y", "mean"), p=("p", "mean"), count=("p", "count"))
        )
        ump = root_mean_squared_error(
            grouped_plus["y"],
            grouped_plus["p"],
            sample_weight=grouped_plus["count"],
        )
        rows.append({"referee": referee, "avg_um": float(um), "ump": float(ump)})
    return rows


def summarize_streams(streams: list[UserStream]) -> pd.DataFrame:
    rows = []
    for stream in streams:
        frame = stream.frame
        y = frame["y"].to_numpy(dtype=float)
        roll_50 = pd.Series(y).rolling(50, min_periods=50).mean()
        roll_200 = pd.Series(y).rolling(200, min_periods=200).mean()
        rows.append(
            {
                "user": stream.user,
                "n": len(frame),
                "success_rate": float(y.mean()),
                "const_user_mean_ll": float(log_loss(y, np.full_like(y, y.mean()))),
                "roll_50_std": float(roll_50.std()),
                "roll_200_std": float(roll_200.std()),
            }
        )
    return pd.DataFrame(rows)


def baseline_metrics(streams: list[UserStream]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pair_rows = []
    overall_success = np.mean(
        np.concatenate([stream.frame["y"].to_numpy(dtype=float) for stream in streams])
    )
    for stream in streams:
        frame = stream.frame
        y = frame["y"].to_numpy(dtype=float)
        baselines = {
            "CONST-0.87": np.full(len(frame), overall_success),
            "ONLINE-AVG": online_average_preds(y),
            "WIN-50": trailing_window_preds(y, 50),
            "WIN-200": trailing_window_preds(y, 200),
        }
        for name, pred in baselines.items():
            pair_metrics = pair_metrics_against_referees(frame, pred)
            avg_um = float(np.mean([item["avg_um"] for item in pair_metrics]))
            max_ump = float(np.max([item["ump"] for item in pair_metrics]))
            rows.append(
                {
                    "user": stream.user,
                    "n": len(frame),
                    "model": name,
                    "ll": float(log_loss(y, clip_probs(pred))),
                    "auc": float(roc_auc_score(y, pred)),
                    "avg_um": avg_um,
                    "max_ump": max_ump,
                }
            )
            for item in pair_metrics:
                pair_rows.append(
                    {
                        "user": stream.user,
                        "n": len(frame),
                        "model": name,
                        "referee": item["referee"],
                        "metric": "avg_um",
                        "value": item["avg_um"],
                    }
                )
                pair_rows.append(
                    {
                        "user": stream.user,
                        "n": len(frame),
                        "model": name,
                        "referee": item["referee"],
                        "metric": "ump",
                        "value": item["ump"],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def aggregate_model_table(
    existing: pd.DataFrame, baselines: pd.DataFrame, metric_suffix: str
) -> pd.DataFrame:
    rows = []
    weights = existing["n"]
    for model in ["FSRS-6", "MOVING-AVG", "AVG", "SM17", "FSRS-6-default"]:
        metric = existing[f"{model}_{metric_suffix}"]
        rows.append(
            {
                "model": model,
                "weighted": weighted_mean(metric, weights),
                "unweighted": float(metric.mean()),
            }
        )
    if metric_suffix in {"ll", "auc"}:
        for model in ["CONST-0.87", "ONLINE-AVG", "WIN-50", "WIN-200"]:
            sub = baselines[baselines["model"] == model]
            rows.append(
                {
                    "model": model,
                    "weighted": weighted_mean(sub[metric_suffix], sub["n"]),
                    "unweighted": float(sub[metric_suffix].mean()),
                }
            )
    elif metric_suffix in {"avg_um", "max_ump"}:
        for model in ["CONST-0.87", "ONLINE-AVG", "WIN-50", "WIN-200"]:
            sub = baselines[baselines["model"] == model]
            rows.append(
                {
                    "model": model,
                    "weighted": weighted_mean(sub[metric_suffix], sub["n"]),
                    "unweighted": float(sub[metric_suffix].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("weighted")


def aggregate_pair_metric(
    result_pairs: pd.DataFrame,
    baseline_pairs: pd.DataFrame,
    metric: str,
    reducer: str,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            result_pairs[result_pairs["metric"] == metric],
            baseline_pairs[baseline_pairs["metric"] == metric],
        ],
        ignore_index=True,
    )
    rows = []
    for model, sub in combined.groupby("model"):
        pairwise = (
            sub.groupby("referee")
            .apply(lambda g: np.average(g["value"], weights=g["n"]))
            .reset_index(name="weighted_pair")
        )
        if reducer == "mean":
            weighted = float(pairwise["weighted_pair"].mean())
        elif reducer == "max":
            weighted = float(pairwise["weighted_pair"].max())
        else:
            raise ValueError(reducer)

        user_level = (
            sub.groupby(["user", "n"])
            .apply(
                lambda g: g["value"].max() if reducer == "max" else g["value"].mean()
            )
            .reset_index(name="per_user")
        )
        rows.append(
            {
                "model": model,
                "weighted": weighted,
                "unweighted": float(user_level["per_user"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted")


def prefix_logloss(streams: list[UserStream]) -> pd.DataFrame:
    segments = []
    for stream in streams:
        frame = stream.frame.reset_index(drop=True)
        n = len(frame)
        segment_defs = {
            "first_50": frame.index < min(50, n),
            "first_200": frame.index < min(200, n),
            "first_1000": frame.index < min(1000, n),
            "last_50pct": frame.index >= n // 2,
        }
        for segment, mask in segment_defs.items():
            subset = frame[mask]
            if len(subset) < 10:
                continue
            row = {"user": stream.user, "segment": segment, "n": len(subset)}
            for model in ["MOVING-AVG", "AVG", "SM17", "FSRS-6"]:
                row[model] = float(
                    log_loss(
                        subset["y"].to_numpy(dtype=float),
                        subset[f"R ({model})"].to_numpy(dtype=float),
                    )
                )
            segments.append(row)
    return pd.DataFrame(segments)


def tune_hybrid(streams: list[UserStream]) -> dict[str, float]:
    lambda_grid = np.linspace(0, 1, 21)
    best = None
    for lam in lambda_grid:
        losses = []
        weights = []
        for stream in streams:
            frame = stream.frame
            split = len(frame) // 2
            if split == 0:
                continue
            y = frame["y"].to_numpy(dtype=float)[:split]
            fsrs = frame["R (FSRS-6)"].to_numpy(dtype=float)[:split]
            moving = frame["R (MOVING-AVG)"].to_numpy(dtype=float)[:split]
            pred = expit(lam * logit(clip_probs(fsrs)) + (1 - lam) * logit(clip_probs(moving)))
            losses.append(float(log_loss(y, clip_probs(pred))))
            weights.append(len(y))
        score = float(np.average(losses, weights=weights))
        if best is None or score < best["train_score"]:
            best = {"lambda": float(lam), "train_score": score}

    eval_losses = {"MOVING-AVG": [], "FSRS-6": [], "HYBRID": []}
    eval_weights = []
    for stream in streams:
        frame = stream.frame
        split = len(frame) // 2
        y = frame["y"].to_numpy(dtype=float)[split:]
        fsrs = frame["R (FSRS-6)"].to_numpy(dtype=float)[split:]
        moving = frame["R (MOVING-AVG)"].to_numpy(dtype=float)[split:]
        hybrid = expit(
            best["lambda"] * logit(clip_probs(fsrs))
            + (1 - best["lambda"]) * logit(clip_probs(moving))
        )
        eval_losses["MOVING-AVG"].append(float(log_loss(y, clip_probs(moving))))
        eval_losses["FSRS-6"].append(float(log_loss(y, clip_probs(fsrs))))
        eval_losses["HYBRID"].append(float(log_loss(y, clip_probs(hybrid))))
        eval_weights.append(len(y))

    best["eval_weighted"] = {
        name: float(np.average(losses, weights=eval_weights))
        for name, losses in eval_losses.items()
    }
    best["eval_unweighted"] = {
        name: float(np.mean(losses)) for name, losses in eval_losses.items()
    }
    return best


def moving_avg_math_example() -> dict[str, float]:
    p = 0.87
    x = logit(p)
    success_x = x + 0.3 * (1 - p)
    failure_x = x - 0.3 * p
    return {
        "p": p,
        "success_delta_x": float(success_x - x),
        "failure_delta_x": float(failure_x - x),
        "success_delta_p": float(expit(success_x) - p),
        "failure_delta_p": float(expit(failure_x) - p),
    }


def lag_dependency_stats(streams: list[UserStream]) -> dict[str, float]:
    weighted_prev_success = []
    weighted_prev_fail = []
    weights = []
    acf_1 = []
    acf_10 = []
    for stream in streams:
        y = stream.frame["y"].to_numpy(dtype=float)
        if len(y) < 2:
            continue
        prev = y[:-1]
        nxt = y[1:]
        if np.any(prev == 1):
            weighted_prev_success.append(float(nxt[prev == 1].mean()))
        else:
            weighted_prev_success.append(np.nan)
        if np.any(prev == 0):
            weighted_prev_fail.append(float(nxt[prev == 0].mean()))
        else:
            weighted_prev_fail.append(np.nan)
        weights.append(len(y))
        acf_1.append(float(np.corrcoef(y[:-1], y[1:])[0, 1]))
        if len(y) > 10:
            acf_10.append(float(np.corrcoef(y[:-10], y[10:])[0, 1]))
        else:
            acf_10.append(np.nan)

    all_frames = []
    for stream in streams:
        frame = stream.frame[["y"]].copy()
        frame["user"] = stream.user
        all_frames.append(frame)
    pooled = pd.concat(all_frames, ignore_index=True)
    pooled["prev"] = pooled.groupby("user")["y"].shift(1)

    return {
        "weighted_prev_success": float(np.average(weighted_prev_success, weights=weights)),
        "weighted_prev_fail": float(np.average(weighted_prev_fail, weights=weights)),
        "weighted_acf_1": float(np.average(np.nan_to_num(acf_1), weights=weights)),
        "weighted_acf_10": float(np.average(np.nan_to_num(acf_10), weights=weights)),
        "pooled_prev_success": float(pooled.loc[pooled["prev"] == 1, "y"].mean()),
        "pooled_prev_fail": float(pooled.loc[pooled["prev"] == 0, "y"].mean()),
    }


def bootstrap_logloss_differences(
    existing: pd.DataFrame, baselines: pd.DataFrame, n_boot: int = 10000, seed: int = 0
) -> pd.DataFrame:
    joined = existing[["user", "n", "FSRS-6_ll", "MOVING-AVG_ll"]].merge(
        baselines.pivot(index="user", columns="model", values="ll").reset_index(),
        on="user",
    )
    rng = np.random.default_rng(seed)
    comparisons = [
        ("FSRS-6 - MOVING-AVG", "FSRS-6_ll", "MOVING-AVG_ll"),
        ("MOVING-AVG - ONLINE-AVG", "MOVING-AVG_ll", "ONLINE-AVG"),
        ("MOVING-AVG - WIN-200", "MOVING-AVG_ll", "WIN-200"),
    ]
    rows = []
    for label, left, right in comparisons:
        point = float(np.average(joined[left] - joined[right], weights=joined["n"]))
        samples = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(joined), len(joined))
            sample = joined.iloc[idx]
            samples.append(float(np.average(sample[left] - sample[right], weights=sample["n"])))
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "comparison": label,
                "point": point,
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def print_table(title: str, frame: pd.DataFrame, digits: int = 4) -> None:
    print(f"\n## {title}")
    rounded = frame.copy()
    for col in rounded.select_dtypes(include=["float64", "float32"]).columns:
        rounded[col] = rounded[col].map(lambda x: round(x, digits))
    print(rounded.to_string(index=False))


def main() -> None:
    streams = load_raw_streams()
    existing = load_result_rows()
    result_pairs = load_result_pairs()
    per_user = summarize_streams(streams)
    baselines, baseline_pairs = baseline_metrics(streams)
    lag_stats = lag_dependency_stats(streams)
    boot = bootstrap_logloss_differences(existing, baselines)

    total_reviews = int(per_user["n"].sum())
    overall_success = np.mean(
        np.concatenate([stream.frame["y"].to_numpy(dtype=float) for stream in streams])
    )
    print(f"Users: {len(streams)}")
    print(f"Reviews: {total_reviews}")
    print(f"Overall success rate: {overall_success:.4f}")
    print(
        "Per-user success rate:"
        f" mean={per_user['success_rate'].mean():.4f},"
        f" std={per_user['success_rate'].std():.4f},"
        f" min={per_user['success_rate'].min():.4f},"
        f" max={per_user['success_rate'].max():.4f}"
    )
    print(
        "Weighted per-user P(success | prev success)="
        f"{lag_stats['weighted_prev_success']:.4f}, "
        "P(success | prev fail)="
        f"{lag_stats['weighted_prev_fail']:.4f}"
    )
    print(
        "Pooled P(success | prev success)="
        f"{lag_stats['pooled_prev_success']:.4f}, "
        "P(success | prev fail)="
        f"{lag_stats['pooled_prev_fail']:.4f}"
    )
    print(
        "Weighted autocorrelation:"
        f" lag-1={lag_stats['weighted_acf_1']:.4f},"
        f" lag-10={lag_stats['weighted_acf_10']:.4f}"
    )

    print_table(
        "Log Loss Summary",
        aggregate_model_table(existing, baselines, "ll"),
    )
    print_table(
        "AUC Summary",
        aggregate_model_table(existing, baselines, "auc").sort_values(
            "weighted", ascending=False
        ),
    )
    print_table(
        "Average Universal Metric Summary",
        aggregate_pair_metric(result_pairs, baseline_pairs, "avg_um", "mean"),
    )
    print_table(
        "UM+ Max Summary",
        aggregate_pair_metric(result_pairs, baseline_pairs, "ump", "max"),
    )

    prefix = prefix_logloss(streams)
    prefix_rows = []
    for segment in ["first_50", "first_200", "first_1000", "last_50pct"]:
        subset = prefix[prefix["segment"] == segment]
        prefix_rows.append(
            {
                "segment": segment,
                "MOVING-AVG": weighted_mean(subset["MOVING-AVG"], subset["n"]),
                "AVG": weighted_mean(subset["AVG"], subset["n"]),
                "SM17": weighted_mean(subset["SM17"], subset["n"]),
                "FSRS-6": weighted_mean(subset["FSRS-6"], subset["n"]),
            }
        )
    print_table("Prefix Log Loss (Weighted)", pd.DataFrame(prefix_rows))

    user_compare = existing[["user", "n", "MOVING-AVG_ll", "FSRS-6_ll"]].copy()
    user_compare["delta_ll"] = user_compare["MOVING-AVG_ll"] - user_compare["FSRS-6_ll"]
    print_table(
        "Users Sorted By MOVING-AVG - FSRS-6 Log Loss",
        user_compare.sort_values("delta_ll"),
    )
    print_table("User-Bootstrap Weighted Log Loss Differences", boot)

    hybrid = tune_hybrid(streams)
    print("\n## Hybrid Logit Mixture")
    print(f"Best lambda on first half (FSRS-6 weight): {hybrid['lambda']:.2f}")
    print(f"Train score: {hybrid['train_score']:.4f}")
    for name, value in hybrid["eval_weighted"].items():
        print(f"Second-half weighted log loss {name}: {value:.4f}")
    for name, value in hybrid["eval_unweighted"].items():
        print(f"Second-half unweighted log loss {name}: {value:.4f}")

    math_example = moving_avg_math_example()
    print("\n## MOVING-AVG Local Update Example")
    print(
        "At p=0.87: "
        f"success delta_x={math_example['success_delta_x']:.4f}, "
        f"failure delta_x={math_example['failure_delta_x']:.4f}, "
        f"success delta_p={math_example['success_delta_p']:.4f}, "
        f"failure delta_p={math_example['failure_delta_p']:.4f}"
    )


if __name__ == "__main__":
    main()
