from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = RESEARCH_DIR / "results"


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def convert_to_datetime(date_str: str) -> pd.Timestamp:
    german_months = {
        "Jan": "Jan",
        "Feb": "Feb",
        "Mär": "Mar",
        "Mrz": "Mar",
        "Apr": "Apr",
        "Mai": "May",
        "Jun": "Jun",
        "Jul": "Jul",
        "Aug": "Aug",
        "Sep": "Sep",
        "Okt": "Oct",
        "Nov": "Nov",
        "Dez": "Dec",
    }
    portuguese_months = {
        "jan": "Jan",
        "fev": "Feb",
        "mar": "Mar",
        "abr": "Apr",
        "mai": "May",
        "jun": "Jun",
        "jul": "Jul",
        "ago": "Aug",
        "set": "Sep",
        "out": "Oct",
        "nov": "Nov",
        "dez": "Dec",
    }

    normalized = str(date_str)
    for short in ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"):
        normalized = normalized.replace(f"{short}.", short)
    for de_month, en_month in german_months.items():
        normalized = normalized.replace(de_month, en_month)
    lowered = normalized.lower()
    for pt_month, en_month in portuguese_months.items():
        if pt_month in lowered:
            lowered = lowered.replace(pt_month, en_month)
            normalized = lowered
            break

    month_abbrs = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    lower_norm = normalized.lower()
    for month in month_abbrs:
        if month in lower_norm:
            normalized = lower_norm.replace(month, month.capitalize(), 1)
            break

    formats = [
        "%b %d %Y %H:%M:%S",
        "%m %d %Y %H:%M:%S",
        "%m月 %d %Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y",
        "%m-%d-%y",
    ]
    for fmt in formats:
        try:
            return pd.Timestamp(datetime.strptime(normalized, fmt))
        except ValueError:
            continue
    return pd.NaT


def load_supermemo_user(csv_path: Path) -> pd.DataFrame:
    df = read_csv_with_fallback(csv_path)
    df.columns = df.columns.str.strip()
    df["source_order"] = np.arange(len(df))
    df["Date"] = df["Date"].map(convert_to_datetime)
    df = df.dropna(subset=["Date"]).copy()
    df = df[df["Success"].isin([0, 1]) & df["Grade"].isin([0, 1, 2, 3, 4, 5])].copy()
    df = df[(df["R (SM16)"] <= 1) & (df["R (SM17)"] <= 1)].copy()

    dataset = df[
        ["Date", "Element No", "Used interval", "Success", "Grade", "source_order"]
    ].rename(
        columns={
            "Date": "review_date",
            "Element No": "card_id",
            "Used interval": "delta_t",
            "Success": "y",
        }
    )
    dataset = dataset[dataset["delta_t"] > 0].copy()
    dataset = dataset.sort_values(["card_id", "review_date", "source_order"], kind="mergesort")
    dataset["i"] = dataset.groupby("card_id").cumcount() + 1
    dataset["day"] = dataset["review_date"].dt.floor("D")
    # Match the benchmark-style evaluation subset by dropping the first record per card.
    dataset = dataset[dataset["i"] > 1].copy()
    # Preserve a deterministic within-timestamp order close to file order.
    dataset = dataset.sort_values(
        ["review_date", "i", "source_order", "card_id"], kind="mergesort"
    ).reset_index(drop=True)
    dataset["first_of_day_card"] = (
        dataset.groupby(["day", "card_id"]).cumcount().eq(0)
    )
    return dataset


def make_pooled_agg() -> dict[str, float]:
    return {
        "ps_num": 0.0,
        "ps_den": 0.0,
        "pf_num": 0.0,
        "pf_den": 0.0,
        "pair_count": 0.0,
        "user_count": 0.0,
        "user_day_count": 0.0,
    }


def make_mean_agg() -> dict[str, float]:
    return {
        "ps_prob_sum": 0.0,
        "ps_prob_cnt": 0.0,
        "pf_prob_sum": 0.0,
        "pf_prob_cnt": 0.0,
        "common_gap_sum": 0.0,
        "common_gap_cnt": 0.0,
        "unit_count": 0.0,
        "pair_count": 0.0,
    }


def merge_agg(dst: dict[str, float], src: dict[str, float]) -> None:
    for key in dst:
        dst[key] += src.get(key, 0.0)


def finalize_pooled(agg: dict[str, float]) -> dict[str, float | int | None]:
    ps = agg["ps_num"] / agg["ps_den"] if agg["ps_den"] else None
    pf = agg["pf_num"] / agg["pf_den"] if agg["pf_den"] else None
    gap = ps - pf if ps is not None and pf is not None else None
    return {
        "p_success_given_prev_success": round(ps, 6) if ps is not None else None,
        "p_success_given_prev_fail": round(pf, 6) if pf is not None else None,
        "gap": round(gap, 6) if gap is not None else None,
        "pair_count": int(agg["pair_count"]),
        "user_count": int(agg["user_count"]),
        "user_day_count": int(agg["user_day_count"]),
    }


def finalize_mean(agg: dict[str, float]) -> dict[str, float | int | None]:
    ps = agg["ps_prob_sum"] / agg["ps_prob_cnt"] if agg["ps_prob_cnt"] else None
    pf = agg["pf_prob_sum"] / agg["pf_prob_cnt"] if agg["pf_prob_cnt"] else None
    gap = ps - pf if ps is not None and pf is not None else None
    common_gap = agg["common_gap_sum"] / agg["common_gap_cnt"] if agg["common_gap_cnt"] else None
    return {
        "p_success_given_prev_success": round(ps, 6) if ps is not None else None,
        "p_success_given_prev_fail": round(pf, 6) if pf is not None else None,
        "gap": round(gap, 6) if gap is not None else None,
        "common_support_gap": round(common_gap, 6) if common_gap is not None else None,
        "defined_prev_success_units": int(agg["ps_prob_cnt"]),
        "defined_prev_fail_units": int(agg["pf_prob_cnt"]),
        "both_defined_units": int(agg["common_gap_cnt"]),
        "unit_count": int(agg["unit_count"]),
        "pair_count": int(agg["pair_count"]),
    }


def counts_from_pairs(prev: np.ndarray, nxt: np.ndarray) -> tuple[float, float, float, float]:
    prev_success = prev == 1
    prev_fail = prev == 0
    return (
        float(nxt[prev_success].sum()),
        float(prev_success.sum()),
        float(nxt[prev_fail].sum()),
        float(prev_fail.sum()),
    )


def mean_agg_from_unit_counts(
    ps_num: np.ndarray,
    ps_den: np.ndarray,
    pf_num: np.ndarray,
    pf_den: np.ndarray,
    pair_count: int,
) -> dict[str, float]:
    agg = make_mean_agg()
    agg["unit_count"] = float(len(ps_den))
    agg["pair_count"] = float(pair_count)

    ps_mask = ps_den > 0
    if np.any(ps_mask):
        agg["ps_prob_sum"] = float((ps_num[ps_mask] / ps_den[ps_mask]).sum())
        agg["ps_prob_cnt"] = float(ps_mask.sum())

    pf_mask = pf_den > 0
    if np.any(pf_mask):
        agg["pf_prob_sum"] = float((pf_num[pf_mask] / pf_den[pf_mask]).sum())
        agg["pf_prob_cnt"] = float(pf_mask.sum())

    both_mask = ps_mask & pf_mask
    if np.any(both_mask):
        common_gap = (ps_num[both_mask] / ps_den[both_mask]) - (pf_num[both_mask] / pf_den[both_mask])
        agg["common_gap_sum"] = float(common_gap.sum())
        agg["common_gap_cnt"] = float(both_mask.sum())
    return agg


def user_day_counts_from_pairs(
    prev: np.ndarray, nxt: np.ndarray, current_days: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(prev) == 0:
        zeros = np.zeros(0, dtype=float)
        return zeros, zeros, zeros, zeros

    _, inverse = np.unique(current_days, return_inverse=True)
    prev_success = (prev == 1).astype(float)
    prev_fail = (prev == 0).astype(float)
    next_success = nxt.astype(float)
    ps_den = np.bincount(inverse, weights=prev_success)
    pf_den = np.bincount(inverse, weights=prev_fail)
    ps_num = np.bincount(inverse, weights=prev_success * next_success)
    pf_num = np.bincount(inverse, weights=prev_fail * next_success)
    return ps_num, ps_den, pf_num, pf_den


def restrict_to_common_support_user_days(
    prev: np.ndarray, nxt: np.ndarray, current_days: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(prev) == 0:
        return prev, nxt, current_days
    day_ps_num, day_ps_den, day_pf_num, day_pf_den = user_day_counts_from_pairs(prev, nxt, current_days)
    keep_days = (day_ps_den > 0) & (day_pf_den > 0)
    if not np.any(keep_days):
        empty = np.zeros(0, dtype=prev.dtype)
        return empty, empty, np.zeros(0, dtype=current_days.dtype)
    _, inverse = np.unique(current_days, return_inverse=True)
    keep_pairs = keep_days[inverse]
    return prev[keep_pairs], nxt[keep_pairs], current_days[keep_pairs]


def sequence_stats_from_pairs(
    prev: np.ndarray, nxt: np.ndarray, current_days: np.ndarray
) -> dict[str, dict[str, float]]:
    pooled = make_pooled_agg()
    equal_user_mean = make_mean_agg()
    equal_user_day_mean = make_mean_agg()
    pooled_on_common_support_user_days = make_pooled_agg()
    equal_user_mean_on_common_support_user_days = make_mean_agg()

    if len(prev) == 0:
        return {
            "pooled": pooled,
            "equal_user_mean": equal_user_mean,
            "equal_user_day_mean": equal_user_day_mean,
            "pooled_on_common_support_user_days": pooled_on_common_support_user_days,
            "equal_user_mean_on_common_support_user_days": equal_user_mean_on_common_support_user_days,
        }

    ps_num, ps_den, pf_num, pf_den = counts_from_pairs(prev, nxt)
    pooled.update(
        {
            "ps_num": ps_num,
            "ps_den": ps_den,
            "pf_num": pf_num,
            "pf_den": pf_den,
            "pair_count": float(len(prev)),
            "user_count": 1.0,
            "user_day_count": float(len(np.unique(current_days))),
        }
    )
    equal_user_mean = mean_agg_from_unit_counts(
        np.array([ps_num]), np.array([ps_den]), np.array([pf_num]), np.array([pf_den]), pair_count=len(prev)
    )
    day_ps_num, day_ps_den, day_pf_num, day_pf_den = user_day_counts_from_pairs(prev, nxt, current_days)
    equal_user_day_mean = mean_agg_from_unit_counts(
        day_ps_num, day_ps_den, day_pf_num, day_pf_den, pair_count=len(prev)
    )

    prev_cs, nxt_cs, day_cs = restrict_to_common_support_user_days(prev, nxt, current_days)
    if len(prev_cs) > 0:
        cs_ps_num, cs_ps_den, cs_pf_num, cs_pf_den = counts_from_pairs(prev_cs, nxt_cs)
        pooled_on_common_support_user_days.update(
            {
                "ps_num": cs_ps_num,
                "ps_den": cs_ps_den,
                "pf_num": cs_pf_num,
                "pf_den": cs_pf_den,
                "pair_count": float(len(prev_cs)),
                "user_count": 1.0,
                "user_day_count": float(len(np.unique(day_cs))),
            }
        )
        equal_user_mean_on_common_support_user_days = mean_agg_from_unit_counts(
            np.array([cs_ps_num]),
            np.array([cs_ps_den]),
            np.array([cs_pf_num]),
            np.array([cs_pf_den]),
            pair_count=len(prev_cs),
        )

    return {
        "pooled": pooled,
        "equal_user_mean": equal_user_mean,
        "equal_user_day_mean": equal_user_day_mean,
        "pooled_on_common_support_user_days": pooled_on_common_support_user_days,
        "equal_user_mean_on_common_support_user_days": equal_user_mean_on_common_support_user_days,
    }


def analyze_raw_long_term(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    y = df["y"].to_numpy(dtype=np.int8)
    if len(y) < 2:
        return {
            "pooled": make_pooled_agg(),
            "equal_user_mean": make_mean_agg(),
            "equal_user_day_mean": make_mean_agg(),
            "pooled_on_common_support_user_days": make_pooled_agg(),
            "equal_user_mean_on_common_support_user_days": make_mean_agg(),
        }
    prev = y[:-1]
    nxt = y[1:]
    current_days = df["day"].to_numpy()[1:]
    return sequence_stats_from_pairs(prev, nxt, current_days)


def analyze_same_day_first_per_card(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    first_df = df[df["first_of_day_card"]].copy()
    y = first_df["y"].to_numpy(dtype=np.int8)
    day = first_df["day"].to_numpy()
    if len(y) < 2:
        return {
            "pooled": make_pooled_agg(),
            "equal_user_mean": make_mean_agg(),
            "equal_user_day_mean": make_mean_agg(),
            "pooled_on_common_support_user_days": make_pooled_agg(),
            "equal_user_mean_on_common_support_user_days": make_mean_agg(),
        }
    same_day = day[1:] == day[:-1]
    prev = y[:-1][same_day]
    nxt = y[1:][same_day]
    current_days = day[1:][same_day]
    return sequence_stats_from_pairs(prev, nxt, current_days)


def analyze_user(csv_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    df = load_supermemo_user(csv_path)
    return {
        "raw_long_term": analyze_raw_long_term(df),
        "same_day_first_per_card": analyze_same_day_first_per_card(df),
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = {
        sequence: {
            "pooled": make_pooled_agg(),
            "equal_user_mean": make_mean_agg(),
            "equal_user_day_mean": make_mean_agg(),
            "pooled_on_common_support_user_days": make_pooled_agg(),
            "equal_user_mean_on_common_support_user_days": make_mean_agg(),
        }
        for sequence in ("raw_long_term", "same_day_first_per_card")
    }

    user_results = {}
    for csv_path in sorted(DATASET_DIR.glob("*.csv")):
        user = csv_path.stem.split("_", 1)[1]
        per_user = analyze_user(csv_path)
        user_results[user] = per_user
        for sequence, sequence_result in per_user.items():
            merge_agg(aggregate[sequence]["pooled"], sequence_result["pooled"])
            merge_agg(aggregate[sequence]["equal_user_mean"], sequence_result["equal_user_mean"])
            merge_agg(aggregate[sequence]["equal_user_day_mean"], sequence_result["equal_user_day_mean"])
            merge_agg(
                aggregate[sequence]["pooled_on_common_support_user_days"],
                sequence_result["pooled_on_common_support_user_days"],
            )
            merge_agg(
                aggregate[sequence]["equal_user_mean_on_common_support_user_days"],
                sequence_result["equal_user_mean_on_common_support_user_days"],
            )

    result = {
        "metadata": {
            "dataset": "fsrs-vs-sm17 SuperMemo collections",
            "users": len(user_results),
            "sequence_notes": {
                "raw_long_term": [
                    "load the SuperMemo CSV with the same row-level filters used by the benchmark preprocessing",
                    "drop the first record per card (i > 1) to match the benchmark evaluation subset",
                    "sort by parsed review_date, then i, then source order",
                    "form adjacent pairs in that full user sequence",
                ],
                "same_day_first_per_card": [
                    "start from the same benchmark subset (i > 1)",
                    "within each user-day-card keep the first parsed-order occurrence",
                    "form adjacent pairs only when both events are on the same day",
                    "no state == Review filter is available in the SuperMemo CSV, so this is only an analog of the Anki target",
                ],
            },
            "caveats": [
                "equal_user_day_mean weights every user-day equally after computing that user-day conditional probability",
                "equal_user_day_mean does not give every user equal total weight; users with more retained days contribute more user-day units",
                "common_support_gap averages the unit-level gap only over user-days where both conditionals are defined",
                "some collections have weak or absent within-day time resolution, so parsed-order adjacency should not be overinterpreted as verified wall-clock adjacency",
            ],
        },
        "aggregate": {
            sequence: {
                "pooled": finalize_pooled(levels["pooled"]),
                "equal_user_mean": finalize_mean(levels["equal_user_mean"]),
                "equal_user_day_mean": finalize_mean(levels["equal_user_day_mean"]),
                "pooled_on_common_support_user_days": finalize_pooled(levels["pooled_on_common_support_user_days"]),
                "equal_user_mean_on_common_support_user_days": finalize_mean(
                    levels["equal_user_mean_on_common_support_user_days"]
                ),
            }
            for sequence, levels in aggregate.items()
        },
        "per_user": user_results,
    }

    output_path = RESULT_DIR / "sm17_conditional_probability_levels.json"
    output_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {output_path}")
    for sequence, levels in result["aggregate"].items():
        print(f"\n## {sequence}")
        for level_name, stats in levels.items():
            print(level_name, stats)


if __name__ == "__main__":
    main()
