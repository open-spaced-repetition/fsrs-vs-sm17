import pathlib
import json
import numpy as np
from statsmodels.stats.weightstats import ttest_ind


def cohen_d(group1, group2, size):
    # weighted mean
    mean1, mean2 = np.average(
        group1, weights=size), np.average(group2, weights=size)
    # weighted variance
    var1, var2 = np.average(
        (group1 - mean1)**2, weights=size), np.average((group2 - mean2)**2, weights=size)

    d = (mean1 - mean2) / np.sqrt((var1 + var2) / 2)

    return d


if __name__ == "__main__":
    FSRS = []
    SM17 = []
    SM16 = []
    sizes = []
    result_dir = pathlib.Path("./result")
    result_files = result_dir.glob("*.json")
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
            FSRS.append(result["FSRS"])
            SM17.append(result["SM17"])
            SM16.append(result["SM16"])
            sizes.append(result["size"])

    sizes = np.array(sizes)
    print(f"Total size: {sizes.sum()}")
    for metric in ("LogLoss", "RMSE",  "UniversalMetric"):
        print(f"metric: {metric}")

        FSRS_metrics = np.array([item[metric] for item in FSRS])
        SM17_metrics = np.array([item[metric] for item in SM17])
        SM16_metrics = np.array([item[metric] for item in SM16])

        print(f"FSRS mean: {np.average(FSRS_metrics, weights=sizes):.4f}, SM17 mean: {np.average(SM17_metrics, weights=sizes):.4f}, SM16 mean: {np.average(SM16_metrics, weights=sizes):.4f}")

        t_stat, p_value, df = ttest_ind(
            FSRS_metrics, SM17_metrics, weights=(sizes, sizes))

        print(f"t-statistic: {t_stat}, p-value: {p_value}, df: {df}")

        if p_value < 0.05:
            print(
                "The performance difference between FSRS and SM17 is statistically significant.")
        else:
            print(
                "The performance difference between FSRS and SM17 is not statistically significant.")

        print(f"Cohen's d: {cohen_d(FSRS_metrics, SM17_metrics, sizes)}")
