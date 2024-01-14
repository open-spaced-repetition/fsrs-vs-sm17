import pathlib
import json
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
import scipy.stats


def cohen_d(group1, group2, size):
    # weighted mean
    mean1, mean2 = np.average(group1, weights=size), np.average(group2, weights=size)
    # weighted variance
    var1, var2 = np.average((group1 - mean1) ** 2, weights=size), np.average(
        (group2 - mean2) ** 2, weights=size
    )

    d = (mean1 - mean2) / np.sqrt((var1 + var2) / 2)

    return d


def confidence_interval(values, sizes):
    identifiers = [i for i in range(len(values))]
    dict_x_w = {
        identifier: (value, weight)
        for identifier, (value, weight) in enumerate(zip(values, sizes))
    }

    def weighted_mean(z, axis):
        # creating an array of weights, by mapping z to dict_x_w
        data = np.vectorize(dict_x_w.get)(z)
        return np.average(data[0], weights=data[1], axis=axis)

    CI_99_bootstrap = scipy.stats.bootstrap(
        (identifiers,),
        statistic=weighted_mean,
        confidence_level=0.99,
        axis=0,
        method="BCa",
        n_resamples=300_000,

    )
    low = list(CI_99_bootstrap.confidence_interval)[0]
    high = list(CI_99_bootstrap.confidence_interval)[1]
    return (high - low) / 2


if __name__ == "__main__":
    FSRSv3 = []
    FSRSv4 = []
    SM17 = []
    SM16 = []
    sizes = []
    result_dir = pathlib.Path("./result")
    result_files = result_dir.glob("*.json")
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
            FSRSv3.append(result["FSRSv3"])
            FSRSv4.append(result["FSRS-4.5"])
            SM17.append(result["SM17"])
            SM16.append(result["SM16"])
            sizes.append(result["size"])

    print(f"Total number of users: {len(sizes)}")
    sizes = np.array(sizes)
    print(f"Total size: {sizes.sum()}")
    sizes = np.log(sizes)
    for metric in ("LogLoss", "RMSE", "RMSE(bins)"):
        print(f"metric: {metric}")
        FSRSv3_metrics = np.array([item[metric] for item in FSRSv3])
        print(f"FSRSv3 mean: {np.average(FSRSv3_metrics, weights=sizes):.4f}±{confidence_interval(FSRSv3_metrics, sizes):.4f}")
        FSRSv4_metrics = np.array([item[metric] for item in FSRSv4])
        print(f"FSRSv4 mean: {np.average(FSRSv4_metrics, weights=sizes):.4f}±{confidence_interval(FSRSv4_metrics, sizes):.4f}")
        SM17_metrics = np.array([item[metric] for item in SM17])
        print(f"SM17 mean: {np.average(SM17_metrics, weights=sizes):.4f}±{confidence_interval(SM17_metrics, sizes):.4f}")
        SM16_metrics = np.array([item[metric] for item in SM16])
        print(f"SM16 mean: {np.average(SM16_metrics, weights=sizes):.4f}±{confidence_interval(SM16_metrics, sizes):.4f}")

        t_stat, p_value, df = ttest_ind(
            FSRSv4_metrics, SM17_metrics, weights=(sizes, sizes)
        )

        print(f"t-statistic: {t_stat}, p-value: {p_value}, df: {df}")

        if p_value < 0.05:
            print(
                "The performance difference between FSRS-4.5 and SM17 is statistically significant."
            )
        else:
            print(
                "The performance difference between FSRS-4.5 and SM17 is not statistically significant."
            )

        print(f"Cohen's d: {cohen_d(FSRSv4_metrics, SM17_metrics, sizes)}")
