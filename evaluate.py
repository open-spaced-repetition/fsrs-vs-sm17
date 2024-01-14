import pathlib
import json
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
import scipy.stats
import math


def cohen_d(group1, group2, size):
    # weighted mean
    mean1, mean2 = np.average(group1, weights=size), np.average(group2, weights=size)
    # weighted variance
    var1, var2 = np.average((group1 - mean1) ** 2, weights=size), np.average(
        (group2 - mean2) ** 2, weights=size
    )

    d = (mean1 - mean2) / np.sqrt((var1 + var2) / 2)

    return d

def sigdig(value, CI):
    def num_lead_zeros(x):
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    def first_nonzero_digit(x):
        x = str(x)
        for digit in x:
            if digit == "0" or digit == ".":
                pass
            else:
                return int(digit)

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = min(len(str(CI)[2 + n_lead_zeros_CI:]), 2)  # assumes CI<1
    decimals = n_lead_zeros_CI + CI_sigdigs
    rounded_CI = round(CI, decimals)
    first_sigdig_CI = first_nonzero_digit(rounded_CI)
    if first_sigdig_CI<5:
        rounded_value = round(value, decimals - 1)
        return str(f'{rounded_value:.{decimals - 1}f}'), str(f'{rounded_CI:.{decimals}f}')
    else:
        rounded_value = round(value, max(decimals - 2, 0))
        rounded_CI = round(CI, max(decimals - 1, 1))
        return str(f'{rounded_value:.{max(decimals - 2, 0)}f}'), str(f'{rounded_CI:.{max(decimals - 1, 1)}f}')


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
    logsizes = np.log(sizes)
    for metric in ("LogLoss", "RMSE", "RMSE(bins)"):
        print(f"metric: {metric}")

        FSRS45_metrics = np.array([item[metric] for item in FSRSv4])
        wmean = np.average(FSRS45_metrics, weights=sizes)
        CI = confidence_interval(FSRS45_metrics, sizes)
        rounded_mean, rounded_CI = sigdig(wmean, CI)
        print(f"FSRS-4.5 mean: {rounded_mean}±{rounded_CI}")

        FSRSv3_metrics = np.array([item[metric] for item in FSRSv3])
        wmean = np.average(FSRSv3_metrics, weights=sizes)
        CI = confidence_interval(FSRSv3_metrics, sizes)
        rounded_mean, rounded_CI = sigdig(wmean, CI)
        print(f"FSRS v3 mean: {rounded_mean}±{rounded_CI}")

        SM17_metrics = np.array([item[metric] for item in SM17])
        wmean = np.average(SM17_metrics, weights=sizes)
        CI = confidence_interval(SM17_metrics, sizes)
        rounded_mean, rounded_CI = sigdig(wmean, CI)
        print(f"SM-17 mean: {rounded_mean}±{rounded_CI}")

        SM16_metrics = np.array([item[metric] for item in SM16])
        wmean = np.average(SM16_metrics, weights=sizes)
        CI = confidence_interval(SM16_metrics, sizes)
        rounded_mean, rounded_CI = sigdig(wmean, CI)
        print(f"SM-16 mean: {rounded_mean}±{rounded_CI}")

        print('')

        print(f"metric: {metric}, log(size)")

        FSRS45_metrics = np.array([item[metric] for item in FSRSv4])
        wmean = np.average(FSRS45_metrics, weights=logsizes)
        CI = confidence_interval(FSRS45_metrics, logsizes)
        rounded_mean, rounded_CI = sigdig(wmean, CI)
        print(f"FSRS-4.5 mean: {rounded_mean}±{rounded_CI}")

        FSRSv3_metrics = np.array([item[metric] for item in FSRSv3])
        wmean = np.average(FSRSv3_metrics, weights=logsizes)
        CI = confidence_interval(FSRSv3_metrics, logsizes)
        rounded_mean, rounded_CI = sigdig(wmean, CI)
        print(f"FSRS v3 mean: {rounded_mean}±{rounded_CI}")

        SM17_metrics = np.array([item[metric] for item in SM17])
        wmean = np.average(SM17_metrics, weights=logsizes)
        CI = confidence_interval(SM17_metrics, logsizes)
        rounded_mean, rounded_CI = sigdig(wmean, CI)
        print(f"SM-17 mean: {rounded_mean}±{rounded_CI}")

        SM16_metrics = np.array([item[metric] for item in SM16])
        wmean = np.average(SM16_metrics, weights=logsizes)
        CI = confidence_interval(SM16_metrics, logsizes)
        rounded_mean, rounded_CI = sigdig(wmean, CI)
        print(f"SM-16 mean: {rounded_mean}±{rounded_CI}")


        t_stat, p_value, df = ttest_ind(
            FSRS45_metrics, SM17_metrics, weights=(sizes, sizes)
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

        print(f"Cohen's d: {cohen_d(FSRS45_metrics, SM17_metrics, sizes)}")
