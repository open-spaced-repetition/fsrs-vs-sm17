import pathlib
import json
import numpy as np
import scipy.stats
import math


def cohen_d(group1, group2, size):
    # weighted mean
    mean1, mean2 = np.average(group1, weights=size), np.average(group2, weights=size)
    # weighted variance
    var1, var2 = (
        np.average((group1 - mean1) ** 2, weights=size),
        np.average((group2 - mean2) ** 2, weights=size),
    )

    d = (mean1 - mean2) / np.sqrt((var1 + var2) / 2)

    return d


def sigdig(value, CI):
    def num_lead_zeros(x):
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = 2
    decimals = n_lead_zeros_CI + CI_sigdigs
    rounded_CI = round(CI, decimals)
    rounded_value = round(value, decimals - 1)
    return str(f"{rounded_value:.{decimals - 1}f}"), str(f"{rounded_CI:.{decimals}f}")


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


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))


if __name__ == "__main__":
    FSRSv3 = ("FSRSv3", [])
    FSRSv4 = ("FSRSv4", [])
    FSRS_4_5 = ("FSRS-4.5", [])
    FSRS_5 = ("FSRS-5", [])
    SM17 = ("SM17", [])
    SM16 = ("SM16", [])
    sizes = []
    result_dir = pathlib.Path("./result")
    result_files = result_dir.glob("*.json")
    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
            FSRSv3[1].append(result["FSRSv3"])
            FSRSv4[1].append(result["FSRSv4"])
            FSRS_4_5[1].append(result["FSRS-4.5"])
            FSRS_5[1].append(result["FSRS-5"])
            SM17[1].append(result["SM17"])
            SM16[1].append(result["SM16"])
            sizes.append(result["size"])

    print(f"Total number of users: {len(sizes)}")
    sizes = np.array(sizes)
    print(f"Total size: {sizes.sum()}")
    for scale, size in (
        ("reviews", np.array(sizes)),
        ("log(reviews)", np.log(sizes)),
        ("users", np.ones_like(sizes)),
    ):
        print(f"Scale: {scale}")
        for metric in ("LogLoss", "RMSE(bins)"):
            for model in (FSRS_5, FSRS_4_5, FSRSv4, FSRSv3, SM17, SM16):
                metrics = np.array([item[metric] for item in model[1]])
                wmean, wstd = weighted_avg_and_std(metrics, size)
                CI = confidence_interval(metrics, size)
                rounded_mean, rounded_CI = sigdig(wmean, CI)
                print(f"{model[0]} {metric}: {rounded_mean}±{rounded_CI}")
                print(f"{model[0]} {metric} (mean±std): {wmean:.3f}±{wstd:.3f}")
            print()
