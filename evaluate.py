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
    rounded_value = round(value, decimals)
    if n_lead_zeros_CI > num_lead_zeros(rounded_CI):
        return str(f"{round(value, decimals - 1):.{decimals - 1}f}"), str(
            f"{round(CI, decimals - 1):.{decimals - 1}f}"
        )
    else:
        return str(f"{rounded_value:.{decimals}f}"), str(f"{rounded_CI:.{decimals}f}")


# tests to ensure that sigdigs is working as intended
value = 0.084011111
CI = 0.0010011111
assert sigdig(value, CI) == ("0.0840", "0.0010")

value2 = 0.083999999
CI2 = 0.0009999999
assert sigdig(value2, CI2) == ("0.0840", "0.0010")


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
    FSRS_6 = ("FSRS-6", [])
    FSRS_6_default = ("FSRS-6-default", [])
    SM17 = ("SM-17", [])
    SM16 = ("SM-16", [])
    AVG = ("AVG", [])
    MOVING_AVG = ("MOVING-AVG", [])
    ADVERSARIAL = ("ADVERSARIAL", [])
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
            FSRS_6[1].append(result["FSRS-6"])
            FSRS_6_default[1].append(result["FSRS-6-default"])
            SM17[1].append(result["SM17"])
            SM16[1].append(result["SM16"])
            AVG[1].append(result["AVG"])
            MOVING_AVG[1].append(result["MOVING-AVG"])
            sizes.append(result["size"])
            if "ADVERSARIAL" in result:
                ADVERSARIAL[1].append(result["ADVERSARIAL"])

    print(f"Total users: {len(sizes)}")
    sizes = np.array(sizes)
    print(f"\nTotal repetitions: {sizes.sum():,}\n")

    # Define metrics with their display names and sort order
    metrics_config = [
        ("LogLoss", "Log Loss↓", "lower"),
        ("RMSE(bins)", "RMSE (bins)↓", "lower"),
        ("AUC", "AUC↑", "higher"),
    ]

    # Define model order for display
    models = [
        FSRS_6,
        FSRS_5,
        FSRS_4_5,
        AVG,
        FSRSv4,
        SM17,
        SM16,
        FSRSv3,
        MOVING_AVG,
        FSRS_6_default,
    ]

    if len(ADVERSARIAL[1]) == len(sizes):
        models.append(ADVERSARIAL)
    elif len(ADVERSARIAL[1]) != 0:
        print(
            "Skipping adversarial aggregate metrics: result count mismatch with sizes."
        )

    for scale_name, size in [
        ("Weighted by number of repetitions", np.array(sizes)),
        ("Unweighted (per user)", np.ones_like(sizes)),
    ]:
        print(f"### {scale_name}\n")

        # Calculate all metrics for all models
        results = {}
        for model in models:
            results[model[0]] = {}
            for metric_key, _, _ in metrics_config:
                metrics = np.array([item[metric_key] for item in model[1]])
                wmean, wstd = weighted_avg_and_std(metrics, size)
                CI = confidence_interval(metrics, size)
                rounded_mean, rounded_CI = sigdig(wmean, CI)
                results[model[0]][metric_key] = (rounded_mean, rounded_CI, wmean)

        # Find best value for each metric
        best_values = {}
        for metric_key, _, direction in metrics_config:
            values = [(name, results[name][metric_key][2]) for name in results.keys()]
            if direction == "lower":
                best_name = min(values, key=lambda x: x[1])[0]
            else:  # higher
                best_name = max(values, key=lambda x: x[1])[0]
            best_values[metric_key] = best_name

        # Print table header
        header = "| Algorithm |"
        separator = "| --- |"
        for _, display_name, _ in metrics_config:
            header += f" {display_name} |"
            separator += " --- |"
        print(header)
        print(separator)

        # Sort models by Log Loss (ascending)
        sorted_models = sorted(models, key=lambda m: results[m[0]]["LogLoss"][2])

        # Print table rows
        for model in sorted_models:
            model_name = model[0]
            row = f"| {model_name} |"
            for metric_key, _, _ in metrics_config:
                rounded_mean, rounded_CI, _ = results[model_name][metric_key]
                value_str = f"{rounded_mean}±{rounded_CI}"
                # Bold if best
                if best_values[metric_key] == model_name:
                    value_str = f"**{value_str}**"
                    row = row.replace(f"| {model_name} |", f"| **{model_name}** |")
                row += f" {value_str} |"
            print(row)

        print()

    # Calculate Universal Metrics
    print("### Universal Metrics (Cross Comparison)\n")

    # Collect all Universal Metrics from all users
    universal_metrics_data = {}
    result_files = result_dir.glob("*.json")

    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
            if "Universal_Metrics" in result:
                for metric_name, metric_value in result["Universal_Metrics"].items():
                    if metric_name not in universal_metrics_data:
                        universal_metrics_data[metric_name] = []
                    universal_metrics_data[metric_name].append(metric_value)

    if universal_metrics_data:
        # Calculate statistics for each Universal Metric
        um_results = {}
        for metric_name, values in universal_metrics_data.items():
            values_array = np.array(values)
            wmean, wstd = weighted_avg_and_std(values_array, sizes)
            CI = confidence_interval(values_array, sizes)
            rounded_mean, rounded_CI = sigdig(wmean, CI)
            um_results[metric_name] = (rounded_mean, rounded_CI, wmean)

        # Find best (lowest) Universal Metric for each algorithm
        algorithm_um_scores = {}
        for metric_name, (_, _, wmean) in um_results.items():
            # Extract algorithm name from metric name (e.g., "FSRS-6_evaluated_by_SM16" -> "FSRS-6")
            algo_name = metric_name.split("_evaluated_by_")[0]
            if algo_name not in algorithm_um_scores:
                algorithm_um_scores[algo_name] = []
            algorithm_um_scores[algo_name].append(wmean)

        # Calculate average UM score for each algorithm
        algo_avg_um = {}
        for algo_name, scores in algorithm_um_scores.items():
            algo_avg_um[algo_name] = np.mean(scores)

        # Sort algorithms by average Universal Metric (lower is better)
        sorted_algorithms = sorted(algo_avg_um.items(), key=lambda x: x[1])

        # Print Universal Metrics table
        print("| Algorithm | Average Universal Metric↓ |")
        print("| --- | --- |")
        for i, (algo_name, avg_um) in enumerate(sorted_algorithms):
            # Format the average UM value
            formatted_um = f"{avg_um:.4f}"
            if i == 0:  # Best algorithm
                print(f"| **{algo_name}** | **{formatted_um}** |")
            else:
                print(f"| {algo_name} | {formatted_um} |")

        print()
    else:
        print("No Universal Metrics data found in result files.\n")
