import json
import math
import pathlib
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def wilcoxon_effect_size(x, y):
    """
    Calculate the effect size r for Wilcoxon signed-rank test
    """
    wilcoxon_result = stats.wilcoxon(x, y, zero_method="wilcox", correction=False)

    W = wilcoxon_result.statistic
    p_value = wilcoxon_result.pvalue

    differences = np.array(x) - np.array(y)
    differences = differences[differences != 0]
    n = len(differences)

    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    z = (W - mu) / sigma

    r = z / np.sqrt(n)

    return {
        "W": W,
        "p_value": p_value,
        "z": z,
        "r": abs(r),
        "mid": np.median(differences),
    }


def ttest_effect_size(x, y):
    ttest_result = stats.ttest_rel(x, y)
    cohen_d = (np.mean(x) - np.mean(y)) / np.sqrt(
        (np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2
    )
    return {
        "t": ttest_result.statistic,
        "p_value": ttest_result.pvalue,
        "cohen_d": abs(cohen_d),
        "mean_diff": np.mean(x) - np.mean(y),
    }


def logp_wilcox(x, y, correction=False):
    # method='wilcox'
    # mode='approx'
    # alternative='two-sided'
    assert len(x) == len(y)
    x = np.asarray(x)
    y = np.asarray(y)

    def rankdata(a, method="average"):
        a = np.asarray(a)
        if a.size == 0:
            return np.empty(a.shape)
        sorter = np.argsort(a)
        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size, dtype=np.intp)

        if method == "ordinal":
            result = inv + 1
        else:
            a = a[sorter]
            obs = np.r_[True, a[1:] != a[:-1]]
            dense = obs.cumsum()[inv]

            if method == "dense":
                result = dense
            else:
                # cumulative counts of each unique value
                count = np.r_[np.nonzero(obs)[0], len(obs)]

                if method == "max":
                    result = count[dense]

                if method == "min":
                    result = count[dense - 1] + 1

                if method == "average":
                    result = 0.5 * (count[dense] + count[dense - 1] + 1)

        return result

    diff = x - y
    count = diff.size

    ranks = rankdata(abs(diff))
    r_plus = np.sum((diff > 0) * ranks)
    r_minus = np.sum((diff < 0) * ranks)
    if r_plus > r_minus:
        # x is greater than y
        which_one = 0
    else:
        # y is greater than x
        which_one = 1

    T = min(r_plus, r_minus)

    mn = count * (count + 1.0) * 0.25
    se = count * (count + 1.0) * (2.0 * count + 1.0)

    replist, repnum = stats.find_repeats(ranks)
    if repnum.size != 0:
        # correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = np.sqrt(se / 24)

    # apply continuity correction if applicable
    d = 0
    if correction:
        d = 0.5 * np.sign(T - mn)

    # compute statistic
    z = (T - mn - d) / se
    if abs(z) > 37:
        a = 0.62562732
        b = 0.22875463
        logp_approx = np.log1p(-np.exp(-a * abs(z))) - np.log(abs(z)) - (z**2) / 2 - b
    else:
        logp_approx = np.log(2.0 * stats.norm.sf(abs(z)))

    # returns the decimal logarithm of the p-value
    return np.log10(np.e) * logp_approx, which_one


def format(exponent, n):
    sci_notation_exponent = math.floor(exponent)
    sci_notation_mantissa = 10 ** (exponent - sci_notation_exponent)
    if round(sci_notation_mantissa, n) == 10:
        return f"{sci_notation_mantissa / 10:.{n}f}e{sci_notation_exponent + 1:.0f}"
    elif round(sci_notation_mantissa, n) < 1:
        return f"{sci_notation_mantissa * 10:.{n}f}e{sci_notation_exponent - 1:.0f}"
    else:
        return f"{sci_notation_mantissa:.{n}f}e{sci_notation_exponent:.0f}"


if __name__ == "__main__":
    models = (
        "FSRS-6",
        "MOVING-AVG",
        "AVG",
        "FSRS-4.5",
        "FSRS-5",
        "FSRS-6-default",
        "FSRSv4",
        "SM16",
        "FSRSv3",
        "SM17",
    )
    csv_name = f"{len(models)} models.csv"
    df = pd.DataFrame()
    for model in models:
        RMSE = []
        logloss = []
        result_files = pathlib.Path("./result").glob("*.json")
        for result_file in result_files:
            with open(result_file, "r") as f:
                result = json.load(f)
                logloss.append(result[model]["LogLoss"])
                RMSE.append(result[model]["RMSE(bins)"])
        print(f"Model: {model}")
        result_dir = pathlib.Path(f"./result/{model}")
        result_files = result_dir.glob("*.json")
        for result_file in result_files:
            with open(result_file, "r") as f:
                result = json.load(f)
                logloss.append(result[model]["LogLoss"])
                RMSE.append(result[model]["RMSE(bins)"])
        series1 = pd.Series(logloss, name=f"{model}, LogLoss")
        series2 = pd.Series(RMSE, name=f"{model}, RMSE (bins)")
        df = pd.concat([df, series1], axis=1)
        df = pd.concat([df, series2], axis=1)

    df.to_csv(csv_name)

    # you have to run the commented out code above first
    df = pd.read_csv(csv_name)

    n_collections = len(df)
    print(n_collections)
    models_name = list(models)

    n = len(models_name)
    wilcox = np.full((n, n), -1.0)
    color_wilcox = np.full((n, n), -1.0)
    ttest = np.full((n, n), -1.0)
    color_ttest = np.full((n, n), -1.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                wilcox[i, j] = np.nan
                color_wilcox[i, j] = np.nan
                ttest[i, j] = np.nan
                color_ttest[i, j] = np.nan
            else:
                df1 = df[f"{models_name[i]}, LogLoss"]
                df2 = df[f"{models_name[j]}, LogLoss"]
                result = wilcoxon_effect_size(df1[:n_collections], df2[:n_collections])
                p_value = result["p_value"]
                wilcox[i, j] = result["r"]

                if p_value > 0.05:
                    # color for insignificant p-values
                    color_wilcox[i, j] = 3
                else:
                    if result["mid"] > 0:
                        if result["r"] > 0.5:
                            color_wilcox[i, j] = 0
                        elif result["r"] > 0.2:
                            color_wilcox[i, j] = 1
                        else:
                            color_wilcox[i, j] = 2
                    else:
                        if result["r"] > 0.5:
                            color_wilcox[i, j] = 6
                        elif result["r"] > 0.2:
                            color_wilcox[i, j] = 5
                        else:
                            color_wilcox[i, j] = 4

                result = ttest_effect_size(df1[:n_collections], df2[:n_collections])
                ttest[i, j] = result["cohen_d"]
                if result["p_value"] > 0.05:
                    # color for insignificant p-values
                    color_ttest[i, j] = 3
                else:
                    if result["mean_diff"] > 0:
                        if result["cohen_d"] > 0.5:
                            color_ttest[i, j] = 0
                        elif result["cohen_d"] > 0.2:
                            color_ttest[i, j] = 1
                        else:
                            color_ttest[i, j] = 2
                    else:
                        if result["cohen_d"] > 0.5:
                            color_ttest[i, j] = 6
                        elif result["cohen_d"] > 0.2:
                            color_ttest[i, j] = 5
                        else:
                            color_ttest[i, j] = 4

    # small changes to labels
    index_v3 = models_name.index("FSRSv3")
    index_v4 = models_name.index("FSRSv4")
    index_sm16 = models_name.index("SM16")
    index_sm17 = models_name.index("SM17")
    index_fsrs_6_default = models_name.index("FSRS-6-default")
    models_name[index_v3] = "FSRS v3"
    models_name[index_v4] = "FSRS v4"
    models_name[index_sm16] = "SM-16"
    models_name[index_sm17] = "SM-17"
    models_name[index_fsrs_6_default] = "FSRS-6\ndefault\nparameters"

    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    ax.set_title(
        f"Wilcoxon signed-rank test, r-values ({n_collections} collections)",
        fontsize=24,
        pad=30,
    )
    cmap = matplotlib.colors.ListedColormap(
        ["darkred", "red", "coral", "silver", "limegreen", "#199819", "darkgreen"]
    )
    plt.imshow(
        color_wilcox,
        interpolation="none",
        vmin=color_wilcox[~np.isnan(color_wilcox)].min(),
        cmap=cmap,
    )

    for i in range(n):
        for j in range(n):
            if math.isnan(wilcox[i][j]):
                pass
            else:
                text = ax.text(
                    j,
                    i,
                    f"{wilcox[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=14,
                )

    ax.set_xticks(np.arange(n), labels=models_name, fontsize=16, rotation=45)
    ax.set_yticks(np.arange(n), labels=models_name, fontsize=16)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")
    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)
    pathlib.Path("./plots").mkdir(parents=True, exist_ok=True)
    title = f"Wilcoxon-{n_collections}-collections"
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    ax.set_title(
        f"T-test, Cohen's d ({n_collections} collections)",
        fontsize=24,
        pad=30,
    )
    plt.imshow(
        color_ttest,
        interpolation="none",
        vmin=color_ttest[~np.isnan(color_ttest)].min(),
        cmap=cmap,
    )
    for i in range(n):
        for j in range(n):
            if math.isnan(ttest[i][j]):
                pass
            else:
                text = ax.text(
                    j,
                    i,
                    f"{ttest[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=14,
                )

    ax.set_xticks(np.arange(n), labels=models_name, fontsize=16, rotation=45)
    ax.set_yticks(np.arange(n), labels=models_name, fontsize=16)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")
    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)
    title = f"T-test-{n_collections}-collections"
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")

    percentages = np.full((n, n), -1.0)
    for i in range(n):
        for j in range(n):
            if i == j:  # diagonal
                pass
            elif percentages[i, j] > 0:  # we already calculated this one
                pass
            else:
                df1 = df[f"{models[i]}, LogLoss"]
                df2 = df[f"{models[j]}, LogLoss"]
                greater = 0
                lower = 0
                # there is probably a better way to do this using Pandas
                for value1, value2 in zip(df1, df2):
                    if value1 > value2:
                        greater += 1
                    else:
                        lower += 1
                percentages[i, j] = lower / (greater + lower)

                true_i_j = percentages[i, j]
                true_j_i = 1 - percentages[i, j]
                i_j_up = math.ceil(true_i_j * 1000) / 1000
                i_j_down = math.floor(true_i_j * 1000) / 1000
                j_i_up = math.ceil(true_j_i * 1000) / 1000
                j_i_down = math.floor(true_j_i * 1000) / 1000

                up_down_error = abs(i_j_up - true_i_j) + abs(
                    j_i_down - true_j_i
                )  # sum of rounding errors
                down_up_error = abs(i_j_down - true_i_j) + abs(
                    j_i_up - true_j_i
                )  # sum of rounding errors
                if (
                    up_down_error < down_up_error
                ):  # choose whichever combination of rounding results in the lowest total absolute error
                    percentages[i, j] = i_j_up
                    percentages[j, i] = j_i_down
                else:
                    percentages[i, j] = i_j_down
                    percentages[j, i] = j_i_up

    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    ax.set_title(
        f"Percent of collections where algorithm A (row) outperforms algorithm B (column)",
        fontsize=15,
        pad=10,
    )

    def rgb2hex(list):
        return f"#{int(round(list[0])):02x}{int(round(list[1])):02x}{int(round(list[2])):02x}"

    start_color = [255, 0, 0]
    end_color = [45, 180, 0]
    N = 256
    colors = ["white", rgb2hex(start_color)]
    positions = [0, 1e-6]
    for i in range(1, N + 1):
        pos = i / N
        # this results in brighter colors than linear
        quadratic_interp_R = np.sqrt(
            pos * np.power(end_color[0], 2) + (1 - pos) * np.power(start_color[0], 2)
        )
        quadratic_interp_G = np.sqrt(
            pos * np.power(end_color[1], 2) + (1 - pos) * np.power(start_color[1], 2)
        )
        quadratic_interp_B = np.sqrt(
            pos * np.power(end_color[2], 2) + (1 - pos) * np.power(start_color[2], 2)
        )
        RGB_list = [quadratic_interp_R, quadratic_interp_G, quadratic_interp_B]
        colors.append(rgb2hex(RGB_list))
        positions.append(pos)

    cmap = LinearSegmentedColormap.from_list(
        "custom_linear", list(zip(positions, colors))
    )

    def clamp_percentages(percentages):
        percentages = np.clip(percentages, a_min=0.005, a_max=1.0)
        for i in range(n):
            percentages[i, i] = -1.0
        return percentages

    plt.imshow(clamp_percentages(percentages), vmin=0, cmap=cmap)

    for i in range(n):
        for j in range(n):
            if percentages[i, j] == -1:
                pass
            else:
                string = f"{100*percentages[i, j]:.1f}%"
                text = ax.text(
                    j,
                    i,
                    string,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                )

    ax.set_xticks(np.arange(n), labels=models_name, fontsize=16, rotation=45)
    ax.set_yticks(np.arange(n), labels=models_name, fontsize=16)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")

    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)

    title = f"Superiority-{n_collections}-collections"
    plt.savefig(
        f"./plots/{title}.png",
        bbox_inches="tight",
    )

    # Universal Metrics Matrix Heatmap
    print("Generating Universal Metrics heatmap...")

    # Collect Universal Metrics data
    universal_metrics_data = {}
    result_files = pathlib.Path("./result").glob("*.json")

    for result_file in result_files:
        with open(result_file, "r") as f:
            result = json.load(f)
            if "Universal_Metrics" in result:
                for metric_name, metric_value in result["Universal_Metrics"].items():
                    if metric_name not in universal_metrics_data:
                        universal_metrics_data[metric_name] = []
                    universal_metrics_data[metric_name].append(metric_value)

    if universal_metrics_data:
        # Load user sizes for weighted average
        sizes = []
        result_files = pathlib.Path("./result").glob("*.json")
        for result_file in result_files:
            with open(result_file, "r") as f:
                result = json.load(f)
                sizes.append(result["size"])
        sizes = np.array(sizes)

        # Calculate weighted average Universal Metrics for each pair
        um_matrix_data = {}
        for metric_name, values in universal_metrics_data.items():
            values_array = np.array(values)
            um_matrix_data[metric_name] = np.average(values_array, weights=sizes)

        # Get all unique algorithms
        all_algorithms = set()
        for metric_name in um_matrix_data.keys():
            algo_a, algo_b = metric_name.split("_evaluated_by_")
            all_algorithms.add(algo_a)
            all_algorithms.add(algo_b)

        # Calculate average Universal Metrics for each algorithm (as evaluated)
        algo_avg_um = {}
        for algo_name in all_algorithms:
            scores = []
            for metric_name, value in um_matrix_data.items():
                if metric_name.startswith(f"{algo_name}_evaluated_by_"):
                    scores.append(value)
            if scores:
                algo_avg_um[algo_name] = np.mean(scores)

        # Sort algorithms by average Universal Metric (lower is better)
        sorted_algorithms = sorted(algo_avg_um.items(), key=lambda x: x[1])
        sorted_algorithms = [algo for algo, _ in sorted_algorithms]
        n_um = len(sorted_algorithms)

        # Create Universal Metrics matrix
        um_matrix = np.full((n_um, n_um), np.nan)

        for i, algo_a in enumerate(sorted_algorithms):
            for j, algo_b in enumerate(sorted_algorithms):
                if i != j:  # Skip diagonal
                    metric_name = f"{algo_a}_evaluated_by_{algo_b}"
                    if metric_name in um_matrix_data:
                        um_matrix[i, j] = um_matrix_data[metric_name]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
        ax.set_title(
            f"Universal Metrics Matrix ({n_collections} collections)",
            fontsize=24,
            pad=30,
        )

        # Use a colormap that goes from low (good) to high (bad) values
        cmap = (
            plt.cm.viridis_r
        )  # Reverse viridis: bright = low (good), dark = high (bad)

        # Create the heatmap
        im = ax.imshow(um_matrix, cmap=cmap, interpolation="none")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Universal Metric (lower is better)", fontsize=14)

        # Add text annotations
        for i in range(n_um):
            for j in range(n_um):
                if not np.isnan(um_matrix[i, j]):
                    text = ax.text(
                        j,
                        i,
                        f"{um_matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if um_matrix[i, j] > np.nanmean(um_matrix)
                            else "black"
                        ),
                        fontsize=12,
                        weight="bold",
                    )
                else:
                    text = ax.text(
                        j,
                        i,
                        "-",
                        ha="center",
                        va="center",
                        color="gray",
                        fontsize=14,
                    )

        # Set labels
        ax.set_xticks(
            np.arange(n_um), labels=sorted_algorithms, fontsize=16, rotation=45
        )
        ax.set_yticks(np.arange(n_um), labels=sorted_algorithms, fontsize=16)
        ax.set_xticks(np.arange(n_um) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_um) - 0.5, minor=True)

        # Add grid
        plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")

        # Enhance borders
        for location in ["left", "right", "top", "bottom"]:
            ax.spines[location].set_linewidth(2)

        # Add axis labels
        ax.set_xlabel("Evaluating Algorithm", fontsize=16)
        ax.set_ylabel("Evaluated Algorithm", fontsize=16)

        # Save the plot
        title = f"Universal-Metrics-Matrix-{n_collections}-collections"
        plt.savefig(f"./plots/{title}.png", bbox_inches="tight")
        plt.close()

        print(f"Universal Metrics heatmap saved as ./plots/{title}.png")
    else:
        print("No Universal Metrics data found for heatmap generation.")
