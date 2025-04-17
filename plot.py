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
        "FSRS-4.5",
        "FSRS-5",
        "FSRSv4",
        "FSRSv3",
        "SM17",
        "SM16",
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
    wilcox = [[-1 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                wilcox[i][j] = float("NaN")
            else:
                df1 = df[f"{models_name[i]}, LogLoss"]
                df2 = df[f"{models_name[j]}, LogLoss"]
                if n_collections > 50:
                    result = logp_wilcox(df1[:n_collections], df2[:n_collections])[0]
                else:
                    result = np.log10(
                        stats.wilcoxon(df1[:n_collections], df2[:n_collections]).pvalue
                    )
                wilcox[i][j] = result

    color_wilcox = [[-1 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                color_wilcox[i][j] = float("NaN")
            else:
                df1 = df[f"{models_name[i]}, LogLoss"]
                df2 = df[f"{models_name[j]}, LogLoss"]
                # we'll need the second value return by my function to determine the color
                approx = logp_wilcox(df1[:n_collections], df2[:n_collections])
                if n_collections > 50:
                    result = approx[0]
                else:
                    # use the exact result for small n
                    result = np.log10(
                        stats.wilcoxon(df1[:n_collections], df2[:n_collections]).pvalue
                    )

                if np.power(10, result) > 0.05:
                    # color for insignificant p-values
                    color_wilcox[i][j] = 0.5
                else:
                    if approx[1] == 0:
                        color_wilcox[i][j] = 0
                    else:
                        color_wilcox[i][j] = 1

    # small changes to labels
    index_v3 = models_name.index("FSRSv3")
    index_v4 = models_name.index("FSRSv4")
    index_sm16 = models_name.index("SM16")
    index_sm17 = models_name.index("SM17")
    models_name[index_v3] = "FSRS v3"
    models_name[index_v4] = "FSRS v4"
    models_name[index_sm16] = "SM-16"
    models_name[index_sm17] = "SM-17"

    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    ax.set_title(
        f"Wilcoxon signed-rank test, p-values ({n_collections} collections)",
        fontsize=24,
        pad=30,
    )
    cmap = matplotlib.colors.ListedColormap(["red", "#989a98", "#2db300"])
    plt.imshow(color_wilcox, interpolation="none", vmin=0, cmap=cmap)

    for i in range(n):
        for j in range(n):
            if math.isnan(wilcox[i][j]):
                pass
            else:
                if 10 ** wilcox[i][j] > 0.1:
                    string = f"{10 ** wilcox[i][j]:.2f}"
                elif 10 ** wilcox[i][j] > 0.01:
                    string = f"{10 ** wilcox[i][j]:.3f}"
                else:
                    string = format(wilcox[i][j], 1)
                text = ax.text(
                    j,
                    i,
                    string,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=20,
                )

    ax.set_xticks(np.arange(n), labels=models_name, fontsize=16)
    ax.set_yticks(np.arange(n), labels=models_name, fontsize=16)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")
    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)
    pathlib.Path("./plots").mkdir(parents=True, exist_ok=True)
    title = f"Wilcoxon-{n_collections}-collections"
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
                    fontsize=18,
                )

    ax.set_xticks(np.arange(n), labels=models_name, fontsize=16)
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
