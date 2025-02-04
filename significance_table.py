import json
import math
import pathlib
import warnings

import matplotlib
import matplotlib.pyplot as plt
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
        "FSRS-4.5",
        "FSRS-5",
        "SM17",
        "FSRSv4",
        "FSRSv3",
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
    models2 = []
    for i in range(len(models)):
        models2.append(models[i])

    n = len(models2)
    wilcox = [[-1 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                wilcox[i][j] = float("NaN")
            else:
                df1 = df[f"{models2[i]}, RMSE (bins)"]
                df2 = df[f"{models2[j]}, RMSE (bins)"]
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
                df1 = df[f"{models2[i]}, RMSE (bins)"]
                df2 = df[f"{models2[j]}, RMSE (bins)"]
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
    index_v3 = models2.index("FSRSv3")
    index_v4 = models2.index("FSRSv4")
    index_sm16 = models2.index("SM16")
    index_sm17 = models2.index("SM17")
    models2[index_v3] = "FSRS v3"
    models2[index_v4] = "FSRS v4"
    models2[index_sm16] = "SM-16"
    models2[index_sm17] = "SM-17"

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
                    fontsize=24,
                )

    ax.set_xticks(np.arange(n), labels=models2, fontsize=18)
    ax.set_yticks(np.arange(n), labels=models2, fontsize=18)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    plt.grid(True, alpha=1, color="black", linewidth=2, which="minor")
    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(2)
    pathlib.Path("./plots").mkdir(parents=True, exist_ok=True)
    title = f"Wilcoxon-{n_collections}-collections"
    plt.savefig(f"./plots/{title}.png", bbox_inches="tight")
    plt.show()
