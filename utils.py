import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error


def get_bin(x, bins=10):
    # Simple equal-width binning with 0.1 intervals from 0 to 1
    # This matches the binning method shown in the graph
    return np.round(x * bins) / bins


def cross_comparison(revlogs, algoA, algoB):
    cross_comparison_record = revlogs[[f"R ({algoA})", f"R ({algoB})", "y"]].copy()

    for algo in (algoA, algoB):
        cross_comparison_record[f"{algo}_B-W"] = (
            cross_comparison_record[f"R ({algo})"] - cross_comparison_record["y"]
        )
        cross_comparison_record[f"{algo}_bin"] = cross_comparison_record[
            f"R ({algo})"
        ].map(get_bin)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.axhline(y=0.0, color="black", linestyle="-")
    result = {}

    for referee, player in [(algoA, algoB), (algoB, algoA)]:
        cross_comparison_group = cross_comparison_record.groupby(
            by=f"{referee}_bin"
        ).agg(
            {
                "y": ["mean"],
                f"{player}_B-W": ["mean"],
                f"R ({player})": ["mean", "count"],
            }
        )
        universal_metric = root_mean_squared_error(
            cross_comparison_group["y", "mean"],
            cross_comparison_group[f"R ({player})", "mean"],
            sample_weight=cross_comparison_group[f"R ({player})", "count"],
        )
        result[f"{player}_evaluated_by_{referee}"] = round(universal_metric, 4)
        cross_comparison_group[f"R ({player})", "percent"] = (
            cross_comparison_group[f"R ({player})", "count"]
            / cross_comparison_group[f"R ({player})", "count"].sum()
        )
        ax.scatter(
            cross_comparison_group.index,
            cross_comparison_group[f"{player}_B-W", "mean"],
            s=cross_comparison_group[f"R ({player})", "percent"] * 1024,
            alpha=0.5,
        )
        ax.plot(
            cross_comparison_group[f"{player}_B-W", "mean"],
            label=f"{player} by {referee}, UM={universal_metric:.4f}",
        )

    ax.legend(loc="lower center")
    ax.grid(linestyle="--")
    ax.set_title(f"{algoA} vs {algoB}")
    ax.set_xlabel("Predicted R")
    ax.set_ylabel("B-W Metric")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.show()

    return result


if __name__ == "__main__":
    values = np.linspace(0, 1, 100)
    bins = 10
    plt.plot(values, [get_bin(value, bins) for value in values])
    plt.grid(True)
    plt.title(f"Equal-width binning with {bins} bins")
    plt.xlabel("Value")
    plt.ylabel("Bin")
    plt.show()
