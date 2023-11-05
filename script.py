import pandas as pd
from datetime import datetime
from itertools import accumulate
from fsrs_optimizer import (
    lineToTensor,
    collate_fn,
    power_forgetting_curve,
    FSRS,
    RevlogDataset,
    WeightClipper,
)
from models import FSRSv3, FSRS3WeightClipper
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
import pathlib
import json

tqdm.pandas()


def data_preprocessing(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, encoding="utf-8")
    except:
        df = pd.read_csv(csv_file_path, encoding="gbk")
    df.columns = df.columns.str.strip()

    def convert_to_datetime(date_str):
        try:
            return datetime.strptime(date_str, "%b %d %Y %H:%M:%S")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%m月 %d %Y %H:%M:%S")
            except ValueError:
                try:
                    return datetime.strptime(date_str, "%d/%m/%Y %H:%M")
                except ValueError:
                    return pd.NaT

    df["Date"] = df["Date"].apply(convert_to_datetime)
    df.dropna(subset=["Date"], inplace=True)
    df = df[df["Success"].isin([0, 1]) & df["Grade"].isin([0, 1, 2, 3, 4, 5])].copy()
    df = df[
        (df["R (SM16)"] <= 1) & (df["R (SM17)"] <= 1) & (df["R (SM17)(exp)"] <= 1)
    ].copy()
    df["R (SM17)(exp)"] = df["R (SM17)(exp)"].map(lambda x: np.clip(x, 0.001, 0.999))
    dataset = df[
        [
            "Date",
            "Element No",
            "Used interval",
            "R (SM16)",
            "R (SM17)",
            "R (SM17)(exp)",
            "Grade",
            "Success",
        ]
    ].sort_values(by=["Element No", "Date"])
    dataset.rename(
        columns={
            "Element No": "card_id",
            "Date": "review_date",
            "Used interval": "delta_t",
            "R (SM17)(exp)": "R (SM17(exp))",
            "Success": "y",
        },
        inplace=True,
    )
    dataset["i"] = dataset.groupby("card_id").cumcount() + 1
    dataset["review_rating"] = dataset["Grade"].map(
        {0: 1, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}
    )

    def cum_concat(x):
        return list(accumulate(x))

    t_history = dataset.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[int(i)] for i in x])
    )
    dataset["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history for item in sublist
    ]
    r_history = dataset.groupby("card_id", group_keys=False)["review_rating"].apply(
        lambda x: cum_concat([[int(i)] for i in x])
    )
    dataset["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history for item in sublist
    ]
    dataset = dataset[
        (dataset["i"] > 1)
        & (dataset["delta_t"] > 0)
        & (dataset["t_history"].str.count(",0") == 0)
    ].copy()
    return dataset


def FSRS_v4_train(revlogs):
    revlogs = revlogs[
        (revlogs["i"] > 1)
        & (revlogs["delta_t"] > 0)
        & (revlogs["t_history"].str.count(",0") == 0)
    ].copy()
    revlogs["tensor"] = revlogs.progress_apply(
        lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]), axis=1
    )
    revlogs.sort_values(by=["review_date"], inplace=True)
    revlogs.reset_index(drop=True, inplace=True)

    model = FSRS(
        [
            0.4,
            0.6,
            2.4,
            5.8,
            4.93,
            0.94,
            0.86,
            0.01,
            1.49,
            0.14,
            0.94,
            2.18,
            0.05,
            0.34,
            1.26,
            0.29,
            2.61,
        ]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
    loss_fn = torch.nn.BCELoss(reduction="none")
    enable_experience_replay = True
    replay_steps = 32

    dataset = RevlogDataset(revlogs)
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn)
    clipper = WeightClipper()
    d = []
    s = []
    r = []

    for i, sample in enumerate(tqdm(dataloader)):
        model.train()
        optimizer.zero_grad()
        sequence, delta_t, label, seq_len = sample
        output, _ = model(sequence)
        stability, difficulty = output[seq_len - 1, 0].transpose(0, 1)
        d.append(difficulty.detach().numpy()[0])
        s.append(stability.detach().numpy()[0])
        retention = power_forgetting_curve(delta_t, stability)
        r.append(retention.detach().numpy()[0])
        loss = loss_fn(retention, label).sum()
        loss.backward()
        optimizer.step()
        model.apply(clipper)

        if enable_experience_replay and (i + 1) % replay_steps == 0:
            # experience replay
            replay_dataset = RevlogDataset(revlogs[: i + 1])  # avoid data leakage
            replay_generator = torch.Generator().manual_seed(42 + i)
            replay_dataloader = DataLoader(
                replay_dataset,
                batch_size=(i + 1) // 32,
                shuffle=True,
                collate_fn=collate_fn,
                generator=replay_generator,
            )
            for j, batch in enumerate(replay_dataloader):
                model.train()
                optimizer.zero_grad()
                sequences, delta_ts, labels, seq_lens = batch
                real_batch_size = seq_lens.shape[0]
                outputs, _ = model(sequences)
                stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]
                retentions = power_forgetting_curve(delta_ts, stabilities)
                loss = loss_fn(retentions, labels).sum()
                loss.backward()
                optimizer.step()
                model.apply(clipper)

    revlogs["R (FSRSv4)"] = r

    return revlogs


def FSRS_v3_train(revlogs):
    revlogs = revlogs[
        (revlogs["i"] > 1)
        & (revlogs["delta_t"] > 0)
        & (revlogs["t_history"].str.count(",0") == 0)
    ].copy()
    revlogs["tensor"] = revlogs.progress_apply(
        lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]), axis=1
    )
    revlogs.sort_values(by=["review_date"], inplace=True)
    revlogs.reset_index(drop=True, inplace=True)

    model = FSRSv3()
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
    loss_fn = torch.nn.BCELoss(reduction="none")
    enable_experience_replay = True
    replay_steps = 32

    dataset = RevlogDataset(revlogs)
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn)
    clipper = FSRS3WeightClipper()
    d = []
    s = []
    r = []

    for i, sample in enumerate(tqdm(dataloader)):
        model.train()
        optimizer.zero_grad()
        sequence, delta_t, label, seq_len = sample
        output, _ = model(sequence)
        stability, difficulty = output[seq_len - 1, 0].transpose(0, 1)
        d.append(difficulty.detach().numpy()[0])
        s.append(stability.detach().numpy()[0])
        retention = model.forgetting_curve(delta_t, stability)
        r.append(retention.detach().numpy()[0])
        loss = loss_fn(retention, label).sum()
        loss.backward()
        optimizer.step()
        model.apply(clipper)

        if enable_experience_replay and (i + 1) % replay_steps == 0:
            # experience replay
            replay_dataset = RevlogDataset(revlogs[: i + 1])  # avoid data leakage
            replay_generator = torch.Generator().manual_seed(42 + i)
            replay_dataloader = DataLoader(
                replay_dataset,
                batch_size=(i + 1) // 32,
                shuffle=True,
                collate_fn=collate_fn,
                generator=replay_generator,
            )
            for j, batch in enumerate(replay_dataloader):
                model.train()
                optimizer.zero_grad()
                sequences, delta_ts, labels, seq_lens = batch
                real_batch_size = seq_lens.shape[0]
                outputs, _ = model(sequences)
                stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]
                retentions = model.forgetting_curve(delta_ts, stabilities)
                loss = loss_fn(retentions, labels).sum()
                loss.backward()
                optimizer.step()
                model.apply(clipper)

    revlogs["R (FSRSv3)"] = r

    return revlogs


def evaluate(revlogs):
    sm16_rmse = mean_squared_error(revlogs["y"], revlogs["R (SM16)"], squared=False)
    sm17_rmse = mean_squared_error(
        revlogs["y"], revlogs["R (SM17(exp))"], squared=False
    )
    fsrs_v4_rmse = mean_squared_error(
        revlogs["y"], revlogs["R (FSRSv4)"], squared=False
    )
    fsrs_v3_rmse = mean_squared_error(
        revlogs["y"], revlogs["R (FSRSv3)"], squared=False
    )
    sm16_logloss = log_loss(revlogs["y"], revlogs["R (SM16)"])
    sm17_logloss = log_loss(revlogs["y"], revlogs["R (SM17(exp))"])
    fsrs_v4_logloss = log_loss(revlogs["y"], revlogs["R (FSRSv4)"])
    fsrs_v3_logloss = log_loss(revlogs["y"], revlogs["R (FSRSv3)"])
    return {
        "FSRSv4": {
            "RMSE": fsrs_v4_rmse,
            "LogLoss": fsrs_v4_logloss,
        },
        "FSRSv3": {
            "RMSE": fsrs_v3_rmse,
            "LogLoss": fsrs_v3_logloss,
        },
        "SM16": {
            "RMSE": sm16_rmse,
            "LogLoss": sm16_logloss,
        },
        "SM17": {
            "RMSE": sm17_rmse,
            "LogLoss": sm17_logloss,
        },
    }


def cross_comparison(revlogs, algoA, algoB):
    if algoA != algoB:
        cross_comparison_record = revlogs[[f"R ({algoA})", f"R ({algoB})", "y"]].copy()
        bin_algo = (
            algoA,
            algoB,
        )
        pair_algo = [(algoA, algoB), (algoB, algoA)]
    else:
        cross_comparison_record = revlogs[[f"R ({algoA})", "y"]].copy()
        bin_algo = (algoA,)
        pair_algo = [(algoA, algoA)]

    def get_bin(x, bins=20):
        return (
            np.log(np.minimum(np.floor(np.exp(np.log(bins + 1) * x) - 1), bins - 1) + 1)
            / np.log(bins)
        ).round(3)

    for algo in bin_algo:
        cross_comparison_record[f"{algo}_B-W"] = (
            cross_comparison_record[f"R ({algo})"] - cross_comparison_record["y"]
        )
        cross_comparison_record[f"{algo}_bin"] = cross_comparison_record[
            f"R ({algo})"
        ].map(get_bin)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.axhline(y=0.0, color="black", linestyle="-")

    universal_metric_list = []

    for algoA, algoB in pair_algo:
        cross_comparison_group = cross_comparison_record.groupby(by=f"{algoA}_bin").agg(
            {"y": ["mean"], f"{algoB}_B-W": ["mean"], f"R ({algoB})": ["mean", "count"]}
        )
        universal_metric = mean_squared_error(
            cross_comparison_group["y", "mean"],
            cross_comparison_group[f"R ({algoB})", "mean"],
            sample_weight=cross_comparison_group[f"R ({algoB})", "count"],
            squared=False,
        )
        cross_comparison_group[f"R ({algoB})", "percent"] = (
            cross_comparison_group[f"R ({algoB})", "count"]
            / cross_comparison_group[f"R ({algoB})", "count"].sum()
        )
        ax.scatter(
            cross_comparison_group.index,
            cross_comparison_group[f"{algoB}_B-W", "mean"],
            s=cross_comparison_group[f"R ({algoB})", "percent"] * 1024,
            alpha=0.5,
        )
        ax.plot(
            cross_comparison_group[f"{algoB}_B-W", "mean"],
            label=f"{algoB} by {algoA}, UM={universal_metric:.4f}",
        )
        universal_metric_list.append(universal_metric)

    ax.legend(loc="lower center")
    ax.grid(linestyle="--")
    ax.set_title(f"{algoA} vs {algoB}")
    ax.set_xlabel("Predicted R")
    ax.set_ylabel("B-W Metric")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    fig.show()

    return universal_metric_list


if __name__ == "__main__":
    for file in pathlib.Path("dataset").iterdir():
        plt.close("all")
        if file.is_file() and file.suffix == ".csv":
            if file.stem in map(lambda x: x.stem, pathlib.Path("result").iterdir()):
                print(f"{file.stem} already exists, skip")
                continue
            try:
                _, user = file.stem.split("_")
            except:
                continue
            revlogs = data_preprocessing(file)
            revlogs = FSRS_v3_train(revlogs)
            revlogs = FSRS_v4_train(revlogs)
            result = evaluate(revlogs)
            # sm17_by_sm16, sm16_by_sm17 = cross_comparison(revlogs, "SM16", "SM17(exp)")
            # sm17_by_fsrs, fsrs_by_sm17 = cross_comparison(revlogs, "FSRS", "SM17(exp)")
            # fsrs_by_sm16, sm16_by_fsrs = cross_comparison(revlogs, "SM16", "FSRS")
            # result["FSRS"]["UniversalMetric"] = (fsrs_by_sm17 + fsrs_by_sm16) / 2
            # result["SM16"]["UniversalMetric"] = (sm16_by_sm17 + sm16_by_fsrs) / 2
            # result["SM17"]["UniversalMetric"] = (sm17_by_sm16 + sm17_by_fsrs) / 2
            sm17_rmse_bin = cross_comparison(revlogs, "SM17(exp)", "SM17(exp)")[0]
            sm16_rmse_bin = cross_comparison(revlogs, "SM16", "SM16")[0]
            fsrs_v3_rmse_bin = cross_comparison(revlogs, "FSRSv3", "FSRSv3")[0]
            fsrs_v4_rmse_bin = cross_comparison(revlogs, "FSRSv4", "FSRSv4")[0]
            result["SM17"]["RMSE(bins)"] = sm17_rmse_bin
            result["SM16"]["RMSE(bins)"] = sm16_rmse_bin
            result["FSRSv3"]["RMSE(bins)"] = fsrs_v3_rmse_bin
            result["FSRSv4"]["RMSE(bins)"] = fsrs_v4_rmse_bin

            result["user"] = user
            result["size"] = revlogs.shape[0]
            # save as json
            pathlib.Path("result").mkdir(parents=True, exist_ok=True)
            with open(f"result/{file.stem}.json", "w") as f:
                json.dump(result, f, indent=4)
