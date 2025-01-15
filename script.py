import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from datetime import datetime
from itertools import accumulate
from fsrs_optimizer import (
    lineToTensor,
    power_forgetting_curve,
    rmse_matrix,
    FSRS,
    BatchDataset,
    BatchLoader,
    ParameterClipper,
    DEFAULT_PARAMETER,
)
from models import FSRS3, FSRS4
from tqdm.auto import tqdm
from sklearn.metrics import log_loss
from pathlib import Path

tqdm.pandas()


def data_preprocessing(csv_file_path, save_csv=False):
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
    dataset = dataset[
        (dataset["i"] > 1)
        & (dataset["delta_t"] > 0)
        & (dataset["t_history"].str.count(",0") == 0)
    ].copy()
    dataset["tensor"] = dataset.progress_apply(
        lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]), axis=1
    )
    dataset.sort_values(by=["review_date"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    if save_csv:
        Path("converted").mkdir(parents=True, exist_ok=True)
        save = dataset[
            [
                "card_id",
                "review_date",
                "delta_t",
                "review_rating",
                "R (SM16)",
                "R (SM17(exp))",
            ]
        ].copy()
        save["review_date"] = save["review_date"].rank(method="dense").astype("int64")
        save["card_id"] = pd.factorize(save["card_id"])[0]
        save.rename(
            columns={"review_rating": "rating", "review_date": "review_th"},
            inplace=True,
        )
        save.to_csv(f"converted/{csv_file_path.stem}.csv", index=False)
    return dataset


enable_experience_replay = True
replay_steps = 64
replay_size = 8192
lr = 8e-3
batch_size = 512


def FSRS_latest_train(revlogs):
    model = FSRS(DEFAULT_PARAMETER)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss(reduction="none")

    dataset = BatchDataset(revlogs, 1, False)
    dataloader = BatchLoader(dataset, shuffle=False)
    clipper = ParameterClipper()
    d = []
    s = []
    r = []

    for i, sample in enumerate(tqdm(dataloader)):
        model.train()
        optimizer.zero_grad()
        sequence, delta_t, label, seq_len, weights = sample
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
            replay_dataset = BatchDataset(
                revlogs[max(0, i + 1 - replay_size) : i + 1],
                batch_size,
            )  # avoid data leakage
            replay_dataloader = BatchLoader(replay_dataset, seed=42 + i)
            for j, batch in enumerate(replay_dataloader):
                model.train()
                optimizer.zero_grad()
                sequences, delta_ts, labels, seq_lens, weights = batch
                real_batch_size = seq_lens.shape[0]
                outputs, _ = model(sequences)
                stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]
                retentions = power_forgetting_curve(delta_ts, stabilities)
                loss = loss_fn(retentions, labels).sum()
                loss.backward()
                optimizer.step()
                model.apply(clipper)

    revlogs["R (FSRS-5)"] = r

    return revlogs


def FSRS_old_train(revlogs):
    for model, name in ((FSRS3(), "FSRSv3"), (FSRS4(), "FSRSv4")):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss(reduction="none")

        dataset = BatchDataset(revlogs, 1, False)
        dataloader = BatchLoader(dataset, shuffle=False)
        d = []
        s = []
        r = []

        for i, sample in enumerate(tqdm(dataloader)):
            model.train()
            optimizer.zero_grad()
            sequence, delta_t, label, seq_len, weights = sample
            output, _ = model(sequence)
            stability, difficulty = output[seq_len - 1, 0].transpose(0, 1)
            d.append(difficulty.detach().numpy()[0])
            s.append(stability.detach().numpy()[0])
            retention = model.forgetting_curve(delta_t, stability)
            r.append(retention.detach().numpy()[0])
            loss = loss_fn(retention, label).sum()
            loss.backward()
            optimizer.step()
            model.apply(model.clipper)

            if enable_experience_replay and (i + 1) % replay_steps == 0:
                # experience replay
                replay_dataset = BatchDataset(
                    revlogs[max(0, i + 1 - replay_size) : i + 1], batch_size
                )  # avoid data leakage
                replay_dataloader = BatchLoader(replay_dataset, seed=42 + i)
                for j, batch in enumerate(replay_dataloader):
                    model.train()
                    optimizer.zero_grad()
                    sequences, delta_ts, labels, seq_lens, weights = batch
                    real_batch_size = seq_lens.shape[0]
                    outputs, _ = model(sequences)
                    stabilities = outputs[
                        seq_lens - 1, torch.arange(real_batch_size), 0
                    ]
                    retentions = model.forgetting_curve(delta_ts, stabilities)
                    loss = loss_fn(retentions, labels).sum()
                    loss.backward()
                    optimizer.step()
                    model.apply(model.clipper)

        revlogs[f"R ({name})"] = r

    return revlogs


def evaluate(revlogs):
    sm16_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (SM16)"]
        ].rename(columns={"R (SM16)": "p"})
    )
    sm17_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (SM17(exp))"]
        ].rename(columns={"R (SM17(exp))": "p"})
    )
    fsrs_v5_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (FSRS-5)"]
        ].rename(columns={"R (FSRS-5)": "p"})
    )
    fsrs_v4_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (FSRSv4)"]
        ].rename(columns={"R (FSRSv4)": "p"})
    )
    fsrs_v3_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (FSRSv3)"]
        ].rename(columns={"R (FSRSv3)": "p"})
    )
    sm16_logloss = log_loss(revlogs["y"], revlogs["R (SM16)"])
    sm17_logloss = log_loss(revlogs["y"], revlogs["R (SM17(exp))"])
    fsrs_v5_logloss = log_loss(revlogs["y"], revlogs["R (FSRS-5)"])
    fsrs_v4_logloss = log_loss(revlogs["y"], revlogs["R (FSRSv4)"])
    fsrs_v3_logloss = log_loss(revlogs["y"], revlogs["R (FSRSv3)"])
    return {
        "FSRS-5": {
            "RMSE(bins)": fsrs_v5_rmse,
            "LogLoss": fsrs_v5_logloss,
        },
        "FSRSv4": {
            "RMSE(bins)": fsrs_v4_rmse,
            "LogLoss": fsrs_v4_logloss,
        },
        "FSRSv3": {
            "RMSE(bins)": fsrs_v3_rmse,
            "LogLoss": fsrs_v3_logloss,
        },
        "SM16": {
            "RMSE(bins)": sm16_rmse,
            "LogLoss": sm16_logloss,
        },
        "SM17": {
            "RMSE(bins)": sm17_rmse,
            "LogLoss": sm17_logloss,
        },
    }


if __name__ == "__main__":
    Path("result").mkdir(parents=True, exist_ok=True)
    for file in Path("dataset").iterdir():
        plt.close("all")
        if file.is_file() and file.suffix == ".csv":
            if file.stem in map(lambda x: x.stem, Path("result").iterdir()):
                print(f"{file.stem} already exists, skip")
                continue
            try:
                _, user = file.stem.split("_")
            except:
                continue
            revlogs = data_preprocessing(file)
            revlogs = FSRS_old_train(revlogs)
            revlogs = FSRS_latest_train(revlogs)
            result = evaluate(revlogs)

            result["user"] = user
            result["size"] = revlogs.shape[0]
            # save as json
            Path("result").mkdir(parents=True, exist_ok=True)
            with open(f"result/{file.stem}.json", "w") as f:
                json.dump(result, f, indent=4)
