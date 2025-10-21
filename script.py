import collections
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
from models import FSRS3, FSRS4, FSRS4dot5, FSRS5
from tqdm.auto import tqdm
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
from pathlib import Path
from utils import get_bin

tqdm.pandas()


def compute_adversarial_predictions(revlogs, algorithms, bins=10):
    """Craft predictions that exploit referee bins by matching observed outcomes.

    The attack groups each record by the tuple of bin assignments induced by the
    provided algorithms and sets its prediction to the average success rate of
    that tuple. This guarantees that, for every referee, the adversary is
    perfectly calibrated at the bin level while still using the same prediction
    column for binning and scoring.
    """

    if "R (ADVERSARIAL)" in revlogs.columns:
        return revlogs

    # Ensure we operate on probability columns
    base_cols = [f"R ({algo})" for algo in algorithms]
    missing_cols = [col for col in base_cols if col not in revlogs.columns]
    if missing_cols:
        raise KeyError(
            "Missing prediction columns required for adversarial computation: "
            + ", ".join(missing_cols)
        )

    # Bin identifiers mirror the cross-comparison binning scheme
    bin_cols = []
    for algo in algorithms:
        col_name = f"_adv_bin_{algo}"
        predictions = revlogs[f"R ({algo})"].clip(0, 1)
        revlogs[col_name] = get_bin(predictions, bins)
        bin_cols.append(col_name)

    revlogs["R (ADVERSARIAL)"] = (
        revlogs.groupby(bin_cols)["y"].transform("mean").astype(float)
    )

    revlogs.drop(columns=bin_cols, inplace=True)
    return revlogs


def data_preprocessing(csv_file_path, save_csv=False):
    try:
        df = pd.read_csv(csv_file_path, encoding="utf-8")
    except:
        df = pd.read_csv(csv_file_path, encoding="gbk")
    df.columns = df.columns.str.strip()

    def convert_to_datetime(date_str):
        german_months = {
            "Jan": "Jan",
            "Feb": "Feb",
            "Mär": "Mar",
            "Mrz": "Mar",
            "Apr": "Apr",
            "Mai": "May",
            "Jun": "Jun",
            "Jul": "Jul",
            "Aug": "Aug",
            "Sep": "Sep",
            "Okt": "Oct",
            "Nov": "Nov",
            "Dez": "Dec",
        }

        portuguese_months = {
            "jan": "Jan",
            "fev": "Feb",
            "mar": "Mar",
            "abr": "Apr",
            "mai": "May",
            "jun": "Jun",
            "jul": "Jul",
            "ago": "Aug",
            "set": "Sep",
            "out": "Oct",
            "nov": "Nov",
            "dez": "Dec",
        }

        date_str_normalized = date_str

        date_str_normalized = (
            date_str_normalized.replace("jan.", "jan")
            .replace("feb.", "feb")
            .replace("mar.", "mar")
        )
        date_str_normalized = (
            date_str_normalized.replace("apr.", "apr")
            .replace("may.", "may")
            .replace("jun.", "jun")
        )
        date_str_normalized = (
            date_str_normalized.replace("jul.", "jul")
            .replace("aug.", "aug")
            .replace("sep.", "sep")
        )
        date_str_normalized = (
            date_str_normalized.replace("oct.", "oct")
            .replace("nov.", "nov")
            .replace("dec.", "dec")
        )

        month_abbrs = [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
        for month in month_abbrs:
            if month in date_str_normalized.lower():
                date_str_normalized = date_str_normalized.replace(
                    month, month.capitalize()
                )
                break

        for de_month, en_month in german_months.items():
            if de_month in date_str_normalized:
                date_str_normalized = date_str_normalized.replace(de_month, en_month)
                break

        for pt_month, en_month in portuguese_months.items():
            if pt_month in date_str_normalized.lower():
                date_str_normalized = date_str_normalized.lower().replace(
                    pt_month, en_month
                )
                break

        date_formats = [
            "%b %d %Y %H:%M:%S",
            "%m %d %Y %H:%M:%S",
            "%m月 %d %Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y",
            "%m-%d-%y",
        ]

        for date_format in date_formats:
            try:
                return datetime.strptime(date_str_normalized, date_format)
            except ValueError:
                continue
        print(f"Failed to convert {date_str} to datetime")
        return pd.NaT

    df["Date"] = df["Date"].apply(convert_to_datetime)
    df.dropna(subset=["Date"], inplace=True)
    df = df[df["Success"].isin([0, 1]) & df["Grade"].isin([0, 1, 2, 3, 4, 5])].copy()
    df = df[(df["R (SM16)"] <= 1) & (df["R (SM17)"] <= 1)].copy()
    dataset = df[
        [
            "Date",
            "Element No",
            "Used interval",
            "R (SM16)",
            "R (SM17)",
            "Grade",
            "Success",
        ]
    ].sort_values(by=["Element No", "Date"])
    dataset.rename(
        columns={
            "Element No": "card_id",
            "Date": "review_date",
            "Used interval": "delta_t",
            "Success": "y",
        },
        inplace=True,
    )
    dataset = dataset[dataset["delta_t"] > 0].copy()
    dataset.reset_index(drop=True, inplace=True)
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

    def get_tensor(row):
        if row["i"] > 1:
            return lineToTensor(list(zip([row["t_history"]], [row["r_history"]]))[0])
        else:
            return np.nan

    dataset["tensor"] = dataset.progress_apply(get_tensor, axis=1)
    dataset.sort_values(by=["review_date", "i"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    dataset["index"] = range(0, dataset.shape[0])
    dataset[["next_index", "next_tensor", "next_delta_t", "next_y"]] = dataset.groupby(
        "card_id"
    )[["index", "tensor", "delta_t", "y"]].shift(-1)
    if save_csv:
        Path("converted").mkdir(parents=True, exist_ok=True)
        save = dataset[
            [
                "card_id",
                "review_date",
                "delta_t",
                "review_rating",
                "R (SM16)",
                "R (SM17)",
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
    clipper = ParameterClipper()
    r = [np.nan for _ in range(len(revlogs))]
    queue = collections.deque(maxlen=replay_size)

    for i in tqdm(range(len(revlogs))):
        row = revlogs.iloc[i]
        if row["i"] > 1:
            queue.append(row.to_dict())

        if not np.isnan(row["next_index"]):
            # predict the retention of the next review of the same card
            sequence = (
                torch.tensor(row["next_tensor"].tolist()).unsqueeze(0).transpose(0, 1)
            )
            seq_len = torch.tensor(row["next_tensor"].size(0), dtype=torch.long)
            delta_t = torch.tensor(row["next_delta_t"], dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                output, _ = model(sequence)
            stability = output[seq_len - 1, torch.arange(1), 0]
            retention = power_forgetting_curve(delta_t, stability, -model.w[20])
            r[int(row["next_index"])] = retention.detach().numpy().round(3)[0]

        if enable_experience_replay and len(queue) > 0 and (i + 1) % replay_steps == 0:
            # experience replay
            replay_buffer = pd.DataFrame(queue, columns=revlogs.columns)
            x = np.linspace(0, 1, len(replay_buffer))
            replay_buffer["weights"] = 0.25 + 0.75 * np.power(x, 3)
            replay_dataset = BatchDataset(
                replay_buffer,
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
                retentions = power_forgetting_curve(delta_ts, stabilities, -model.w[20])
                loss = (loss_fn(retentions, labels) * weights).sum()
                loss.backward()
                optimizer.step()
                model.apply(clipper)

    revlogs["R (FSRS-6)"] = r

    return revlogs, model


def FSRS_old_train(revlogs):
    trained_models = {}
    for model, name in (
        (FSRS3(), "FSRSv3"),
        (FSRS4(), "FSRSv4"),
        (FSRS4dot5(), "FSRS-4.5"),
        (FSRS5(), "FSRS-5"),
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss(reduction="none")
        r = [np.nan for _ in range(len(revlogs))]
        queue = collections.deque(maxlen=replay_size)

        for i in tqdm(range(len(revlogs))):
            row = revlogs.iloc[i]
            if row["i"] > 1:
                queue.append(row.to_dict())

            if not np.isnan(row["next_index"]):
                # predict the retention of the next review of the same card
                sequence = (
                    torch.tensor(row["next_tensor"].tolist())
                    .unsqueeze(0)
                    .transpose(0, 1)
                )
                seq_len = torch.tensor(row["next_tensor"].size(0), dtype=torch.long)
                delta_t = torch.tensor(
                    row["next_delta_t"], dtype=torch.float
                ).unsqueeze(0)
                with torch.no_grad():
                    output, _ = model(sequence)
                stability = output[seq_len - 1, torch.arange(1), 0]
                retention = model.forgetting_curve(delta_t, stability)
                r[int(row["next_index"])] = retention.detach().numpy().round(3)[0]

            if (
                enable_experience_replay
                and len(queue) > 0
                and (i + 1) % replay_steps == 0
            ):
                # experience replay
                replay_buffer = pd.DataFrame(queue, columns=revlogs.columns)
                x = np.linspace(0, 1, len(replay_buffer))
                replay_buffer["weights"] = 0.25 + 0.75 * np.power(x, 3)
                replay_dataset = BatchDataset(
                    replay_buffer,
                    batch_size,
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
                    loss = (loss_fn(retentions, labels) * weights).sum()
                    loss.backward()
                    optimizer.step()
                    model.apply(model.clipper)

        revlogs[f"R ({name})"] = r
        trained_models[name] = model

    return revlogs, trained_models


def average(revlogs):
    tot_y = 0.9
    tot_n = 1
    predictions = [np.nan for _ in range(len(revlogs))]
    for i in range(len(revlogs)):
        row = revlogs.iloc[i]
        assert i == row["index"]
        if not np.isnan(row["next_index"]):
            predictions[int(row["next_index"])] = tot_y / tot_n
        tot_y += row["y"]
        tot_n += 1.0

    revlogs["R (AVG)"] = predictions
    return revlogs


def FSRS6_default(revlogs):
    model = FSRS(DEFAULT_PARAMETER)
    predictions = [np.nan for _ in range(len(revlogs))]
    with torch.no_grad():
        for i in tqdm(range(len(revlogs))):
            row = revlogs.iloc[i]
            assert i == row["index"]
            if not np.isnan(row["next_index"]):
                sequence = (
                    torch.tensor(row["next_tensor"].tolist())
                    .unsqueeze(0)
                    .transpose(0, 1)
                )
                seq_len = torch.tensor(row["next_tensor"].size(0), dtype=torch.long)
                output, _ = model(sequence)
                stability = output[seq_len - 1, torch.arange(1), 0]
                retention = power_forgetting_curve(
                    row["next_delta_t"], stability, -model.w[20]
                )
                predictions[int(row["next_index"])] = (
                    retention.detach().numpy().round(3)[0]
                )
    revlogs["R (FSRS-6-default)"] = predictions
    return revlogs


def moving_average(revlogs):
    x = 1.2
    w = 0.3
    predictions = [np.nan for _ in range(len(revlogs))]
    for i in range(len(revlogs)):
        row = revlogs.iloc[i]
        assert i == row["index"]
        y_pred = 1 / (np.exp(-x) + 1)
        predictions[i] = y_pred
        # gradient step
        if row["y"] == 1:
            x += w / (np.exp(x) + 1)
        else:
            x -= w * (np.exp(x)) / (np.exp(x) + 1)
    revlogs["R (MOVING-AVG)"] = predictions
    return revlogs


def evaluate(revlogs):
    # Define binning function
    # Calculate Universal Metrics for each algorithm pair
    def calculate_universal_metric(algoA, algoB):
        cross_comparison_record = revlogs[[f"R ({algoA})", f"R ({algoB})", "y"]].copy()

        for algo in (algoA, algoB):
            cross_comparison_record[f"{algo}_B-W"] = (
                cross_comparison_record[f"R ({algo})"] - cross_comparison_record["y"]
            )
            cross_comparison_record[f"{algo}_bin"] = cross_comparison_record[
                f"R ({algo})"
            ].map(get_bin)

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

        return result

    # Calculate all Universal Metrics
    universal_metrics = {}
    base_algorithms = [
        "FSRS-6",
        "FSRS-5",
        "FSRS-4.5",
        "FSRSv4",
        "FSRSv3",
        "SM16",
        "SM17",
        "AVG",
        "FSRS-6-default",
        "MOVING-AVG",
    ]

    compute_adversarial_predictions(revlogs, base_algorithms)

    algorithms = base_algorithms + ["ADVERSARIAL"]

    for i, algoA in enumerate(algorithms):
        for algoB in algorithms[i + 1 :]:
            um_result = calculate_universal_metric(algoA, algoB)
            universal_metrics.update(um_result)

    # Original metrics calculation
    avg_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (AVG)"]
        ].rename(columns={"R (AVG)": "p"})
    )
    moving_avg_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (MOVING-AVG)"]
        ].rename(columns={"R (MOVING-AVG)": "p"})
    )
    adversarial_rmse = rmse_matrix(
        revlogs[
            [
                "card_id",
                "r_history",
                "t_history",
                "delta_t",
                "i",
                "y",
                "R (ADVERSARIAL)",
            ]
        ].rename(columns={"R (ADVERSARIAL)": "p"})
    )
    sm16_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (SM16)"]
        ].rename(columns={"R (SM16)": "p"})
    )
    sm17_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (SM17)"]
        ].rename(columns={"R (SM17)": "p"})
    )
    fsrs_v6_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (FSRS-6)"]
        ].rename(columns={"R (FSRS-6)": "p"})
    )
    fsrs_v5_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (FSRS-5)"]
        ].rename(columns={"R (FSRS-5)": "p"})
    )
    fsrs_v4dot5_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (FSRS-4.5)"]
        ].rename(columns={"R (FSRS-4.5)": "p"})
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
    fsrs_v6_default_rmse = rmse_matrix(
        revlogs[
            [
                "card_id",
                "r_history",
                "t_history",
                "delta_t",
                "i",
                "y",
                "R (FSRS-6-default)",
            ]
        ].rename(columns={"R (FSRS-6-default)": "p"})
    )
    avg_logloss = log_loss(revlogs["y"], revlogs["R (AVG)"])
    moving_avg_logloss = log_loss(revlogs["y"], revlogs["R (MOVING-AVG)"])
    adversarial_logloss = log_loss(revlogs["y"], revlogs["R (ADVERSARIAL)"])
    sm16_logloss = log_loss(revlogs["y"], revlogs["R (SM16)"])
    sm17_logloss = log_loss(revlogs["y"], revlogs["R (SM17)"])
    fsrs_v6_logloss = log_loss(revlogs["y"], revlogs["R (FSRS-6)"])
    fsrs_v5_logloss = log_loss(revlogs["y"], revlogs["R (FSRS-5)"])
    fsrs_v4dot5_logloss = log_loss(revlogs["y"], revlogs["R (FSRS-4.5)"])
    fsrs_v4_logloss = log_loss(revlogs["y"], revlogs["R (FSRSv4)"])
    fsrs_v3_logloss = log_loss(revlogs["y"], revlogs["R (FSRSv3)"])
    fsrs_v6_default_logloss = log_loss(revlogs["y"], revlogs["R (FSRS-6-default)"])

    avg_auc = roc_auc_score(revlogs["y"], revlogs["R (AVG)"])
    moving_avg_auc = roc_auc_score(revlogs["y"], revlogs["R (MOVING-AVG)"])
    adversarial_auc = roc_auc_score(revlogs["y"], revlogs["R (ADVERSARIAL)"])
    sm16_auc = roc_auc_score(revlogs["y"], revlogs["R (SM16)"])
    sm17_auc = roc_auc_score(revlogs["y"], revlogs["R (SM17)"])
    fsrs_v6_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRS-6)"])
    fsrs_v5_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRS-5)"])
    fsrs_v4dot5_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRS-4.5)"])
    fsrs_v4_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRSv4)"])
    fsrs_v3_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRSv3)"])
    fsrs_v6_default_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRS-6-default)"])

    result = {
        "FSRS-6": {
            "RMSE(bins)": round(fsrs_v6_rmse, 4),
            "LogLoss": round(fsrs_v6_logloss, 4),
            "AUC": round(fsrs_v6_auc, 4),
        },
        "FSRS-5": {
            "RMSE(bins)": round(fsrs_v5_rmse, 4),
            "LogLoss": round(fsrs_v5_logloss, 4),
            "AUC": round(fsrs_v5_auc, 4),
        },
        "FSRS-4.5": {
            "RMSE(bins)": round(fsrs_v4dot5_rmse, 4),
            "LogLoss": round(fsrs_v4dot5_logloss, 4),
            "AUC": round(fsrs_v4dot5_auc, 4),
        },
        "FSRSv4": {
            "RMSE(bins)": round(fsrs_v4_rmse, 4),
            "LogLoss": round(fsrs_v4_logloss, 4),
            "AUC": round(fsrs_v4_auc, 4),
        },
        "FSRSv3": {
            "RMSE(bins)": round(fsrs_v3_rmse, 4),
            "LogLoss": round(fsrs_v3_logloss, 4),
            "AUC": round(fsrs_v3_auc, 4),
        },
        "SM16": {
            "RMSE(bins)": round(sm16_rmse, 4),
            "LogLoss": round(sm16_logloss, 4),
            "AUC": round(sm16_auc, 4),
        },
        "SM17": {
            "RMSE(bins)": round(sm17_rmse, 4),
            "LogLoss": round(sm17_logloss, 4),
            "AUC": round(sm17_auc, 4),
        },
        "AVG": {
            "RMSE(bins)": round(avg_rmse, 4),
            "LogLoss": round(avg_logloss, 4),
            "AUC": round(avg_auc, 4),
        },
        "MOVING-AVG": {
            "RMSE(bins)": round(moving_avg_rmse, 4),
            "LogLoss": round(moving_avg_logloss, 4),
            "AUC": round(moving_avg_auc, 4),
        },
        "ADVERSARIAL": {
            "RMSE(bins)": round(adversarial_rmse, 4),
            "LogLoss": round(adversarial_logloss, 4),
            "AUC": round(adversarial_auc, 4),
        },
        "FSRS-6-default": {
            "RMSE(bins)": round(fsrs_v6_default_rmse, 4),
            "LogLoss": round(fsrs_v6_default_logloss, 4),
            "AUC": round(fsrs_v6_default_auc, 4),
        },
    }

    # Add Universal Metrics to result
    result["Universal_Metrics"] = universal_metrics

    return result


def process_single_file(file):
    try:
        if not file.is_file() or file.suffix != ".csv":
            return None

        if file.stem in map(lambda x: x.stem, Path("result").iterdir()):
            print(f"{file.stem} already exists, skip")
            return None

        try:
            _, user = file.stem.split("_")
        except:
            return None

        plt.close("all")
        revlogs = data_preprocessing(file)
        revlogs = average(revlogs)
        revlogs = moving_average(revlogs)
        revlogs = FSRS6_default(revlogs)
        revlogs, trained_models = FSRS_old_train(revlogs)
        revlogs, model = FSRS_latest_train(revlogs)

        # exclude the first learning entry for each card
        revlogs = revlogs[revlogs["i"] > 1].copy()
        revlogs.reset_index(drop=True, inplace=True)
        result = evaluate(revlogs)

        revlogs[["y"] + [col for col in revlogs.columns if col.startswith("R")]].to_csv(
            f"./raw/{file.stem}.csv", index=False
        )

        result["user"] = user
        result["size"] = revlogs.shape[0]

        # Add model parameters to result
        result["parameters"] = {}

        # Add FSRS-6 parameters
        result["parameters"]["FSRS-6"] = list(
            map(lambda x: round(x, 4), model.w.detach().numpy().tolist())
        )

        # Add other FSRS model parameters
        for model_name, trained_model in trained_models.items():
            if hasattr(trained_model, "w"):
                result["parameters"][model_name] = list(
                    map(
                        lambda x: round(x, 4), trained_model.w.detach().numpy().tolist()
                    )
                )

        # save as json
        with open(f"result/{file.stem}.json", "w") as f:
            json.dump(result, f, indent=4)

        return file.stem

    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return None


if __name__ == "__main__":
    import multiprocessing as mp
    from multiprocessing import Pool

    # Create result directory
    Path("result").mkdir(parents=True, exist_ok=True)

    # Get list of files to process
    files = list(Path("dataset").glob("*.csv"))

    # Create process pool
    with Pool() as pool:
        # Process files in parallel
        results = list(
            tqdm(
                pool.imap_unordered(process_single_file, files),
                total=len(files),
                desc="Processing files",
            )
        )

        # Print summary
        processed = [r for r in results if r is not None]
        print(f"\nProcessed {len(processed)} files successfully")
