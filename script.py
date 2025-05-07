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
from sklearn.metrics import log_loss, roc_auc_score
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

    return revlogs


def FSRS_old_train(revlogs):
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

    return revlogs


def baseline(revlogs):
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


def evaluate(revlogs):
    avg_rmse = rmse_matrix(
        revlogs[
            ["card_id", "r_history", "t_history", "delta_t", "i", "y", "R (AVG)"]
        ].rename(columns={"R (AVG)": "p"})
    )
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
    avg_logloss = log_loss(revlogs["y"], revlogs["R (AVG)"])
    sm16_logloss = log_loss(revlogs["y"], revlogs["R (SM16)"])
    sm17_logloss = log_loss(revlogs["y"], revlogs["R (SM17(exp))"])
    fsrs_v6_logloss = log_loss(revlogs["y"], revlogs["R (FSRS-6)"])
    fsrs_v5_logloss = log_loss(revlogs["y"], revlogs["R (FSRS-5)"])
    fsrs_v4dot5_logloss = log_loss(revlogs["y"], revlogs["R (FSRS-4.5)"])
    fsrs_v4_logloss = log_loss(revlogs["y"], revlogs["R (FSRSv4)"])
    fsrs_v3_logloss = log_loss(revlogs["y"], revlogs["R (FSRSv3)"])
    avg_auc = roc_auc_score(revlogs["y"], revlogs["R (AVG)"])
    sm16_auc = roc_auc_score(revlogs["y"], revlogs["R (SM16)"])
    sm17_auc = roc_auc_score(revlogs["y"], revlogs["R (SM17(exp))"])
    fsrs_v6_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRS-6)"])
    fsrs_v5_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRS-5)"])
    fsrs_v4dot5_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRS-4.5)"])
    fsrs_v4_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRSv4)"])
    fsrs_v3_auc = roc_auc_score(revlogs["y"], revlogs["R (FSRSv3)"])

    return {
        "FSRS-6": {
            "RMSE(bins)": fsrs_v6_rmse,
            "LogLoss": fsrs_v6_logloss,
            "AUC": fsrs_v6_auc,
        },
        "FSRS-5": {
            "RMSE(bins)": fsrs_v5_rmse,
            "LogLoss": fsrs_v5_logloss,
            "AUC": fsrs_v5_auc,
        },
        "FSRS-4.5": {
            "RMSE(bins)": fsrs_v4dot5_rmse,
            "LogLoss": fsrs_v4dot5_logloss,
            "AUC": fsrs_v4dot5_auc,
        },
        "FSRSv4": {
            "RMSE(bins)": fsrs_v4_rmse,
            "LogLoss": fsrs_v4_logloss,
            "AUC": fsrs_v4_auc,
        },
        "FSRSv3": {
            "RMSE(bins)": fsrs_v3_rmse,
            "LogLoss": fsrs_v3_logloss,
            "AUC": fsrs_v3_auc,
        },
        "SM16": {
            "RMSE(bins)": sm16_rmse,
            "LogLoss": sm16_logloss,
            "AUC": sm16_auc,
        },
        "SM17": {
            "RMSE(bins)": sm17_rmse,
            "LogLoss": sm17_logloss,
            "AUC": sm17_auc,
        },
        "AVG": {
            "RMSE(bins)": avg_rmse,
            "LogLoss": avg_logloss,
            "AUC": avg_auc,
        },
    }


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
        revlogs = baseline(revlogs)
        revlogs = FSRS_old_train(revlogs)
        revlogs = FSRS_latest_train(revlogs)

        # exclude the first learning entry for each card
        revlogs = revlogs[revlogs["i"] > 1].copy()
        revlogs.reset_index(drop=True, inplace=True)
        result = evaluate(revlogs)

        revlogs[["y"] + [col for col in revlogs.columns if col.startswith("R")]].to_csv(
            f"./raw/{file.stem}.csv", index=False
        )

        result["user"] = user
        result["size"] = revlogs.shape[0]

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
    files = list(Path("dataset").iterdir())

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
