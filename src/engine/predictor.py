import pandas as pd
from torch import from_numpy
import yaml
from datetime import datetime
import numpy as np
from tqdm import tqdm


def fetch_index(given_time, seq_length, csv_path):
    min_range = pd.read_csv(csv_path, usecols=[0],
                            skiprows=lambda x: x not in [seq_length, seq_length + 1]).to_numpy().reshape(-1)[0]
    max_range = "2021-01-01 00:10:00"
    fin_index = 893646

    if given_time < min_range:
        raise ValueError
    else:
        if max_range > given_time:
            timestamps = pd.read_csv(csv_path, usecols=["time"])
            index = timestamps[timestamps["time"] == given_time].index[0]
        else:
            fmt = '%Y-%m-%d %H:%M:%S'
            t1 = datetime.strptime(max_range, fmt)
            t2 = datetime.strptime(given_time, fmt)
            diff = (t2 - t1).total_seconds() // 600
            index = int(fin_index + diff)

        return index


def load_mmscaler_values():
    with open("config/mmscaler_values.yaml", "r") as f:
        scaler_values = yaml.safe_load(f)
        f.close()
    return scaler_values


def predict_single_record(model, record_arr, device="cpu"):
    record_arr = from_numpy(record_arr).unsqueeze(0)
    op = model(record_arr, device).squeeze(0)
    if op.device != "cpu":
        op = op.cpu()
    op = op.detach().numpy().reshape(-1)
    return op.tolist()


def anti_transform(op_arr):
    scaler_dict = load_mmscaler_values()
    for index in range(len(op_arr)):
        op_arr[index] = (op_arr[index] * (scaler_dict["max_values"][index]
                                          - scaler_dict["min_values"][index])) + scaler_dict["min_values"][index]
    ans = [round(el, 3) for el in op_arr.tolist()]
    return ans


def predict_results(csv_path, index, model, seq_len):
    if index < 893646:
        df = pd.read_csv(csv_path, skiprows=lambda x: x not in [index+1], header=None)
        df = df.to_numpy()[:, 1:].reshape(-1)
        ans = anti_transform(df)
    else:
        num_repeats = index - 893646
        data = pd.read_csv(csv_path,
                           skiprows=lambda x: x not in [i for i in range(893646 - seq_len, 893646)], header=None)
        data = data.to_numpy()[:, 1:]
        temp_data = data.astype("float32")
        print(f"Predicting Future for {num_repeats} iteration")
        for index in tqdm(range(num_repeats)):
            op = predict_single_record(model, temp_data)
            temp_data = temp_data[1:, :].tolist()
            temp_data.append(op)
            temp_data = np.array(temp_data).astype("float32")

        ans = anti_transform(temp_data[-1, :])

    return ans
