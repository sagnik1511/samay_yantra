import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import yaml


def fetch_imp_features(dataframe, job_config):
    fet_cols = job_config["imp_features"]
    renamed_cols = job_config["renamed_cols"]
    assert len(fet_cols) == len(renamed_cols), "Size mismatch, can not rename columns."
    df = dataframe.copy()
    df = df[fet_cols]
    df.columns = renamed_cols

    return df


def process_outliers_to_nan(dataframe):
    df = dataframe.copy()
    df.replace(-9999.0, np.nan, inplace=True)
    df.replace(-9999.990, np.nan, inplace=True)
    df["wind_speed"].replace(28.4900, np.nan, inplace=True)

    return df


def perform_interpolation(dataframe, job_config):

    method = job_config["interpolation_method"]
    df = dataframe.copy()
    for fet in df.columns[1:]:
        df[fet] = df[fet].interpolate(method=method)

    return df


def perform_scaling(dataframe):
    df = dataframe.copy()
    mm_scalers = [MinMaxScaler() for _ in range(len(df.columns) - 1)]
    mm_scaler_values = {
        "min_values": [],
        "max_values": []
    }
    for index, fet in enumerate(df.columns[1:]):
        df[fet] = mm_scalers[index].fit_transform(df[fet].to_numpy().reshape(-1, 1)).reshape(-1)
        mm_scaler_values["min_values"].append(mm_scalers[index].data_min_.tolist()[0])
        mm_scaler_values["max_values"].append(mm_scalers[index].data_max_.tolist()[0])

    return df, mm_scaler_values


def prepare_dataset(job_config):

    ip_path = job_config["data_path"]["input"]
    op_path = job_config["data_path"]["output"]
    mmscaler_conf_path = job_config["data_path"]["mmscaler_values"]
    assert os.path.isfile(ip_path), "Source data not found ..."
    dataframe = pd.read_csv(ip_path, parse_dates=["Date Time"])
    print("Data Loaded ...")
    df = dataframe.copy()
    df = fetch_imp_features(dataframe=df, job_config=job_config)
    print("Important features filtered ...")
    df = process_outliers_to_nan(dataframe=df)
    df.drop_duplicates("time", inplace=True)
    df = perform_interpolation(dataframe=df, job_config=job_config)
    print("Performed interpolation ...")
    df, scaler_values = perform_scaling(dataframe=df)
    print('Performed Scaling ...')
    with open(mmscaler_conf_path, "w") as f:
        yaml.dump(scaler_values, f)
        f.close()
    print("Scaler values stored ...")
    df.to_csv(op_path, index=False)
    print(f"Processed data stored to {op_path}")


def main():
    job_config_path = os.path.join(os.getcwd(), "config", "prepare_data.json")
    with open(job_config_path, "r") as f:
        job_config = json.load(f)
        f.close()
    print(job_config)
    prepare_dataset(job_config=job_config)


if __name__ == "__main__":
    main()
