#!/usr/bin/env python
# coding: utf-8
import boto3
import os
import pandas as pd
import pickle
import sys


def get_input_path(year, month):
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = "s3://nyc-duration-prediction/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename):
    ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
    if ENDPOINT_URL:
        options = {"client_kwargs": {"endpoint_url": ENDPOINT_URL}}
        df = pd.read_parquet("s3://bucket/file.parquet", storage_options=options)
    else:
        df = pd.read_parquet(filename)

    return df


def prepare_data(df, categorical):
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


def process_data(df, categorical, year, month):
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    return df, y_pred


def prepare_result_df(df, y_pred):
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred
    return df_result


def save_df_to_s3(
    df_result,
    filename,
    endpoint_url="http://localhost:4566",
    bucket_name="nyc-duration",
):
    options = {"client_kwargs": {"endpoint_url": endpoint_url}}
    s3_client = boto3.client("s3", endpoint_url=endpoint_url)
    s3_client.create_bucket(Bucket=bucket_name)

    df_result.to_parquet(
        f"s3://nyc-duration/{filename}",
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options,
    )


def main(year, month):
    # year = int(sys.argv[1])
    # month = int(sys.argv[2])

    year = 2023
    month = 1

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    categorical = ["PULocationID", "DOLocationID"]

    df = read_data(input_file)
    df = prepare_data(df, categorical)
    df, y_pred = process_data(df, categorical, year, month)

    print("predicted mean duration:", y_pred.mean())

    df_result = prepare_result_df(df, y_pred)
    filename = f"{year}-{month}-taxi-data-analysis"
    save_df_to_s3(df_result, filename)


if __name__ == "__main__":
    main(2023, 3)
