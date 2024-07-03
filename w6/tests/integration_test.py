import pandas as pd

from datetime import datetime

from batch_v2 import prepare_data, process_data, prepare_result_df, save_df_to_s3


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_s3_connection():
    year = 2023
    month = 1

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    categorical = ["PULocationID", "DOLocationID"]

    df = pd.DataFrame(data, columns=columns)
    df = prepare_data(df, categorical)
    df, y_pred = process_data(df, categorical, year, month)
    df_result = prepare_result_df(df, y_pred)
    filename = f"{year}-{month}-taxi-data-analysis"
    save_df_to_s3(df_result, filename)
