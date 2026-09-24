import pandas as pd
import os
import json
from src.backend.config import DevConfig
from src.backend.data.clean import clean_timeseries

__all__ = ["save_df_as_csv", "save_dict_as_df", "save_dict_as_json"]


def save_df_as_csv(config: DevConfig, data: pd.DataFrame, filename: str):
    if not os.path.exists(f"{config.cache_dir}/{filename}"):
        data.to_csv(f"{config.cache_dir}/{filename}")
    else:
        print("File already exists.")


def save_dict_as_df(config: DevConfig, data_dict: dict, filename: str):
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    cleaned_df = clean_timeseries(df)
    cleaned_df.to_csv(f"{config.cache_dir}/{filename}")


def save_dict_as_json(config: DevConfig, data_dict: dict, filename: str):
    with open(f"{config.cache_dir}/{filename}", "w") as file:
        json.dump(data_dict, file)
