import pandas as pd


def clean_timeseries(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d.rename(columns={col: col.lower() for col in d.columns}, inplace=True)
    if type(d.index) is not pd.DatetimeIndex:
        d.reset_index(inplace=True, names=["Date"])
        d["Date"] = pd.to_datetime(d["Date"], format="%Y%m%d")
        d.set_index("Date", inplace=True)
        if "return" in d.columns:
            d = d.dropna(subset=["return"])
            d = d["datetime,open,close,high,low,volume,return".split(",")]
        else:
            d = d["datetime,open,close,high,low,volume".split(",")]
        return d
    else:
        d = d.dropna()
        return d


def to_panel(frames: dict, field: str) -> pd.DataFrame:
    """
    Retrieves a dictionary of dataframes keyed by their individual tickers and returns a
    panel containing all ticker data for the specified field.

    Parameters
    ----------
    frames: dict
        A dictionary containing ticker : price_data key-value pairs
    field: str
        The data column to extract from each individual dataframe

    Returns
    -------
    pd.DataFrame
        A dataframe containing the 'field' column from all dataframes in 'frames'
    """
    return pd.concat({ticker: frame[field] for ticker, frame in frames.items()}, axis=1)
