from typing import List
import pandas as pd
from pandas._libs.lib import no_default
from pandas import DataFrame
from .utils import get_number_if_possible


def all_preprocess_df(df: DataFrame, preprocessing_step_list: List[dict]):
    for preprocessing_step in preprocessing_step_list:
        df = preprocess_df(df, preprocessing_step)
    return df


def preprocess_df(df: DataFrame, preprocessing_step: dict):
    params = {
        k: v
        for k, v in preprocessing_step.items()
        if v is not None and k != "option_type"
    }
    processing_function = None
    if preprocessing_step["option_type"] == "drop na":
        processing_function = drop_missing_value
    elif preprocessing_step["option_type"] == "fill na":
        processing_function = fill_missing_value
    if processing_function is None:
        return df
    return processing_function(df, **params)


def drop_missing_value(
    df: DataFrame, axis: int = 0, threshold: int = no_default, subset: str = None
) -> DataFrame:
    if subset is not None:
        subset = None if subset == "" else subset.split(",")

    return df.dropna(axis=axis, thresh=threshold, subset=subset)


def fill_missing_value(df: DataFrame, fill_value: str) -> DataFrame:
    fill_value = get_number_if_possible(fill_value)

    return df.fillna(value=fill_value)
