from typing import List
from pandas._libs.lib import no_default
from pandas import DataFrame
from .utils import get_number_if_possible


def all_preprocess_df(df: DataFrame, preprocessing_step_list: List[dict]):
    for preprocessing_step in preprocessing_step_list:
        df = preprocess_df(df, preprocessing_step)
    return df


def preprocess_df(df: DataFrame, preprocessing_step: dict):
    # Remove entries with value == None
    params = {
        k: v for k, v in preprocessing_step["input_params"].items() if v is not None
    }
    processing_function = None
    option_type = preprocessing_step["option_type"]
    if option_type == "drop na":
        processing_function = drop_missing_value
    elif option_type == "fill na":
        processing_function = fill_missing_value
    elif option_type == "drop duplicates":
        processing_function = drop_duplicates
    elif option_type == "sort values":
        processing_function = sort_values
    elif option_type == "strip entries":
        processing_function = strip_characters
    elif option_type == "split column":
        processing_function = split_column
    elif option_type == "replace":
        processing_function = replace
    elif option_type == "string case":
        processing_function = string_case
    if processing_function is None:
        return df
    return processing_function(df, **params)


def drop_missing_value(
    df: DataFrame, axis: int = 0, threshold: int = no_default, subset: str = None
) -> DataFrame:
    if subset is not None:
        subset = None if subset == "" else subset.split(",")

    df.dropna(axis=axis, thresh=threshold, subset=subset, inplace=True)

    return df


def fill_missing_value(df: DataFrame, fill_value: str) -> DataFrame:
    fill_value = get_number_if_possible(fill_value)

    df.fillna(value=fill_value, inplace=True)

    return df


def drop_duplicates(
    df: DataFrame, subset: str = None, keep: str = "first", ignore_index: bool = False
) -> DataFrame:
    if subset is not None:
        subset = None if subset == "" else subset.split(",")

    keep = False if keep == "none" else keep

    df.drop_duplicates(subset=subset, keep=keep, ignore_index=ignore_index, inplace=True)

    return df


def sort_values(df: DataFrame, by: str, ascending: bool = False):
    df.sort_values(by=by, ascending=ascending, inplace=True)
    return df


def strip_characters(
    df: DataFrame, characters: List[str] = "", subset: str = None, position: str = "both"
) -> DataFrame:
    if subset is not None:
        subset = df.keys() if subset == "" else subset.split(",")
    else:
        subset = df.keys()

    function = None
    if position == "front":
        function = lambda s, c: s.lstrip(c)
    elif position == "end":
        function = lambda s, c: s.rstrip(c)
    elif position == "both":
        function = lambda s, c: s.strip(c)

    for k in subset:
        if k in df:
            if {type(el) for el in df[k]} == {str}:
                df[k] = function(df[k].str, characters)

    return df


def split_column(
    df: DataFrame,
    column: str,
    by: str,
    new_columns: str = "",
    remove_column: bool = False,
) -> DataFrame:
    if column not in df.keys():
        raise ValueError(f"The dataframe has no column {column}.")

    new_columns = new_columns.split(",")
    for n_c in new_columns:
        if n_c in df.keys():
            raise ValueError(
                f"Columns {new_columns} already exist in the given dataframe."
            )

    df[new_columns] = df[column].str.split(by, expand=True)
    if remove_column:
        df.drop(column, axis=1, inplace=True)
    return df


def replace(df: DataFrame, sub_str: str, new_str: str, subset: str = "") -> DataFrame:
    if subset is not None:
        subset = df.keys() if subset == "" else subset.split(",")
    else:
        subset = df.keys()

    for k in subset:
        if k in df:
            if {type(el) for el in df[k]} == {str}:
                df[k] = df[k].str.replace(sub_str, new_str)

    return df


def string_case(df: DataFrame, case: str, subset: str = "") -> DataFrame:
    if subset is not None:
        subset = df.keys() if subset == "" else subset.split(",")
    else:
        subset = df.keys()

    function = None
    if case == "upper":
        function = lambda s: s.upper()
    elif case == "lower":
        function = lambda s: s.lower()
    elif case == "title":
        function = lambda s: s.title()

    for k in subset:
        if k in df:
            if {type(el) for el in df[k]} == {str}:
                df[k] = function(df[k].str)

    return df
