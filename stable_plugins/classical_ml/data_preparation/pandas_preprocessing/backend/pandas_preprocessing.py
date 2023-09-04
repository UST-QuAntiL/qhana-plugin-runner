from typing import List, Optional
from pandas._libs.lib import no_default
from pandas import DataFrame
from .utils import get_number_if_possible
from enum import Enum
from json import loads as json_loads


class PreprocessingEnum(Enum):
    do_nothing = "do nothing"
    drop_na = "drop na"
    fill_na = "fill na"
    drop_duplicates = "drop duplicates"
    sort_values = "sort values"
    strip_entries = "strip entries"
    split_column = "split column"
    replace = "replace"
    string_case = "string case"

    def preprocess_df(self, df: DataFrame, preprocessing_params) -> Optional[DataFrame]:
        if self == PreprocessingEnum.drop_na:
            processing_function = drop_missing_value
        elif self == PreprocessingEnum.fill_na:
            processing_function = fill_missing_value
        elif self == PreprocessingEnum.drop_duplicates:
            processing_function = drop_duplicates
        elif self == PreprocessingEnum.sort_values:
            processing_function = sort_values
        elif self == PreprocessingEnum.strip_entries:
            processing_function = strip_characters
        elif self == PreprocessingEnum.split_column:
            processing_function = split_column
        elif self == PreprocessingEnum.replace:
            processing_function = replace
        elif self == PreprocessingEnum.string_case:
            processing_function = string_case
        else:
            return None

        params = {k: v for k, v in preprocessing_params.items() if v is not None}
        for k, v in params.items():
            if isinstance(v, Enum):
                params[k] = v.get()
        return processing_function(df, **params)


class AxisEnum(Enum):
    rows = "Rows"
    columns = "Columns"

    def get(self):
        if self == AxisEnum.rows:
            return 0
        return 1


class KeepEnum(Enum):
    first = "first"
    last = "last"
    none = "none"

    def get(self):
        return self.value


class PositionEnum(Enum):
    front = "front"
    end = "end"
    both = "both"

    def get(self):
        return self.value


class CaseEnum(Enum):
    upper = "upper"
    lower = "lower"
    title = "title"

    def get(self):
        return self.value


def process_subset(subset: Optional[str]) -> List[str]:
    if subset is not None:
        result = None
        if subset != "":
            result = json_loads(subset)
            if isinstance(subset, list):
                result = None if len(result) == 0 else result
            else:
                result = None
    return result


def drop_missing_value(
    df: DataFrame,
    axis: int = 0,
    threshold: int = no_default,
    subset: str = None,
    **kwargs,
) -> DataFrame:
    subset = process_subset(subset)

    df.dropna(axis=axis, thresh=threshold, subset=subset, inplace=True)

    return df


def fill_missing_value(df: DataFrame, fill_value: str, **kwargs) -> DataFrame:
    fill_value = get_number_if_possible(fill_value)

    df.fillna(value=fill_value, inplace=True)

    return df


def drop_duplicates(
    df: DataFrame,
    subset: str = None,
    keep: str = "first",
    ignore_index: bool = False,
    **kwargs,
) -> DataFrame:
    subset = process_subset(subset)

    keep = False if keep == "none" else keep

    df.drop_duplicates(subset=subset, keep=keep, ignore_index=ignore_index, inplace=True)

    return df


def sort_values(df: DataFrame, by: str, ascending: bool = False, **kwargs):
    df.sort_values(by=by, ascending=ascending, inplace=True)
    return df


def strip_characters(
    df: DataFrame,
    characters: List[str] = "",
    subset: str = None,
    position: str = "both",
    **kwargs,
) -> DataFrame:
    subset = process_subset(subset)
    if subset is None:
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
    **kwargs,
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


def replace(
    df: DataFrame, substring: str, new_str: str, subset: str = "", **kwargs
) -> DataFrame:
    subset = process_subset(subset)
    if subset is None:
        subset = df.keys()

    for k in subset:
        if k in df:
            if {type(el) for el in df[k]} == {str}:
                df[k] = df[k].str.replace(substring, new_str)

    return df


def string_case(df: DataFrame, case: str, subset: str = "", **kwargs) -> DataFrame:
    subset = process_subset(subset)
    if subset is None:
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
