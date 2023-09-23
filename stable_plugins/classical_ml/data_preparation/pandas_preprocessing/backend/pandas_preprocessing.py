from typing import List, Optional, Sequence
from pandas._libs.lib import no_default
from pandas import DataFrame
from .utils import get_number_if_possible
from enum import Enum
from json import loads as json_loads


class PreprocessingEnum(Enum):
    do_nothing = "do nothing"
    drop_na = "drop entries with missing values"
    fill_na = "fill missing values"
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


def process_subset(subset: Optional[str]) -> Optional[List[str]]:
    """
    Deserializes a json string, called subset in this method. If subset is a list of strings, then this list is
    returned, else None is returned.
    :param subset: Optional[str]
    :return: List[str]
    """
    result = None
    if subset:
        result = json_loads(subset)
        if isinstance(result, list):
            contents_type = {isinstance(el, str) for el in result}
            if False not in contents_type:
                result = None if len(result) == 0 else result
            else:
                result = None
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
    """
    First transforms subset from JSON encoded list into a list of strings and then continues with the dropna function of pandas'
    dataframes (https://pandas.pydata.org/pandas-docs/version/1.5.0/reference/api/pandas.DataFrame.dropna.html) and
    returns the new dataframe.
    :param df: DataFrame
    :param axis: int
    :param threshold: int
    :param subset: str containing the columns or rows separated by commas
    :return: DataFrame
    """
    subset = process_subset(subset)

    df.dropna(axis=axis, thresh=threshold, subset=subset, inplace=True)

    return df


def fill_missing_value(df: DataFrame, fill_value: str, **kwargs) -> DataFrame:
    """
    First transforms fill_value into a number, if possible and then continues with the fillna function of pandas'
    dataframes (https://pandas.pydata.org/pandas-docs/version/1.5.0/reference/api/pandas.DataFrame.fillna.html) and
    returns the new dataframe.
    :param df: DataFrame
    :param fill_value: str
    :return: DataFrame
    """
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
    """
    First transforms subset from a string into a list of strings and then continues with the drop_duplicates function of
    pandas' dataframes (https://pandas.pydata.org/pandas-docs/version/1.5.0/reference/api/pandas.DataFrame.drop_duplicates.html),
    with the key difference that keep == False is replaced by keep == 'none'. Afterwards, return the new dataframe.
    :param df: DataFrame
    :param subset: str containing the columns separated by commas
    :param keep: str
    :param ignore_index: bool
    :return: DataFrame
    """
    subset = process_subset(subset)

    keep = False if keep == "none" else keep

    df.drop_duplicates(subset=subset, keep=keep, ignore_index=ignore_index, inplace=True)

    return df


def sort_values(df: DataFrame, by: str, ascending: bool = False, **kwargs):
    """
    Executes the sort_values function of pandas' dataframes (https://pandas.pydata.org/pandas-docs/version/1.5.0/reference/api/pandas.DataFrame.sort_values.html)
     and returns the new dataframe.
    :param df: DataFrame
    :param by: str
    :param ascending: str
    :return: DataFrame
    """
    df.sort_values(by=by, ascending=ascending, inplace=True)
    return df


def strip_characters(
    df: DataFrame,
    characters: Sequence[str] = "",
    subset: str = None,
    position: str = "both",
    **kwargs,
) -> DataFrame:
    """
    First transforms subset from a string into a list of strings and then continues to strip each entry in each given
    column from the specified characters. If subset is empty, then all columns will be considered.
    - position='front': A left strip will be used, to strip each entry.
    - position='end': A right strip will be used, to strip each entry.
    - position='both': Both left and right side of an entry will be striped.
    :param df: DataFrame
    :param characters: Sequence[str] contains the characters that will be striped
    :param subset: str containing the columns separated by commas
    :param position: str
    :return: DataFrame
    """
    subset = process_subset(subset)
    if not subset:
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
            if all(isinstance(el, str) for el in df[k]):
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
    """
    Splits a column into many columns, using the string `by` to split each entry.
    :param df: DataFrame
    :param column: str contains the name of the old column
    :param by: str
    :param new_columns: str contains the names of the new columns separated by commas
    :param remove_column: bool
    :return: DataFrame
    """
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
    """
    First transforms subset from a string into a list of strings and then continues to replace each occurrence of
    substring in each entry of the specified columns with new_str.
    :param df: DataFrame
    :param substring: str
    :param new_str: str
    :param subset: str containing the columns separated by commas
    :return: DataFrame
    """
    subset = process_subset(subset)
    if not subset:
        subset = df.keys()

    for k in subset:
        if k in df:
            if all(isinstance(el, str) for el in df[k]):
                df[k] = df[k].str.replace(substring, new_str)

    return df


def string_case(df: DataFrame, case: str, subset: str = "", **kwargs) -> DataFrame:
    """
    First transforms subset from a string into a list of strings and then continues to transform each entry to comply
    with the specified case.
    - case='upper': Each entry will be written in upper case.
    - case='lower': Each entry will be written in lower case.
    - case='title': Each entry will be written in title case.
    :param df: DataFrame
    :param case: str
    :param subset: str containing the columns separated by commas
    :return: DataFrame
    """
    subset = process_subset(subset)
    if not subset:
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
            if all(isinstance(el, str) for el in df[k]):
                df[k] = function(df[k].str)

    return df
