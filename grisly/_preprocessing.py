from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Union

import polars as pl
from polars.plugins import register_plugin_function
from polars.type_aliases import IntoExpr

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]

PLUGIN_PATH = Path(__file__).resolve().parents[0]


def move_column(df: pl.DataFrame, index: int, column_name: str):
    """Move column to specified index. This operation is in place.

    Parameters
    ----------
    df : pl.DataFrame
    index : int
        Index to move the column to.
    column_name : str
        The name of the column to move.

    Examples
    --------
    >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6], "baz": [97, 98, 99]})
    >>> grisly.move_column(df, 1, "baz")
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ baz ┆ bar │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 97  ┆ 4   │
    │ 2   ┆ 98  ┆ 5   │
    │ 3   ┆ 99  ┆ 6   │
    └─────┴─────┴─────┘
    """
    s = df[column_name]
    df = df.drop(column_name)
    df.insert_column(index, s)


def waterfall_join(
    left: PolarsFrame, right: PolarsFrame, left_on: Iterable[str], right_on=str
) -> PolarsFrame:
    index_col = "adsl;fkj89u-alskdfn_3iifaocxkm,mv"
    left = left.with_row_count(index_col)
    seen: list[int] = []
    outputs: list[PolarsFrame] = []
    for col in left_on:
        output = left.filter(~pl.col(index_col).is_in(seen)).join(
            right, left_on=col, right_on=right_on, how="inner"
        )

        seen.extend(
            output.select(index_col).lazy().collect().get_column(index_col).to_list()
        )
        outputs.append(output)

    return pl.concat(outputs).sort(index_col).drop(index_col)


def replace_with_null(
    expr: pl.Expr, value: Union[str, Iterable[str]], literal: bool = True
) -> pl.Expr:
    if isinstance(value, str):
        value = [value]

    if literal:
        for x in value:
            expr = pl.when(expr == pl.lit(x)).then(None).otherwise(expr)
    else:
        for pattern in value:
            expr = (
                pl.when(expr.str.count_matches(pattern) > 0).then(None).otherwise(expr)
            )

    return expr.keep_name()


def normalize_whitespace(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(" +", " ")


def replace_whitespace(expr: pl.Expr, value: str) -> pl.Expr:
    return expr.str.replace_all(r"\s", value)


def remove_whitespace(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(r"\s", "")


def replace_digits(expr: pl.Expr, value: str, only_blocks: bool = False) -> pl.Expr:
    if only_blocks:
        pattern = r"\b\d+\b"
    else:
        pattern = r"\d+"

    return expr.str.replace_all(pattern, value)


def remove_digits(expr: pl.Expr, only_blocks: bool = False) -> pl.Expr:
    return replace_digits(expr, "", only_blocks)


def coerce_ascii(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all("[^\p{Ascii}]", "")


def unique_words(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="unique_words",
        args=expr,
        is_elementwise=True,
    )


def map_words(expr: IntoExpr, mapping: dict[str, str]) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="map_words",
        args=expr,
        kwargs={"mapping": mapping},
        is_elementwise=True,
    )


def replace_chars(expr: IntoExpr, unwanted: Iterable[str], value: str) -> pl.Expr:
    for char in unwanted:
        expr = expr.str.replace_all(char, value, literal=True)

    return expr


def remove_chars(expr: IntoExpr, unwanted: Iterable[str]) -> pl.Expr:
    return replace_chars(expr, unwanted, "")


def keep_only(expr: IntoExpr, to_keep: str) -> pl.Expr:
    return expr.str.replace_all(f"[^{to_keep}]", "")


def remove_generational_suffixes(expr: pl.Expr) -> pl.Expr:
    return (
        expr.str.replace_all(r"(?i)\s?J\.*?R\.*\s*?$", "")
        .str.replace_all(r"(?i)\s?S\.*?R\.*\s*?$", "")
        .str.replace_all(r"(?i)\s?III\s*?$", "")
        .str.replace_all(r"(?i)\s?IV\s*?$", "")
    )


def normalize(expr: IntoExpr, form: Literal["NFC", "NKFC", "NFD", "NKFD"]) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="normalize",
        args=expr,
        kwargs={"form": form},
        is_elementwise=True,
    )


def remove_diacritics(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="remove_diacritics",
        args=expr,
        is_elementwise=True,
    )


def remove_bracketed_content(
    expr: pl.Expr,
    open_brackets: str = "([{<",
    close_brackets: str = ")]}>",
    any_combination: bool = False,
) -> pl.Expr:
    if any_combination:
        brackets = []
        for a in open_brackets:
            for b in close_brackets:
                brackets.append(a)
                brackets.append(b)
        brackets = "".join(brackets)
    else:
        brackets = "".join(f"{a}{b}" for a, b in zip(open_brackets, close_brackets))

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="removed_bracketed_content",
        args=expr,
        kwargs={"brackets": brackets},
        is_elementwise=True,
    )
