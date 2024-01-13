from __future__ import annotations

from collections.abc import Iterable
from typing import Union

import polars as pl
from polars.utils.udfs import _get_shared_lib_location

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]

LIB = _get_shared_lib_location(__file__)


def move_column(df: pl.DataFrame, idx: int, col: str) -> pl.DataFrame:
    """Move column to specified index.

    Parameters
    ----------
    df : pl.DataFrame
        _description_
    idx : int
        _description_
    col : str
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    s = df[col]
    df = df.drop(col)
    df.insert_column(idx, s)


def waterfall_join(
    left: PolarsFrame, right: PolarsFrame, left_on: Iterable[str], right_on=str
) -> PolarsFrame:
    left = left.with_row_count("index")
    seen: list[int] = []
    outputs: list[PolarsFrame] = []
    for col in left_on:
        output = left.filter(~pl.col("index").is_in(seen)).join(
            right, left_on=col, right_on=right_on, how="inner"
        )

        seen.extend(
            output.select("index").lazy().collect().get_column("index").to_list()
        )
        outputs.append(output)

    return pl.concat(outputs).sort("index").drop("index")


def replace_with_null(
    expr: pl.Expr, to_replace: Union[str, Iterable[str]], literal: bool = True
) -> pl.Expr:
    if isinstance(to_replace, str):
        to_replace = [to_replace]

    if literal:
        for x in to_replace:
            expr = pl.when(expr == pl.lit(x)).then(None).otherwise(expr)
    else:
        for pattern in to_replace:
            expr = (
                pl.when(expr.str.count_matches(pattern) > 0).then(None).otherwise(expr)
            )

    return expr.keep_name()


def normalize_whitespace(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(" +", " ")


def remove_whitespace(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(r"\s", "")


def coerce_ascii(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all("[^\p{Ascii}]", "")


def unique_words(expr: pl.Expr) -> pl.Expr:
    return expr._register_plugin(
        lib=LIB,
        symbol="unique_words",
        is_elementwise=True,
    )


def map_words(expr: pl.Expr, mapper: dict[str, str]) -> pl.Expr:
    return expr._register_plugin(
        lib=LIB,
        args=[],
        kwargs={"mapper": mapper},
        symbol="map_words",
        is_elementwise=True,
    )


def remove_chars(expr: pl.Expr, unwanted: Iterable[str]) -> pl.Expr:
    for char in unwanted:
        expr = expr.str.replace_all(char, "", literal=True)

    return expr


def keep_only(expr: pl.Expr, to_keep: str) -> pl.Expr:
    return expr.str.replace_all(f"[^{to_keep}]", "")


def remove_generational_suffixes(expr: pl.Expr) -> pl.Expr:
    return (
        expr.str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?III\s*?$", "")
        .str.replace_all(r"\s?IV\s*?$", "")
    )
