import polars as pl
from polars.utils.udfs import _get_shared_lib_location

LIB = _get_shared_lib_location(__file__)


def normalize_whitespace(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(" +", " ", literal=False)


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


def remove_chars(expr: pl.Expr, unwanted: list[str]) -> pl.Expr:
    for char in unwanted:
        expr = expr.str.replace_all(char, "", literal=True)

    return expr


def remove_generational_suffixes(expr: pl.Expr) -> pl.Expr:
    return (
        expr.str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?III\s*?$", "")
        .str.replace_all(r"\s?IV\s*?$", "")
    )
