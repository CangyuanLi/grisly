# ruff: noqa: E402,F401,F403

from collections.abc import Iterable
from typing import Literal, Union

import polars as pl

from ._preprocessing import (
    coerce_ascii,
    keep_only,
    map_words,
    move_column,
    normalize,
    normalize_whitespace,
    remove_bracketed_content,
    remove_chars,
    remove_diacritics,
    remove_digits,
    remove_generational_suffixes,
    remove_whitespace,
    replace_chars,
    replace_digits,
    replace_whitespace,
    replace_with_null,
    unique_words,
    waterfall_join,
)


@pl.api.register_expr_namespace("grs")
class GrislyExpr:
    def __init__(self, expr) -> None:
        self._expr = expr

    def coerce_ascii(self):
        return coerce_ascii(self._expr)

    def keep_only(self, to_keep: str):
        return keep_only(self._expr, to_keep)

    def map_words(self, mapping: dict[str, str]):
        return map_words(self._expr, mapping)

    def normalize(self, form: Literal["NFC", "NKFC", "NFD", "NKFD"]):
        return normalize(self._expr, form)

    def normalize_whitespace(self):
        return normalize_whitespace(self._expr)

    def remove_bracketed_content(
        self,
        open_brackets: str = "([{<",
        close_brackets: str = ")]}>",
        any_combination: bool = False,
    ):
        return remove_bracketed_content(
            self._expr, open_brackets, close_brackets, any_combination
        )

    def remove_chars(self, unwanted: Iterable[str]):
        return remove_chars(self._expr, unwanted)

    def remove_diacritics(self):
        return remove_diacritics(self._expr)

    def remove_digits(self, only_blocks: bool = False):
        return remove_digits(self._expr, only_blocks)

    def remove_generational_suffixes(self):
        return remove_generational_suffixes(self._expr)

    def remove_whitespace(self):
        return remove_whitespace(self._expr)

    def replace_chars(self, unwanted: str, value: str):
        return replace_chars(self._expr, unwanted, value)

    def replace_digits(self, value: str, only_blocks: bool = False):
        return replace_digits(self._expr, value, only_blocks)

    def replace_whitespace(self, value: str):
        return replace_whitespace(self._expr, value)

    def replace_with_null(self, value: Union[str, Iterable[str]], literal: bool = True):
        return replace_with_null(self._expr, value, literal)

    def unique_words(self):
        return unique_words(self._expr)
