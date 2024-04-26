# ruff: noqa: E402,F401,F403

from collections.abc import Iterable
from typing import Literal, Union

import polars as pl

from ._preprocessing import (
    map_words,
    move_column,
    normalize,
    normalize_whitespace,
    remove_bracketed_content,
    remove_diacritics,
    remove_generational_suffixes,
    replace_chars,
    replace_digits,
    replace_whitespace,
    replace_with_null,
    unique_words,
    waterfall_join,
)
