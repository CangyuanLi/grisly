import random
import timeit
import unicodedata

import polars as pl

import grisly

c = pl.col


def get_random_unicode(length):
    # Update this to include code point ranges to be sampled
    include_ranges = [
        (0x0021, 0x0021),
        (0x0023, 0x0026),
        (0x0028, 0x007E),
        (0x00A1, 0x00AC),
        (0x00AE, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
        (0x2C60, 0x2C7F),
        (0x16A0, 0x16F0),
        (0x0370, 0x0377),
        (0x037A, 0x037E),
        (0x0384, 0x038A),
        (0x038C, 0x038C),
    ]

    alphabet = [
        chr(code_point)
        for current_range in include_ranges
        for code_point in range(current_range[0], current_range[1] + 1)
    ]

    return "".join(random.choice(alphabet) for _ in range(length))


def grs_norm(df):
    return df.with_columns(y=c("x").pipe(grisly.normalize, "NFC"))


def py_norm(df):
    return df.with_columns(
        y=c("x").map_elements(lambda x: unicodedata.normalize("NFC", x))
    )


def main():
    HEIGHT = 1_000

    df = pl.DataFrame({"x": [get_random_unicode(25) for _ in range(HEIGHT)]})

    print(timeit.timeit(lambda: grs_norm(df)))
    print(timeit.timeit(lambda: py_norm(df)))


if __name__ == "__main__":
    main()
