use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use unicode_normalization::UnicodeNormalization;

fn _unique_words(value: &str, output: &mut String) {
    let mut seen = std::collections::HashSet::new();
    let mut items: Vec<&str> = value.split(' ').collect();
    items.retain(|item| seen.insert(*item));

    output.push_str(items.join(" ").as_str());
}

#[polars_expr(output_type=Utf8)]
fn unique_words(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out = ca.apply_to_buffer(_unique_words);

    Ok(out.into_series())
}

#[derive(serde::Deserialize)]
struct MapWordsKwargs {
    mapping: std::collections::HashMap<String, String>,
}

fn _map_words(
    value: &str,
    mapping: &std::collections::HashMap<String, String>,
    output: &mut String,
) {
    let vec: Vec<&str> = value
        .split_whitespace()
        .map(|word| match mapping.get(word) {
            Some(val) => val,
            None => word,
        })
        .collect();

    output.push_str(vec.join(" ").as_str())
}

#[polars_expr(output_type=Utf8)]
fn map_words(inputs: &[Series], kwargs: MapWordsKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out = ca.apply_to_buffer(|val, buf| _map_words(val, &kwargs.mapping, buf));

    Ok(out.into_series())
}

#[derive(serde::Deserialize)]
struct NormalizeKwargs {
    form: String,
}

fn _normalize(value: &str, form: &str, output: &mut String) {
    if form == "NFC" {
        *output = value.nfc().collect()
    } else if form == "NKFC" {
        *output = value.nfkc().collect()
    } else if form == "NFD" {
        *output = value.nfd().collect()
    } else if form == "NFKD" {
        *output = value.nfkd().collect()
    }
}

#[polars_expr(output_type=Utf8)]
fn normalize(inputs: &[Series], kwargs: NormalizeKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out = ca.apply_to_buffer(|val, buf| _normalize(val, &kwargs.form, buf));

    Ok(out.into_series())
}

fn _remove_diacritics(value: &str, output: &mut String) {
    *output = value.nfd().filter(char::is_ascii).collect()
}

#[polars_expr(output_type=Utf8)]
fn remove_diacritics(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out = ca.apply_to_buffer(_remove_diacritics);

    Ok(out.into_series())
}
