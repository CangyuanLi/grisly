use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

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
    mapper: std::collections::HashMap<String, String>,
}

fn _map_words(
    value: &str,
    mapper: &std::collections::HashMap<String, String>,
    output: &mut String,
) {
    let vec: Vec<&str> = value
        .split_whitespace()
        .map(|word| match mapper.get(word) {
            Some(val) => val,
            None => word,
        })
        .collect();

    output.push_str(vec.join(" ").as_str())
}

#[polars_expr(output_type=Utf8)]
fn map_words(inputs: &[Series], kwargs: MapWordsKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out = ca.apply_to_buffer(|val, buf| _map_words(val, &kwargs.mapper, buf));

    Ok(out.into_series())
}
