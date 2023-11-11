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
