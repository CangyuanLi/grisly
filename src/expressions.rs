use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn _ordered_unique(value: &str, output: &mut String) {
    let mut seen = std::collections::HashSet::new();
    let mut items: Vec<&str> = value.split(' ').collect();
    items.retain(|item| seen.insert(*item));

    output.push_str(items.join(" ").as_str());
}

#[polars_expr(output_type=Utf8)]
fn ordered_unique(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out = ca.apply_to_buffer(_ordered_unique);

    Ok(out.into_series())
}
