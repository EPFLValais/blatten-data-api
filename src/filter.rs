//! STAC API Filter: Basic CQL2 over the listed queryables.
//!
//! A filter is parsed from cql2-text or cql2-json, checked to name only queryables and use only
//! the Basic CQL2 operators (comparisons, `IS NULL`, `AND`, `OR`, `NOT`), and evaluated against
//! an item's properties. The queryables listed here are also the JSON Schema served at
//! `/queryables`, so what a client is told it may filter on and what is accepted are one list.

use crate::catalog::StacItem;
use cql2::Expr;
use serde_json::Value;

/// The properties a filter may name: name, title and JSON Schema type (`date-time` is a string
/// in that format)
const QUERYABLES: &[(&str, &str, &str)] = &[
    ("blatten:series", "Series", "string"),
    ("blatten:sensor", "Sensor", "string"),
    ("blatten:code", "Dataset code", "string"),
    ("blatten:source", "Data provider", "string"),
    ("blatten:processing_level", "Processing level", "integer"),
    ("datetime", "Acquisition time", "date-time"),
    ("start_datetime", "Start of acquisition", "date-time"),
    ("end_datetime", "End of acquisition", "date-time"),
];

/// The Basic CQL2 operators, in the canonical spelling the parser gives them
const BASIC_OPS: &[&str] = &["and", "or", "not", "=", "<>", "<", "<=", ">", ">=", "isNull"];

/// The link relation of a queryables document
pub const QUERYABLES_REL: &str = "http://www.opengis.net/def/rel/ogc/1.0/queryables";

/// The JSON Schema of the queryables, served at `id`
pub fn queryables_schema(id: &str) -> Value {
    let properties: serde_json::Map<String, Value> = QUERYABLES
        .iter()
        .map(|(name, title, kind)| {
            let schema = match *kind {
                "date-time" => serde_json::json!({"title": title, "type": "string", "format": "date-time"}),
                kind => serde_json::json!({"title": title, "type": kind}),
            };
            ((*name).to_string(), schema)
        })
        .collect();
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "$id": id,
        "type": "object",
        "title": "Queryables",
        "properties": properties,
        "additionalProperties": false
    })
}

/// A checked Basic CQL2 filter
#[derive(Debug, Clone)]
pub struct Filter(Expr);

impl Filter {
    /// A filter from a GET query: `filter` in `filter-lang`, cql2-text when not given
    pub fn from_get(filter: &str, lang: Option<&str>) -> Result<Self, String> {
        match lang.unwrap_or("cql2-text") {
            "cql2-text" => Self::from_text(filter),
            "cql2-json" => {
                let value: Value = serde_json::from_str(filter).map_err(|e| format!("invalid cql2-json filter: {e}"))?;
                Self::from_json(value)
            }
            other => Err(unknown_lang(other)),
        }
    }

    /// A filter from a POST body: `filter` in `filter-lang`, cql2-json when not given
    pub fn from_post(filter: Value, lang: Option<&str>) -> Result<Self, String> {
        match (lang.unwrap_or("cql2-json"), filter) {
            ("cql2-json", value) => Self::from_json(value),
            ("cql2-text", Value::String(text)) => Self::from_text(&text),
            ("cql2-text", _) => Err("a cql2-text filter must be a string".into()),
            (other, _) => Err(unknown_lang(other)),
        }
    }

    fn from_text(text: &str) -> Result<Self, String> {
        let expr = cql2::parse_text(text).map_err(|e| format!("invalid cql2-text filter: {e}"))?;
        Self::checked(expr)
    }

    fn from_json(value: Value) -> Result<Self, String> {
        let expr = Expr::try_from(value).map_err(|e| format!("invalid cql2-json filter: {e}"))?;
        Self::checked(expr)
    }

    fn checked(expr: Expr) -> Result<Self, String> {
        check(&expr)?;
        Ok(Self(expr))
    }

    /// Whether the filter is true of `item`. A property the item does not have is null, so a
    /// comparison with it is not true and `IS NULL` of it is.
    pub fn matches(&self, item: &StacItem) -> bool {
        let Ok(Value::Object(properties)) = serde_json::to_value(&item.properties) else {
            return false;
        };
        let record: serde_json::Map<String, Value> = QUERYABLES
            .iter()
            .filter_map(|(name, _, _)| {
                let value = properties.get(*name).filter(|v| !v.is_null())?;
                Some(((*name).to_string(), value.clone()))
            })
            .collect();
        self.0.clone().matches(Some(&Value::Object(record))).unwrap_or(false)
    }
}

fn unknown_lang(lang: &str) -> String {
    format!("filter-lang must be cql2-text or cql2-json, not '{lang}'")
}

/// Refuse an expression that names a property outside the queryables or uses an operator or
/// value outside Basic CQL2
fn check(expr: &Expr) -> Result<(), String> {
    match expr {
        Expr::Operation { op, args } => {
            if !BASIC_OPS.contains(&op.as_str()) {
                return Err(format!("the filter operator '{op}' is not supported: only Basic CQL2 is"));
            }
            args.iter().try_for_each(|arg| check(arg))
        }
        Expr::Property { property } => {
            if QUERYABLES.iter().any(|(name, _, _)| name == property) {
                Ok(())
            } else {
                Err(format!("'{property}' is not a queryable property: see /stac/queryables"))
            }
        }
        Expr::Timestamp { timestamp: value } | Expr::Date { date: value } => match value.as_ref() {
            Expr::Literal(_) => Ok(()),
            _ => Err("TIMESTAMP and DATE take a literal".into()),
        },
        Expr::Float(_) | Expr::Literal(_) | Expr::Bool(_) | Expr::Null => Ok(()),
        other => Err(format!("the filter value {other:?} is not supported: only Basic CQL2 is")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_property_outside_the_queryables_or_an_operator_outside_basic_cql2_is_refused() {
        assert!(Filter::from_get("blatten:series = 'grove-1'", None).is_ok());
        assert!(Filter::from_get("datetime >= DATE('2024-01-01')", None).is_ok());
        let unknown = Filter::from_get("shade = 'dappled'", None).unwrap_err();
        assert!(unknown.contains("shade"), "{unknown}");
        assert!(Filter::from_get("blatten:series LIKE 'grove%'", None).is_err());
        assert!(Filter::from_get("blatten:series IN ('a', 'b')", None).is_err());
        assert!(Filter::from_get("S_INTERSECTS(geometry, POINT(1 1))", None).is_err());
        assert!(Filter::from_get("{\"op\":\"isNull\",\"args\":[{\"property\":\"blatten:code\"}]}", Some("cql2-json")).is_ok());
        assert!(Filter::from_post(Value::from(1), Some("cql2-text")).is_err());
        assert!(Filter::from_get("x = 1", Some("ecql")).is_err());
    }
}
