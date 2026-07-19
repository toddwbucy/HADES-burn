//! Canonical JSON encoding for hashes whose identity must ignore object insertion order.

use serde_json::Value;

pub(crate) fn to_vec(value: &Value) -> Vec<u8> {
    let mut output = Vec::new();
    write_value(value, &mut output);
    output
}

fn write_value(value: &Value, output: &mut Vec<u8>) {
    match value {
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_value(value, output);
            }
            output.push(b']');
        }
        Value::Object(fields) => {
            output.push(b'{');
            let mut keys: Vec<&str> = fields.keys().map(String::as_str).collect();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend(serde_json::to_vec(key).unwrap_or_default());
                output.push(b':');
                write_value(&fields[key], output);
            }
            output.push(b'}');
        }
        scalar => output.extend(serde_json::to_vec(scalar).unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_insertion_order_does_not_change_encoding() {
        let mut first = serde_json::Map::new();
        first.insert("a".into(), serde_json::json!({"x": 1, "y": 2}));
        first.insert("b".into(), serde_json::json!(3));

        let mut nested = serde_json::Map::new();
        nested.insert("y".into(), serde_json::json!(2));
        nested.insert("x".into(), serde_json::json!(1));
        let mut second = serde_json::Map::new();
        second.insert("b".into(), serde_json::json!(3));
        second.insert("a".into(), Value::Object(nested));

        assert_eq!(
            to_vec(&Value::Object(first)),
            to_vec(&Value::Object(second))
        );
    }
}
