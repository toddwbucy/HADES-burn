//! Shared CLI output formatting.
//!
//! Provides a consistent output contract across all commands:
//! - **JSON envelope**: `{success, command, data, timestamp}` wrapper
//!   matching the Python HADES convention for automation compatibility.
//! - **Table format**: Simple column-aligned text for human readability.
//! - **Format validation**: Accepts "json", "jsonl", and "table".

use anyhow::Result;
use serde_json::Value;

/// Supported output formats.
pub enum OutputFormat {
    Json,
    Jsonl,
    Table,
}

impl OutputFormat {
    /// Parse a format string, returning an error for unknown values.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "json" => Ok(Self::Json),
            "jsonl" => Ok(Self::Jsonl),
            "table" => Ok(Self::Table),
            other => anyhow::bail!(
                "unsupported format '{other}' — supported values: json, table, jsonl"
            ),
        }
    }
}

/// Wrap data in the standard HADES JSON envelope.
///
/// Matches the Python HADES output convention:
/// ```json
/// { "success": true, "command": "db.collections", "data": {...}, "timestamp": "..." }
/// ```
pub fn envelope(command: &str, data: Value) -> Value {
    serde_json::json!({
        "success": true,
        "command": command,
        "data": data,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })
}

/// Wrap an error in the standard HADES JSON envelope.
pub fn error_envelope(command: &str, message: &str) -> Value {
    serde_json::json!({
        "success": false,
        "command": command,
        "error": message,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })
}

/// Print data in the requested format with the standard envelope.
pub fn print_output(command: &str, data: Value, format: &OutputFormat) {
    match format {
        OutputFormat::Json => {
            let wrapped = envelope(command, data);
            println!(
                "{}",
                serde_json::to_string_pretty(&wrapped).unwrap_or_default()
            );
        }
        OutputFormat::Jsonl => {
            let wrapped = envelope(command, data);
            println!(
                "{}",
                serde_json::to_string(&wrapped).unwrap_or_default()
            );
        }
        OutputFormat::Table => {
            print_table(&data);
        }
    }
}

/// Render a JSON value as a human-readable table.
///
/// Handles common output shapes:
/// - Object with a top-level array field → tabular rows
/// - Array of objects → tabular rows
/// - Simple object → key-value pairs
fn print_table(value: &Value) {
    match value {
        Value::Object(map) => {
            // Look for the primary array field to render as table rows.
            // Common patterns: "collections", "tasks", "results", etc.
            if let Some((key, arr)) = find_primary_array(map) {
                // Print any scalar fields as a header.
                for (k, v) in map {
                    if k != key && !v.is_array() && !v.is_object() {
                        eprintln!("{k}: {}", format_scalar(v));
                    }
                }
                print_array_table(arr);
            } else {
                // No array — print as key-value pairs.
                print_kv_table(map);
            }
        }
        Value::Array(arr) => {
            print_array_table(arr);
        }
        _ => {
            println!("{}", serde_json::to_string_pretty(value).unwrap_or_default());
        }
    }
}

/// Find the primary array field in an object for table rendering.
fn find_primary_array(
    map: &serde_json::Map<String, Value>,
) -> Option<(&str, &Vec<Value>)> {
    // Prefer known field names.
    for name in &["collections", "tasks", "results", "documents", "entries", "edges", "symbols"] {
        if let Some(Value::Array(arr)) = map.get(*name) {
            return Some((name, arr));
        }
    }
    // Fall back to the first array field.
    for (k, v) in map {
        if let Value::Array(arr) = v {
            return Some((k.as_str(), arr));
        }
    }
    None
}

/// Print an array of objects as aligned columns.
fn print_array_table(arr: &[Value]) {
    if arr.is_empty() {
        println!("(empty)");
        return;
    }

    // Collect all keys from the first object to determine columns.
    let columns: Vec<String> = match &arr[0] {
        Value::Object(map) => map.keys().cloned().collect(),
        _ => {
            // Array of scalars — just print one per line.
            for item in arr {
                println!("{}", format_scalar(item));
            }
            return;
        }
    };

    // Compute column widths.
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    let rows: Vec<Vec<String>> = arr
        .iter()
        .map(|item| {
            columns
                .iter()
                .enumerate()
                .map(|(i, col)| {
                    let s = match item.get(col) {
                        Some(v) => format_scalar(v),
                        None => String::new(),
                    };
                    if s.len() > widths[i] {
                        widths[i] = s.len();
                    }
                    s
                })
                .collect()
        })
        .collect();

    // Cap column widths at 60 chars for readability.
    for w in &mut widths {
        if *w > 60 {
            *w = 60;
        }
    }

    // Print header.
    let header: String = columns
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{:width$}", c, width = widths[i]))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{header}");
    let separator: String = widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>().join("  ");
    println!("{separator}");

    // Print rows.
    for row in &rows {
        let line: String = row
            .iter()
            .enumerate()
            .map(|(i, val)| {
                let truncated = if val.len() > widths[i] {
                    format!("{}…", &val[..widths[i] - 1])
                } else {
                    val.clone()
                };
                format!("{:width$}", truncated, width = widths[i])
            })
            .collect::<Vec<_>>()
            .join("  ");
        println!("{line}");
    }
}

/// Print an object as key-value pairs.
fn print_kv_table(map: &serde_json::Map<String, Value>) {
    let max_key_len = map.keys().map(|k| k.len()).max().unwrap_or(0);
    for (k, v) in map {
        match v {
            Value::Object(_) | Value::Array(_) => {
                println!(
                    "{:width$}  {}",
                    k,
                    serde_json::to_string_pretty(v).unwrap_or_default(),
                    width = max_key_len
                );
            }
            _ => {
                println!("{:width$}  {}", k, format_scalar(v), width = max_key_len);
            }
        }
    }
}

/// Format a scalar JSON value for display.
fn format_scalar(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}
