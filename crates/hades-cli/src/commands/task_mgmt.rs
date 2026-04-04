//! Native Rust handlers for `hades task` CRUD commands.
//!
//! Each function constructs a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout.

use anyhow::{Context, Result};
use serde_json::Value;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{
    self, DaemonCommand, TaskApproveParams, TaskBlockParams, TaskCloseParams, TaskCreateParams,
    TaskHandoffParams, TaskHandoffShowParams, TaskListParams, TaskReviewParams, TaskShowParams,
    TaskStartParams, TaskUnblockParams, TaskUpdateParams,
};

/// Connect, dispatch a command, and print the result in the requested format.
async fn dispatch_and_print(config: &HadesConfig, cmd: DaemonCommand, format: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(&pool, config, cmd).await?;
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(&result)?),
        _ => print_text(&result),
    }
    Ok(())
}

/// Simple text rendering of a JSON value.
fn print_text(value: &Value) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                match v {
                    Value::Object(_) | Value::Array(_) => {
                        println!("{k}:");
                        println!("{}", serde_json::to_string_pretty(v).unwrap_or_default());
                    }
                    _ => println!("{k}: {v}"),
                }
            }
        }
        _ => println!("{}", serde_json::to_string_pretty(value).unwrap_or_default()),
    }
}

/// `hades task list [--status S] [--type T] [--parent P] [--limit N] [--format F]`
pub async fn run_list(
    config: &HadesConfig,
    status: Option<&str>,
    task_type: Option<&str>,
    parent: Option<&str>,
    limit: u32,
    format: &str,
) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskList(TaskListParams {
            status: status.map(String::from),
            task_type: task_type.map(String::from),
            parent: parent.map(String::from),
            limit: Some(limit),
        }),
        format,
    )
    .await
}

/// `hades task show KEY [--format F]`
pub async fn run_show(config: &HadesConfig, key: &str, format: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskShow(TaskShowParams {
            key: key.to_string(),
        }),
        format,
    )
    .await
}

/// `hades task create TITLE [options]`
pub async fn run_create(
    config: &HadesConfig,
    title: &str,
    description: Option<&str>,
    task_type: &str,
    parent: Option<&str>,
    priority: Option<&str>,
    tags: &[String],
) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskCreate(TaskCreateParams {
            title: title.to_string(),
            description: description.map(String::from),
            task_type: task_type.to_string(),
            parent: parent.map(String::from),
            priority: priority.unwrap_or("medium").to_string(),
            tags: tags.to_vec(),
        }),
        "json",
    )
    .await
}

/// `hades task update KEY [options]`
pub async fn run_update(config: &HadesConfig, params: TaskUpdateParams) -> Result<()> {
    dispatch_and_print(config, DaemonCommand::TaskUpdate(params), "json").await
}

/// `hades task close KEY [--message M]`
pub async fn run_close(config: &HadesConfig, key: &str, message: Option<&str>) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskClose(TaskCloseParams {
            key: key.to_string(),
            message: message.map(String::from),
        }),
        "json",
    )
    .await
}

/// `hades task start KEY`
pub async fn run_start(config: &HadesConfig, key: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskStart(TaskStartParams {
            key: key.to_string(),
        }),
        "json",
    )
    .await
}

/// `hades task review KEY [--message M]`
pub async fn run_review(config: &HadesConfig, key: &str, message: Option<&str>) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskReview(TaskReviewParams {
            key: key.to_string(),
            message: message.map(String::from),
        }),
        "json",
    )
    .await
}

/// `hades task approve KEY [--human]`
pub async fn run_approve(config: &HadesConfig, key: &str, human: bool) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskApprove(TaskApproveParams {
            key: key.to_string(),
            human,
        }),
        "json",
    )
    .await
}

/// `hades task block KEY [--message M] [--blocker K]`
pub async fn run_block(
    config: &HadesConfig,
    key: &str,
    message: Option<&str>,
    blocker: Option<&str>,
) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskBlock(TaskBlockParams {
            key: key.to_string(),
            message: message.map(String::from),
            blocker: blocker.map(String::from),
        }),
        "json",
    )
    .await
}

/// `hades task unblock KEY`
pub async fn run_unblock(config: &HadesConfig, key: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskUnblock(TaskUnblockParams {
            key: key.to_string(),
        }),
        "json",
    )
    .await
}

/// `hades task handoff KEY [--message M]`
pub async fn run_handoff(config: &HadesConfig, key: &str, message: Option<&str>) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskHandoff(TaskHandoffParams {
            key: key.to_string(),
            message: message.map(String::from),
        }),
        "json",
    )
    .await
}

/// `hades task handoff-show KEY [--format F]`
pub async fn run_handoff_show(config: &HadesConfig, key: &str, format: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskHandoffShow(TaskHandoffShowParams {
            key: key.to_string(),
        }),
        format,
    )
    .await
}
