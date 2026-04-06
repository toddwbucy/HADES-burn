//! Native Rust handlers for `hades task` CRUD commands.
//!
//! Each function constructs a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout using the standard HADES JSON envelope.

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{
    self, DaemonCommand, TaskApproveParams, TaskBlockParams, TaskCloseParams, TaskContextParams,
    TaskCreateParams, TaskDepParams, TaskGraphIntegrationParams, TaskHandoffParams,
    TaskHandoffShowParams, TaskListParams, TaskLogParams, TaskReviewParams, TaskSessionsParams,
    TaskShowParams, TaskStartParams, TaskUnblockParams, TaskUpdateParams, TaskUsageParams,
};

use super::output::{self, OutputFormat};

/// Connect, dispatch a command, and print the result with envelope.
async fn dispatch_and_print(
    config: &HadesConfig,
    cmd: DaemonCommand,
    command_name: &str,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(&pool, config, cmd).await?;
    output::print_output(command_name, result, &fmt);
    Ok(())
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
        "task.list",
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
        "task.show",
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
        "task.create",
        "json",
    )
    .await
}

/// `hades task update KEY [options]`
pub async fn run_update(config: &HadesConfig, params: TaskUpdateParams) -> Result<()> {
    dispatch_and_print(config, DaemonCommand::TaskUpdate(params), "task.update", "json").await
}

/// `hades task close KEY [--message M]`
pub async fn run_close(config: &HadesConfig, key: &str, message: Option<&str>) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskClose(TaskCloseParams {
            key: key.to_string(),
            message: message.map(String::from),
        }),
        "task.close",
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
        "task.start",
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
        "task.review",
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
        "task.approve",
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
        "task.block",
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
        "task.unblock",
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
        "task.handoff",
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
        "task.handoff-show",
        format,
    )
    .await
}

/// `hades task context KEY`
pub async fn run_context(config: &HadesConfig, key: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskContext(TaskContextParams {
            key: key.to_string(),
        }),
        "task.context",
        "json",
    )
    .await
}

/// `hades task log KEY [--limit N]`
pub async fn run_log(config: &HadesConfig, key: &str, limit: u32) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskLog(TaskLogParams {
            key: key.to_string(),
            limit: Some(limit),
        }),
        "task.log",
        "json",
    )
    .await
}

/// `hades task sessions KEY`
pub async fn run_sessions(config: &HadesConfig, key: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskSessions(TaskSessionsParams {
            key: key.to_string(),
        }),
        "task.sessions",
        "json",
    )
    .await
}

/// `hades task dep KEY [--add K] [--remove K] [--graph]`
pub async fn run_dep(
    config: &HadesConfig,
    key: &str,
    add: Option<&str>,
    remove: Option<&str>,
    graph: bool,
) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskDep(TaskDepParams {
            key: key.to_string(),
            add: add.map(String::from),
            remove: remove.map(String::from),
            graph,
        }),
        "task.dep",
        "json",
    )
    .await
}

/// `hades task usage`
pub async fn run_usage(config: &HadesConfig) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskUsage(TaskUsageParams {}),
        "task.usage",
        "json",
    )
    .await
}

/// `hades task graph-integration [--format F]`
pub async fn run_graph_integration(config: &HadesConfig, format: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::TaskGraphIntegration(TaskGraphIntegrationParams {}),
        "task.graph-integration",
        format,
    )
    .await
}
