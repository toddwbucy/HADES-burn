//! `hades task` subcommands — Persephone task management.

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum TaskCmd {
    /// Create a new task.
    Create {
        /// Task title.
        title: String,

        /// Task description.
        #[arg(short = 'd', long)]
        description: Option<String>,

        /// Task type (epic, task, subtask).
        #[arg(short = 't', long, default_value = "task")]
        r#type: String,

        /// Parent task key.
        #[arg(short = 'p', long)]
        parent: Option<String>,

        /// Priority (critical, high, medium, low).
        #[arg(long)]
        priority: Option<String>,

        /// Tags.
        #[arg(long)]
        tags: Vec<String>,
    },

    /// List tasks with optional filters.
    List {
        /// Filter by status (open, in_progress, in_review, closed, blocked).
        #[arg(short = 's', long)]
        status: Option<String>,

        /// Filter by type (epic, task, subtask).
        #[arg(short = 't', long)]
        r#type: Option<String>,

        /// Filter by parent key.
        #[arg(short = 'p', long)]
        parent: Option<String>,

        /// Maximum results.
        #[arg(short = 'n', long, default_value_t = 50)]
        limit: u32,

        /// Output format (table, json).
        #[arg(short = 'f', long, default_value = "table")]
        format: String,
    },

    /// Show task details.
    Show {
        /// Task key (e.g. task_abc123).
        key: String,

        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },

    /// Update a task's fields.
    Update {
        /// Task key.
        key: String,

        /// New title.
        #[arg(long)]
        title: Option<String>,

        /// New description.
        #[arg(short = 'd', long)]
        description: Option<String>,

        /// New priority (critical, high, medium, low).
        #[arg(long)]
        priority: Option<String>,

        /// Add tags.
        #[arg(long)]
        add_tags: Vec<String>,

        /// Remove tags.
        #[arg(long)]
        remove_tags: Vec<String>,
    },

    /// Close a task (mark as completed).
    Close {
        /// Task key.
        key: String,

        /// Closing notes.
        #[arg(short = 'm', long)]
        message: Option<String>,
    },

    /// Start working on a task (transition to in_progress).
    Start {
        /// Task key.
        key: String,
    },

    /// Submit a task for review.
    Review {
        /// Task key.
        key: String,

        /// Review notes.
        #[arg(short = 'm', long)]
        message: Option<String>,
    },

    /// Approve a reviewed task.
    Approve {
        /// Task key.
        key: String,

        /// Approve as human (override same-session restriction).
        #[arg(long)]
        human: bool,
    },

    /// Block a task with a reason.
    Block {
        /// Task key.
        key: String,

        /// Blocking reason.
        #[arg(short = 'm', long)]
        message: Option<String>,

        /// Key of the blocking task.
        #[arg(long)]
        blocker: Option<String>,
    },

    /// Unblock a task.
    Unblock {
        /// Task key.
        key: String,
    },

    /// Create a handoff context snapshot.
    Handoff {
        /// Task key.
        key: String,

        /// Handoff notes.
        #[arg(short = 'm', long)]
        message: Option<String>,
    },

    /// Show the latest handoff for a task.
    HandoffShow {
        /// Task key.
        key: String,

        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },

    /// Get rich context for a task (description + history + handoffs).
    Context {
        /// Task key.
        key: String,
    },

    /// Show task activity log.
    Log {
        /// Task key.
        key: String,

        /// Maximum log entries.
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: u32,
    },

    /// List sessions associated with a task.
    Sessions {
        /// Task key.
        key: String,
    },

    /// Manage task dependencies.
    Dep {
        /// Task key.
        key: String,

        /// Add dependency (key of task this depends on).
        #[arg(long)]
        add: Option<String>,

        /// Remove dependency.
        #[arg(long)]
        remove: Option<String>,

        /// Show dependency graph.
        #[arg(long)]
        graph: bool,
    },

    /// Show task system usage statistics.
    Usage,

    /// Integrate task with the knowledge graph.
    GraphIntegration {
        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },
}
