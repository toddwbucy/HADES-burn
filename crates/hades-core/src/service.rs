//! Transport-agnostic daemon request service.
//!
//! Everything between "raw request payload" and "response envelope" lives
//! here: parsing, session resolution, access-tier authorization, dispatch,
//! and error mapping. Transports (the Unix-socket listener today, a network
//! endpoint tomorrow) own only framing and connection lifecycle, and hand
//! each payload to [`handle_request`] together with the [`ConnectionPolicy`]
//! their trust boundary implies.
//!
//! The session tier is a property of the transport, not the request. The
//! policy sets a ceiling; a client-supplied `session` field may restrict a
//! connection below that ceiling but can never raise it. Payload bytes are
//! attacker-controlled on a network transport, so nothing parsed from them
//! participates in granting authority.

use std::time::Duration;

use serde_json::Value;
use tokio::time::timeout;

use crate::config::HadesConfig;
use crate::db::ArangoPool;
use crate::dispatch::{
    self, AccessTier, DaemonCommand, DaemonResponse, DispatchError, HandlerError,
};

// ---------------------------------------------------------------------------
// Session policy
// ---------------------------------------------------------------------------

/// Whether a session is an AI agent (restricted) or admin (unrestricted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    /// Unrestricted — human operator on a trusted local transport.
    Admin,
    /// Restricted — limited to [`AccessTier::Agent`] commands.
    Agent,
}

/// The authority ceiling a transport grants its connections.
///
/// Constructed by the transport, never from request bytes. A request may
/// self-restrict below the ceiling via its `session` field; requesting a
/// tier above the ceiling is an error, not a silent clamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectionPolicy {
    /// Highest tier this connection may operate at.
    pub max_tier: SessionKind,
}

impl ConnectionPolicy {
    /// Policy for trusted local transports (Unix socket peers): admin
    /// ceiling, admin default — the daemon's historical local behavior.
    pub fn local_admin() -> Self {
        Self {
            max_tier: SessionKind::Admin,
        }
    }

    /// Policy for network transports: agent ceiling. A request claiming
    /// an admin session is rejected outright.
    pub fn agent_only() -> Self {
        Self {
            max_tier: SessionKind::Agent,
        }
    }
}

/// Resolve the effective session tier from the transport policy and the
/// request's optional self-declared session.
///
/// Absent → the policy ceiling (admin locally, agent on the network).
/// `"agent"` → agent, always honored (self-restriction).
/// `"admin"` → admin only when the ceiling allows it; otherwise
/// `ACCESS_DENIED`, because silently downgrading an explicit claim would
/// misreport what tier the commands then run at.
pub fn resolve_session(
    policy: ConnectionPolicy,
    requested: Option<SessionKind>,
    request_id: &Option<String>,
) -> Result<SessionKind, DaemonResponse> {
    match requested {
        None => Ok(policy.max_tier),
        Some(SessionKind::Agent) => Ok(SessionKind::Agent),
        Some(SessionKind::Admin) => match policy.max_tier {
            SessionKind::Admin => Ok(SessionKind::Admin),
            SessionKind::Agent => Err(DaemonResponse::err(
                "ACCESS_DENIED",
                "transport policy caps this connection at agent tier; \
                 admin session is not available on this endpoint",
            )
            .with_request_id(request_id.clone())),
        },
    }
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

/// A parsed request frame, before authorization.
#[derive(Debug)]
pub struct ParsedRequest {
    /// Client-chosen correlation id, echoed on every response.
    pub request_id: Option<String>,
    /// Self-declared session from the frame, if present. Advisory only —
    /// [`resolve_session`] decides what it is worth.
    pub requested_session: Option<SessionKind>,
    /// The decoded command.
    pub command: DaemonCommand,
}

/// Parse a raw JSON payload into a [`ParsedRequest`].
///
/// Returns `Err(DaemonResponse)` on failure — the caller sends it directly
/// to the client.
pub fn parse_request(payload: &[u8]) -> Result<ParsedRequest, DaemonResponse> {
    let frame: Value = serde_json::from_slice(payload)
        .map_err(|e| DaemonResponse::err("MALFORMED_JSON", e.to_string()))?;

    let request_id = frame
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    let requested_session = match frame.get("session").and_then(|v| v.as_str()) {
        Some("agent") => Some(SessionKind::Agent),
        Some("admin") => Some(SessionKind::Admin),
        None => None,
        Some(other) => {
            return Err(DaemonResponse::err(
                "INVALID_SESSION",
                format!("unknown session type '{other}'; expected \"agent\" or \"admin\""),
            )
            .with_request_id(request_id));
        }
    };

    let command: DaemonCommand = serde_json::from_value(frame).map_err(|e| {
        DaemonResponse::err("UNKNOWN_COMMAND", e.to_string()).with_request_id(request_id.clone())
    })?;

    Ok(ParsedRequest {
        request_id,
        requested_session,
        command,
    })
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

/// Handle one request payload end to end: parse, resolve the session tier
/// against the transport policy, enforce access tiers, dispatch with a
/// timeout, and map errors to the protocol envelope.
///
/// Always returns a [`DaemonResponse`]; transport-level concerns (framing,
/// payload size limits, idle timeouts) belong to the caller.
pub async fn handle_request(
    pool: &ArangoPool,
    config: &HadesConfig,
    policy: ConnectionPolicy,
    payload: &[u8],
    request_timeout: Duration,
) -> DaemonResponse {
    let parsed = match parse_request(payload) {
        Ok(parsed) => parsed,
        Err(resp) => return resp,
    };

    let session = match resolve_session(policy, parsed.requested_session, &parsed.request_id) {
        Ok(session) => session,
        Err(resp) => return resp,
    };

    let request_id = parsed.request_id;
    let cmd = parsed.command;

    // Enforce access tier: agent sessions may only invoke Agent-tier commands.
    if session == SessionKind::Agent && cmd.access_tier() != AccessTier::Agent {
        return DaemonResponse::err(
            "ACCESS_DENIED",
            format!(
                "agent session cannot invoke {:?}-tier command",
                cmd.access_tier(),
            ),
        )
        .with_request_id(request_id);
    }

    // Preserve command payload for NotImplemented subprocess fallback.
    let cmd_value = serde_json::to_value(&cmd).ok();

    match timeout(request_timeout, dispatch::dispatch(pool, config, cmd)).await {
        Ok(Ok(data)) => DaemonResponse::ok(data).with_request_id(request_id),
        Ok(Err(DispatchError::NotImplemented(name))) => {
            let mut resp = DaemonResponse::err(
                "NOT_IMPLEMENTED",
                format!("command '{name}' not yet implemented natively"),
            );
            resp.data = cmd_value;
            resp.with_request_id(request_id)
        }
        Ok(Err(DispatchError::Handler(e))) => {
            DaemonResponse::err(handler_error_code(&e), e.to_string()).with_request_id(request_id)
        }
        Err(_) => DaemonResponse::err("INTERNAL", "request timed out").with_request_id(request_id),
    }
}

/// Map a [`HandlerError`] to a protocol error code string.
pub fn handler_error_code(e: &HandlerError) -> &'static str {
    match e {
        HandlerError::InvalidNodeId { .. }
        | HandlerError::InvalidLimit { .. }
        | HandlerError::InvalidParameter { .. } => "INVALID_PARAMS",
        HandlerError::NodeNotFound(_) | HandlerError::DocumentNotFound { .. } => "NOT_FOUND",
        HandlerError::NoEmbedding { .. } | HandlerError::InvalidEmbedding { .. } => "QUERY_FAILED",
        HandlerError::Query { .. } => "QUERY_FAILED",
        HandlerError::ServiceError(_) => "SERVICE_ERROR",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handler_error_code_mapping() {
        assert_eq!(
            handler_error_code(&HandlerError::NodeNotFound("x".into())),
            "NOT_FOUND",
        );
        assert_eq!(
            handler_error_code(&HandlerError::InvalidNodeId {
                node_id: "x".into(),
                reason: "bad".into(),
            }),
            "INVALID_PARAMS",
        );
        assert_eq!(
            handler_error_code(&HandlerError::InvalidLimit {
                limit: 0,
                max: 1000
            }),
            "INVALID_PARAMS",
        );
        assert_eq!(
            handler_error_code(&HandlerError::NoEmbedding {
                node_id: "x".into(),
            }),
            "QUERY_FAILED",
        );
    }

    #[test]
    fn parse_rejects_malformed_json() {
        let resp = parse_request(b"not json at all").unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("MALFORMED_JSON"));
    }

    #[test]
    fn parse_rejects_unknown_command() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "command": "nonexistent.cmd",
            "params": {}
        }))
        .unwrap();
        let resp = parse_request(&payload).unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("UNKNOWN_COMMAND"));
    }

    #[test]
    fn parse_echoes_request_id_on_error() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "request_id": "req-42",
            "command": "nonexistent.cmd",
            "params": {}
        }))
        .unwrap();
        let resp = parse_request(&payload).unwrap_err();
        assert_eq!(resp.request_id.as_deref(), Some("req-42"));
    }

    #[test]
    fn parse_leaves_absent_session_unresolved() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "request_id": "req-1",
            "command": "orient",
            "params": {}
        }))
        .unwrap();
        let parsed = parse_request(&payload).unwrap();
        assert_eq!(parsed.request_id.as_deref(), Some("req-1"));
        // Absent session is None — the transport policy decides the tier,
        // not the parser.
        assert_eq!(parsed.requested_session, None);
    }

    #[test]
    fn parse_reads_explicit_agent_session() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "session": "agent",
            "command": "orient",
            "params": {}
        }))
        .unwrap();
        let parsed = parse_request(&payload).unwrap();
        assert_eq!(parsed.requested_session, Some(SessionKind::Agent));
    }

    #[test]
    fn parse_rejects_unknown_session() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "session": "superuser",
            "command": "orient",
            "params": {}
        }))
        .unwrap();
        let resp = parse_request(&payload).unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("INVALID_SESSION"));
    }

    // --- resolve_session: the A1 matrix -----------------------------------

    #[test]
    fn local_admin_policy_defaults_to_admin() {
        // Historical local behavior: absent session on the Unix socket → admin.
        let tier = resolve_session(ConnectionPolicy::local_admin(), None, &None).unwrap();
        assert_eq!(tier, SessionKind::Admin);
    }

    #[test]
    fn local_admin_policy_honors_self_restriction() {
        let tier = resolve_session(
            ConnectionPolicy::local_admin(),
            Some(SessionKind::Agent),
            &None,
        )
        .unwrap();
        assert_eq!(tier, SessionKind::Agent);
    }

    #[test]
    fn agent_only_policy_defaults_to_agent() {
        // Absent session on a network transport → agent, never admin.
        let tier = resolve_session(ConnectionPolicy::agent_only(), None, &None).unwrap();
        assert_eq!(tier, SessionKind::Agent);
    }

    #[test]
    fn agent_only_policy_rejects_admin_claim() {
        // The A1 fix: request bytes cannot elevate above the transport ceiling.
        let resp = resolve_session(
            ConnectionPolicy::agent_only(),
            Some(SessionKind::Admin),
            &Some("req-9".into()),
        )
        .unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("ACCESS_DENIED"));
        assert_eq!(resp.request_id.as_deref(), Some("req-9"));
    }

    #[test]
    fn agent_only_policy_accepts_explicit_agent() {
        let tier = resolve_session(
            ConnectionPolicy::agent_only(),
            Some(SessionKind::Agent),
            &None,
        )
        .unwrap();
        assert_eq!(tier, SessionKind::Agent);
    }

    // --- handle_request: authorization before dispatch --------------------

    #[tokio::test]
    async fn agent_session_blocked_from_admin_command() {
        // Tier check fires before dispatch — no live DB needed.
        let mut config = HadesConfig::default();
        config.database.name = Some("bident_burn".to_string());
        let pool = ArangoPool::from_config(&config).unwrap();

        let payload = serde_json::to_vec(&serde_json::json!({
            "session": "agent",
            "command": "db.aql",
            "params": { "aql": "RETURN 1" }
        }))
        .unwrap();

        let resp = handle_request(
            &pool,
            &config,
            ConnectionPolicy::local_admin(),
            &payload,
            Duration::from_secs(5),
        )
        .await;

        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("ACCESS_DENIED"));
    }

    #[tokio::test]
    async fn agent_only_policy_blocks_admin_command_without_session_field() {
        // The frame carries no session at all — under an agent-only policy
        // the admin-tier command must still be denied. This is the exact
        // payload that reached dispatch as Admin before the A1 fix.
        let mut config = HadesConfig::default();
        config.database.name = Some("bident_burn".to_string());
        let pool = ArangoPool::from_config(&config).unwrap();

        let payload = serde_json::to_vec(&serde_json::json!({
            "command": "db.aql",
            "params": { "aql": "RETURN 1" }
        }))
        .unwrap();

        let resp = handle_request(
            &pool,
            &config,
            ConnectionPolicy::agent_only(),
            &payload,
            Duration::from_secs(5),
        )
        .await;

        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("ACCESS_DENIED"));
    }
}
