//! Wisdom Tree daemon — wake word listener + pipeline orchestrator.
//!
//! Receives wake word events from the Python sidecar over /tmp/wakeword.sock,
//! then hands off to `pipeline::run_pipeline` which drives the full
//! STT → LLM → TTS round-trip.

mod pipeline;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Context;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

const SOCKET_PATH: &str = "/tmp/wakeword.sock";

// ---------------------------------------------------------------------------
// State machine  (pub(crate) so pipeline.rs can use it)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssistantState {
    Idle,
    Listening,
    Processing,
}

impl std::fmt::Display for AssistantState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssistantState::Idle       => write!(f, "Idle"),
            AssistantState::Listening  => write!(f, "Listening"),
            AssistantState::Processing => write!(f, "Processing"),
        }
    }
}

// ---------------------------------------------------------------------------
// Event wire format (matches Python sidecar JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WakeWordEvent {
    event:       String,
    model:       String,
    score:       f64,
    #[allow(dead_code)]
    timestamp:   u64,
    /// First ~480 ms of audio captured immediately after detection.
    /// Prevents missing words spoken before capture.py opens the mic.
    preroll_b64: Option<String>,
    /// Which LLM backend to use ("anythingllm" or "gemini").
    backend: Option<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "wisdom_tree=debug,info".into()),
        )
        .init();

    info!("Wisdom Tree daemon starting");

    // Remove stale socket from previous crash.
    let socket_path = Path::new(SOCKET_PATH);
    if socket_path.exists() {
        std::fs::remove_file(socket_path).context("Failed to remove stale socket")?;
    }

    let listener = UnixListener::bind(SOCKET_PATH).context("Failed to bind Unix socket")?;
    info!("Listening on {SOCKET_PATH}");

    let state: Arc<Mutex<AssistantState>> = Arc::new(Mutex::new(AssistantState::Idle));
    let cancel: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let last_interaction: Arc<Mutex<Option<std::time::Instant>>> = Arc::new(Mutex::new(None));

    // Print initial state
    use colored::Colorize;
    println!("{}", "⏸  [IDLE]    Press play/pause to activate…".dimmed());

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, _)) => {
                        info!("Sidecar connected");
                        let state = Arc::clone(&state);
                        let cancel = Arc::clone(&cancel);
                        let last_interaction = Arc::clone(&last_interaction);
                        tokio::spawn(async move {
                            handle_connection(stream, state, cancel, last_interaction).await;
                        });
                    }
                    Err(e) => error!("Accept error: {e}"),
                }
            }
            _ = signal::ctrl_c() => {
                info!("Ctrl-C — shutting down");
                break;
            }
        }
    }

    let _ = std::fs::remove_file(SOCKET_PATH);
    info!("Goodbye.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-connection reader
// ---------------------------------------------------------------------------

async fn handle_connection(stream: UnixStream, state: Arc<Mutex<AssistantState>>, cancel: Arc<AtomicBool>, last_interaction: Arc<Mutex<Option<std::time::Instant>>>) {
    let mut lines = BufReader::new(stream).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                debug!("Raw: {line}");
                match serde_json::from_str::<WakeWordEvent>(&line) {
                    Ok(ev)  => handle_event(ev, Arc::clone(&state), Arc::clone(&cancel), Arc::clone(&last_interaction)).await,
                    Err(e)  => warn!("Parse error: {e} — raw: {line:?}"),
                }
            }
            Ok(None) => { info!("Sidecar disconnected (EOF)"); break; }
            Err(e)   => { error!("Read error: {e}"); break; }
        }
    }
}

// ---------------------------------------------------------------------------
// State machine gate — only starts a pipeline run when Idle
// ---------------------------------------------------------------------------

async fn handle_event(event: WakeWordEvent, state: Arc<Mutex<AssistantState>>, cancel: Arc<AtomicBool>, last_interaction: Arc<Mutex<Option<std::time::Instant>>>) {
    use colored::Colorize;

    if event.event != "detected" {
        debug!("Unknown event type: {}", event.event);
        return;
    }

    {
        let guard = state.lock().await;
        match *guard {
            AssistantState::Idle => {
                info!(
                    model = %event.model,
                    score = event.score,
                    "Wake word detected — launching pipeline"
                );
                // Drop lock before entering the pipeline
            }
            AssistantState::Processing => {
                info!("Button pressed during processing — cancelling");
                println!("{}", "⛔ [STOP]    Interrupted.".red().bold());
                cancel.store(true, Ordering::Relaxed);
                return;
            }
            ref s => {
                debug!("Ignoring detection — state is {s}");
                return;
            }
        }
    }

    cancel.store(false, Ordering::Relaxed);
    tokio::spawn(pipeline::run_pipeline(state, event.preroll_b64, event.backend, cancel, last_interaction));
}
