//! Wisdom Tree pipeline.
//!
//! Orchestrates the full round-trip after a wake word fires:
//!
//!   WAKE → play chime → RECORD (capture.py) → THINK (s2s_service) → SPEAK → IDLE
//!
//! Extension points are marked `TODO(pipeline)`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::Mutex;

use crate::AssistantState;

const S2S_SOCKET: &str = "/tmp/s2s.sock";
const CHIME_SAMPLE_RATE: u32 = 44_100;

// ---------------------------------------------------------------------------
// Public entry point — called from handle_event in main.rs
// ---------------------------------------------------------------------------

pub async fn run_pipeline(
    state: Arc<Mutex<AssistantState>>,
    preroll_b64: Option<String>,
    backend: Option<String>,
    cancel: Arc<AtomicBool>,
    last_interaction: Arc<tokio::sync::Mutex<Option<std::time::Instant>>>,
) {
    // ── WAKE ─────────────────────────────────────────────────────────────────
    println!("{}", "🎯 [WAKE]    Button pressed!".yellow().bold());
    set_state(&state, AssistantState::Listening).await;

    // Check whether this is a fresh session (no interaction in the last 5 min).
    let needs_intro = {
        let guard = last_interaction.lock().await;
        match *guard {
            None => true,
            Some(t) => t.elapsed() > Duration::from_secs(300),
        }
    };
    {
        let mut guard = last_interaction.lock().await;
        *guard = Some(std::time::Instant::now());
    }

    // ── RECORD ───────────────────────────────────────────────────────────────
    // Spawn capture.py immediately in the background so Python's startup
    // (importing torch, loading VAD model, opening mic) runs in parallel
    // with the confirmation sound. By the time the user hears the tone and
    // starts speaking, the mic is already open.
    let pr = preroll_b64.clone();
    let capture_task = tokio::spawn(async move { run_capture(pr.as_deref()).await });

    // If this is a new session, play an intro while capture.py is loading.
    if needs_intro {
        if let Ok(intro) = call_s2s_intro(backend.as_deref()).await {
            if let Some(b64) = intro["response_audio_b64"].as_str() {
                let _ = play_wav_b64(b64, Arc::clone(&cancel)).await;
            }
        }
    }

    // Play the "ready to speak" confirmation sound (≈700 ms).
    // User should speak AFTER this tone finishes.
    println!("{}", "🎙  [RECORD]  Speak after the tone…".cyan().bold());
    tokio::task::spawn_blocking(play_confirm_sound).await.ok();

    let pb = spinner("🎙  [RECORD]  Listening…");

    let capture_json = match tokio::time::timeout(
        Duration::from_secs(40),
        async { capture_task.await.context("capture task panicked")? },
    )
    .await
    {
        Ok(Ok(s))  => s,
        Ok(Err(e)) => {
            pb.finish_and_clear();
            eprintln!("capture.py failed: {e:#}");
            spawn_chime(330.0, 200);
            reset_idle(&state).await;
            return;
        }
        Err(_) => {
            pb.finish_and_clear();
            eprintln!("capture.py timed out");
            spawn_chime(330.0, 200);
            reset_idle(&state).await;
            return;
        }
    };
    pb.finish_and_clear();

    let capture: serde_json::Value = match serde_json::from_str(&capture_json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Bad capture JSON: {e}");
            spawn_chime(330.0, 200);
            reset_idle(&state).await;
            return;
        }
    };

    if !capture["speech_detected"].as_bool().unwrap_or(false) {
        println!("{}", "  (no speech detected)".dimmed());
        spawn_chime(330.0, 150);
        reset_idle(&state).await;
        return;
    }

    let audio_b64 = match capture["audio_b64"].as_str() {
        Some(s) => s.to_owned(),
        None => {
            eprintln!("capture JSON missing audio_b64");
            reset_idle(&state).await;
            return;
        }
    };

    let duration_s = capture["duration_ms"].as_u64().unwrap_or(0) as f64 / 1000.0;
    println!("{}", format!("  Captured {duration_s:.1}s of speech").dimmed());

    // ── THINK ─────────────────────────────────────────────────────────────────
    set_state(&state, AssistantState::Processing).await;
    let pb = spinner("🧠 [THINK]   Processing…");

    // Play soothing hold music on a background thread while the LLM works.
    let thinking = Arc::new(AtomicBool::new(true));
    let thinking_flag = Arc::clone(&thinking);
    let cancel_for_hold = Arc::clone(&cancel);
    let hold_thread = std::thread::spawn(move || play_hold_music(thinking_flag, cancel_for_hold));

    let result = tokio::time::timeout(Duration::from_secs(90), call_s2s(&audio_b64, backend.as_deref())).await;

    // Stop hold music before playing the response.
    thinking.store(false, Ordering::Relaxed);
    hold_thread.join().ok();
    pb.finish_and_clear();

    let response = match result {
        Ok(Ok(r))  => r,
        Ok(Err(e)) => { eprintln!("S2S error: {e:#}");    spawn_chime(330.0, 200); reset_idle(&state).await; return; }
        Err(_)     => { eprintln!("S2S request timed out"); spawn_chime(330.0, 200); reset_idle(&state).await; return; }
    };

    // Check for pipeline-level errors returned by the service
    if let Some(err) = response["error"].as_str() {
        eprintln!("S2S service returned error: {err}");
        // Still try to play the error audio if provided
    }

    // ── SPEAK ─────────────────────────────────────────────────────────────────
    if cancel.load(Ordering::Relaxed) {
        reset_idle(&state).await;
        return;
    }

    println!("{}", "🔊 [SPEAK]   Playing response…".magenta().bold());
    spawn_chime(523.0, 80);

    if let Some(audio_b64) = response["response_audio_b64"].as_str() {
        if let Err(e) = play_wav_b64(audio_b64, Arc::clone(&cancel)).await {
            eprintln!("Playback error: {e}");
        }
    }

    // ── DONE ──────────────────────────────────────────────────────────────────
    if let Some(lat) = response.get("latency_ms") {
        println!(
            "{}",
            format!(
                "  ⏱  stt={}ms  llm={}ms  tts={}ms  total={}ms",
                lat["stt"].as_u64().unwrap_or(0),
                lat["llm"].as_u64().unwrap_or(0),
                lat["tts"].as_u64().unwrap_or(0),
                lat["total"].as_u64().unwrap_or(0),
            )
            .dimmed()
        );
    }

    if let Some(tx) = response["transcript"].as_str() {
        println!("{} {}", "✅ [DONE]".green().bold(), format!("You: {tx}").white());
    }
    if let Some(reply) = response["response_text"].as_str() {
        println!("             {}", format!("Chimpoo: {reply}").cyan());
    }

    reset_idle(&state).await;
}

// ---------------------------------------------------------------------------
// Subprocess: capture.py
// ---------------------------------------------------------------------------

async fn run_capture(preroll_b64: Option<&str>) -> Result<String> {
    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg("capture.py");
    if let Some(pr) = preroll_b64 {
        cmd.arg("--preroll").arg(pr);
    }
    let out = cmd
        .output()
        .await
        .context("Failed to spawn capture.py — is it in the working directory?")?;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("capture.py exited non-zero:\n{err}");
    }

    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

// ---------------------------------------------------------------------------
// Unix socket: /tmp/s2s.sock
// ---------------------------------------------------------------------------

async fn call_s2s(audio_b64: &str, backend: Option<&str>) -> Result<serde_json::Value> {
    let stream = UnixStream::connect(S2S_SOCKET)
        .await
        .context("Cannot reach /tmp/s2s.sock — is s2s_service.py running?")?;

    let (reader, mut writer) = stream.into_split();

    let payload = serde_json::json!({
        "event":     "process_audio",
        "audio_b64": audio_b64,
        "backend":   backend.unwrap_or("anythingllm"),
    });
    let mut bytes = serde_json::to_vec(&payload)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    buf_reader.read_line(&mut line).await?;

    Ok(serde_json::from_str(line.trim())
        .context("Invalid JSON response from s2s_service")?)
}

async fn call_s2s_intro(backend: Option<&str>) -> Result<serde_json::Value> {
    let stream = UnixStream::connect(S2S_SOCKET)
        .await
        .context("Cannot reach /tmp/s2s.sock — is s2s_service.py running?")?;

    let (reader, mut writer) = stream.into_split();

    let payload = serde_json::json!({
        "event":   "intro",
        "backend": backend.unwrap_or("anythingllm"),
    });
    let mut bytes = serde_json::to_vec(&payload)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await?;

    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    buf_reader.read_line(&mut line).await?;

    Ok(serde_json::from_str(line.trim())
        .context("Invalid JSON response from s2s_service (intro)")?)
}

// ---------------------------------------------------------------------------
// Audio: chime (generated) and WAV playback (rodio)
// ---------------------------------------------------------------------------

/// Two-tone ascending "ready to speak" confirmation sound (~700 ms total).
/// Played synchronously so the caller can await it as a cue.
fn play_confirm_sound() {
    let Ok((_stream, handle)) = rodio::OutputStream::try_default() else { return };
    let sr = CHIME_SAMPLE_RATE;

    // Low ding → high ding, like an elevator "doors open" chime.
    for &(freq, dur_ms, vol) in &[(523.0f32, 250u64, 0.28f32), (784.0, 350, 0.22)] {
        let Ok(sink) = rodio::Sink::try_new(&handle) else { return };
        sink.set_volume(vol);
        let n = (sr * dur_ms as u32 / 1_000) as usize;
        let dur_s = dur_ms as f32 / 1_000.0;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let env = (-5.0 * t / dur_s).exp();
                let wave = (2.0 * std::f32::consts::PI * freq * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * freq * 2.0 * t).sin();
                env * wave
            })
            .collect();
        sink.append(rodio::buffer::SamplesBuffer::new(1, sr, samples));
        sink.sleep_until_end();
        std::thread::sleep(Duration::from_millis(80));
    }
}

/// Soothing arpeggiated bell loop played while the LLM is processing.
///
/// Plays a gentle A-C#-E major triad arpeggio (bell-like with exponential
/// decay + soft octave harmonic) and loops until `running` goes false.
fn play_hold_music(running: Arc<AtomicBool>, cancel: Arc<AtomicBool>) {
    let Ok((_stream, handle)) = rodio::OutputStream::try_default() else { return };
    let sr = CHIME_SAMPLE_RATE;
    // Major triad: A4, C#5, E5 — calm and resolved-sounding
    let notes: &[f32] = &[440.0, 523.25, 659.25];
    let note_ms: u64 = 480;
    let gap_ms:  u64 = 120;
    let rest_ms: u64 = 900;

    'outer: loop {
        for &freq in notes {
            if !running.load(Ordering::Relaxed) || cancel.load(Ordering::Relaxed) { break 'outer; }

            let Ok(sink) = rodio::Sink::try_new(&handle) else { break 'outer; };
            sink.set_volume(0.11);

            let n = (sr * note_ms as u32 / 1_000) as usize;
            let dur_s = note_ms as f32 / 1_000.0;
            let samples: Vec<f32> = (0..n)
                .map(|i| {
                    let t = i as f32 / sr as f32;
                    // Bell envelope: instant attack, exponential decay
                    let env = (-5.5 * t / dur_s).exp();
                    // Fundamental + soft octave gives a bell-like timbre
                    let wave = (2.0 * std::f32::consts::PI * freq * t).sin()
                        + 0.25 * (2.0 * std::f32::consts::PI * freq * 2.0 * t).sin();
                    env * wave
                })
                .collect();

            sink.append(rodio::buffer::SamplesBuffer::new(1, sr, samples));
            sink.sleep_until_end();
            std::thread::sleep(Duration::from_millis(gap_ms));
        }
        if !running.load(Ordering::Relaxed) || cancel.load(Ordering::Relaxed) { break; }
        std::thread::sleep(Duration::from_millis(rest_ms));
    }
}

/// Fire-and-forget chime on a blocking thread so we don't hold the executor.
fn spawn_chime(freq: f32, duration_ms: u64) {
    std::thread::spawn(move || play_chime_sync(freq, duration_ms));
}

fn play_chime_sync(freq: f32, duration_ms: u64) {
    let Ok((_stream, handle)) = rodio::OutputStream::try_default() else { return };
    let Ok(sink) = rodio::Sink::try_new(&handle) else { return };

    let n = (CHIME_SAMPLE_RATE * duration_ms as u32 / 1_000) as usize;
    let fade = (CHIME_SAMPLE_RATE as usize / 100).max(1).min(n / 4); // 10 ms
    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let t  = i as f32 / CHIME_SAMPLE_RATE as f32;
            let fi = (i as f32 / fade as f32).min(1.0);
            let fo = ((n - i) as f32 / fade as f32).min(1.0);
            0.25 * fi * fo * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    sink.append(rodio::buffer::SamplesBuffer::new(1, CHIME_SAMPLE_RATE, samples));
    sink.sleep_until_end();
}

async fn play_wav_b64(audio_b64: &str, cancel: Arc<AtomicBool>) -> Result<()> {
    let bytes = B64.decode(audio_b64).context("base64 decode failed")?;
    tokio::task::spawn_blocking(move || {
        let cursor = std::io::Cursor::new(bytes);
        let Ok((_stream, handle)) = rodio::OutputStream::try_default() else {
            return Err(anyhow::anyhow!("No audio output device"));
        };
        let sink = rodio::Sink::try_new(&handle).context("Sink creation failed")?;
        let source = rodio::Decoder::new(cursor).context("WAV decode failed")?;
        sink.append(source);
        while !sink.empty() {
            if cancel.load(Ordering::Relaxed) {
                sink.stop();
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Ok(())
    })
    .await
    .context("spawn_blocking panicked")?
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.set_message(msg.to_owned());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

async fn set_state(state: &Arc<Mutex<AssistantState>>, s: AssistantState) {
    *state.lock().await = s;
}

async fn reset_idle(state: &Arc<Mutex<AssistantState>>) {
    set_state(state, AssistantState::Idle).await;
    println!("{}", "⏸  [IDLE]    Press play/pause to activate…".dimmed());
}
