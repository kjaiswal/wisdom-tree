# Wisdom Tree — "Chimpoo"

A local, offline-first voice assistant running on macOS (Apple Silicon). Activated by pressing the play/pause button on a Satechi remote.

## Architecture

```
[Satechi Remote] → play/pause key
        ↓
keypress_listener.py   (Python, macOS Quartz event tap)
        ↓  JSON over /tmp/wakeword.sock
wisdom-tree            (Rust daemon — src/main.rs + src/pipeline.rs)
        ↓  spawns
capture.py             (records mic audio until silence)
        ↓  audio_b64 over /tmp/s2s.sock
s2s_service.py         (Python — STT → LLM → TTS)
        ↓
rodio (Rust)           (plays back audio response)
```

## Services

| Component | Language | Role |
|---|---|---|
| `keypress_listener.py` | Python | Intercepts media play/pause via Quartz CGEventTap (consuming it), sends "detected" event to Rust daemon over `/tmp/wakeword.sock` |
| `src/main.rs` | Rust/Tokio | State machine (Idle → Listening → Processing → Idle), listens on `/tmp/wakeword.sock`, spawns pipeline |
| `src/pipeline.rs` | Rust | Orchestrates full round-trip: plays chimes, runs `capture.py`, calls s2s service, plays response |
| `capture.py` | Python | Opens mic, records until VAD detects silence, returns base64 WAV via stdout |
| `s2s_service.py` | Python | STT (faster-whisper) → LLM (LM Studio) → TTS (Kokoro), listens on `/tmp/s2s.sock` |

## Pipeline Flow (per activation)

1. Play/pause pressed → JSON event sent to Rust daemon
2. Rust plays a two-tone "ready" chime and spawns `capture.py`
3. `capture.py` records until silence, returns base64 WAV
4. Rust sends audio to `s2s_service.py` over Unix socket
5. Python runs Whisper STT → LM Studio LLM → Kokoro TTS
6. Rust receives response audio and plays it back via rodio
7. Latency breakdown printed (stt/llm/tts/total ms)

## Key Design Decisions

- **All local**: Whisper (STT), AnythingLLM at `localhost:3001` (LLM), Kokoro (TTS) — no cloud
- **Unix sockets** for IPC between Rust and Python processes
- **Preroll audio**: ~480ms of audio captured at wake time passed to `capture.py` so words spoken immediately aren't missed
- **Conversation history**: last 6 turns retained in `s2s_service.py`
- **Hold music**: soft A-C#-E bell arpeggio plays while LLM is processing
- **DeepSeek-R1 / reasoning model support**: strips `</think>` blocks from LLM output
- **Event tap consumption**: play/pause key is consumed at the Quartz level so it doesn't reach other apps (e.g. Apple Music)

## Running

```bash
./run_all.sh [--whisper-model large-v3]
```

Start order: AnythingLLM must be running → `s2s_service.py` loads models → Rust daemon starts → `keypress_listener.py` starts.

Pass your AnythingLLM API key and workspace slug:
```bash
./run_all.sh --api-key YOUR_KEY --workspace your-workspace-slug
```

## Dependencies

- **Rust**: tokio, rodio, serde_json, indicatif, colored, base64, anyhow
- **Python (requirements.txt)**: openwakeword, pyaudio, numpy, onnxruntime
- **Python (requirements_s2s.txt)**: faster-whisper, kokoro, openai, soundfile, sounddevice, numpy
- **System**: `brew install portaudio`, AnythingLLM running with at least one workspace configured

## IPC Sockets

| Socket | Owner | Protocol |
|---|---|---|
| `/tmp/wakeword.sock` | Rust daemon | Newline-delimited JSON, `{"event": "detected", "model": "...", "score": 1.0, "timestamp": ms, "preroll_b64": null}` |
| `/tmp/s2s.sock` | `s2s_service.py` | Newline-delimited JSON request/response |

## Assistant Persona

System prompt: **Chimpoo** — concise, natural spoken language only, no markdown or bullet points, responses under 3 sentences unless asked for more.
