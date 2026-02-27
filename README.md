# Wisdom Tree

A fully local, offline-first voice assistant for macOS (Apple Silicon). Press the play/pause button on a Satechi remote to activate — no cloud, no subscription, no data leaves your machine.

<img width="2752" height="1536" alt="wisdom-tree" src="https://github.com/user-attachments/assets/2494f650-8f6c-489f-a21e-453598962e48" />


## What it does

Press a button → speak → get a spoken response. The full pipeline:

1. **Activation** — play/pause key on a Satechi remote (or keypad `+`) triggers the assistant
2. **Chime** — a two-tone confirmation sound plays so you know when to speak
3. **STT** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcribes your speech locally
4. **LLM** — your query is sent to [AnythingLLM](https://anythingllm.com/) running at `localhost:3001`
5. **TTS** — [Kokoro](https://github.com/hexgrad/kokoro) synthesizes the response into speech
6. **Playback** — the response audio plays back through your default audio output

Press play/pause again while the assistant is thinking to interrupt it.

## Architecture

```
[Satechi Remote] → play/pause key
        ↓
keypress_listener.py   (Python — macOS Quartz CGEventTap, consumes the key event)
        ↓  JSON over /tmp/wakeword.sock
wisdom-tree            (Rust daemon — src/main.rs + src/pipeline.rs)
        ↓  spawns
capture.py             (records mic audio until silence, returns base64 WAV)
        ↓  audio_b64 over /tmp/s2s.sock
s2s_service.py         (Python — Whisper STT → AnythingLLM → Kokoro TTS)
        ↓
rodio (Rust)           (plays back the response audio)
```

The Rust daemon manages state (Idle → Listening → Processing → Idle) and orchestrates the pipeline. Python processes communicate with it via Unix domain sockets.

## Hardware Requirements

- Mac with Apple Silicon (M1 / M2 / M3 / M4)
- [Satechi Bluetooth remote](https://satechi.net/) with a play/pause button, **or** any keyboard with media keys / a keypad `+` key
- A microphone (built-in or external)

## Prerequisites

**System packages**

```bash
brew install portaudio ffmpeg
```

**Rust toolchain**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**[AnythingLLM](https://anythingllm.com/) desktop app** — must be running with at least one workspace configured before you start the assistant.

**macOS permissions** — the keypress listener intercepts media keys at the system level and requires **Input Monitoring** access:

> System Settings → Privacy & Security → Input Monitoring → enable for your terminal (Terminal, iTerm2, Ghostty, etc.)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/wisdom-tree.git
cd wisdom-tree

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt -r requirements_s2s.txt
```

On first run, the following models are downloaded automatically:

| Model | Size | Purpose |
|---|---|---|
| faster-whisper `medium.en` | ~780 MB | Speech-to-text |
| Kokoro TTS weights | ~350 MB | Text-to-speech |
| Silero VAD | ~2 MB | End-of-speech detection |

Use `--whisper-model large-v3` (~1.5 GB) for higher transcription accuracy.

## Configuration

The two required settings are your AnythingLLM API key and the URL of the machine running AnythingLLM. Both can be set as environment variables (add to `~/.zshrc` or `~/.bash_profile` to make permanent) or passed as flags at runtime.

| Environment variable | Flag | Default | Description |
|---|---|---|---|
| `ANYTHINGLLM_API_KEY` | `--api-key` | _(empty)_ | API key from AnythingLLM → Settings → API Keys |
| `ANYTHINGLLM_URL` | `--anythingllm-url` | `http://localhost:3001` | Base URL of your AnythingLLM instance |

**Local machine (AnythingLLM running on the same computer):**

```bash
export ANYTHINGLLM_API_KEY="your-api-key-here"
./run_all.sh --workspace my-workspace
```

**Remote machine (AnythingLLM running on another machine on your network):**

```bash
export ANYTHINGLLM_API_KEY="your-api-key-here"
export ANYTHINGLLM_URL="http://192.168.1.x:3001"
./run_all.sh --workspace my-workspace
```

Or pass everything inline:

```bash
./run_all.sh --workspace my-workspace --api-key your-key --anythingllm-url http://192.168.1.x:3001
```

> **Note:** When connecting to AnythingLLM on another machine, make sure it is bound to `0.0.0.0` (not just `localhost`) so it accepts connections from the network. Check AnythingLLM's server settings if requests are refused.

## Usage

Make sure AnythingLLM is running first, then:

```bash
./run_all.sh --workspace your-workspace-slug
```

All services start in the correct order. Press **Ctrl-C** to shut everything down cleanly.

**Common options:**

```
--workspace        AnythingLLM workspace slug (required)
--whisper-model    STT model: tiny.en, base.en, medium.en (default), large-v3
--voice            Kokoro voice ID (default: af_sky)
--timeout          LLM request timeout in seconds (default: 90)
--cpu-only         Force CPU for Whisper (use if MPS causes issues)
```

**Example with a larger Whisper model:**

```bash
./run_all.sh --workspace my-workspace --whisper-model large-v3
```

### Latency

After each response, timing is printed to the console:

```
⏱  stt=210ms  llm=1840ms  tts=95ms  total=2145ms
```

LLM latency dominates and depends on your model and hardware.

## Persona

The assistant is named **Chimpoo** — concise, natural spoken language, no markdown or bullet points, responses under 3 sentences by default.

Customize the persona by editing the `SYSTEM_PROMPT` constant in `s2s_service.py`.

## Files

| File | Purpose |
|---|---|
| `src/main.rs` | Rust daemon — state machine, Unix socket server |
| `src/pipeline.rs` | Rust — full STT→LLM→TTS round-trip orchestration |
| `keypress_listener.py` | Intercepts play/pause via Quartz CGEventTap |
| `capture.py` | VAD-gated mic recording → base64 WAV |
| `s2s_service.py` | Speech-to-Speech pipeline service |
| `wakeword_listener.py` | Alternative activation via wake word model (OpenWakeWord / custom ONNX) |
| `detect_key.py` | Debug utility — prints raw key events from your remote |
| `run_all.sh` | Start all services in order |
| `run.sh` | Start daemon + keypress listener only (no S2S service) |

## Troubleshooting

**"Daemon not reachable"** — the Rust daemon must be running before `keypress_listener.py` starts. Use `run_all.sh` which handles ordering automatically.

**No audio output** — check your default audio output in System Settings → Sound.

**AnythingLLM errors** — verify AnythingLLM is running at `localhost:3001`, the workspace slug matches, and your API key is correct.

**Key not intercepted** — grant Input Monitoring permission to your terminal app and restart it.

**Whisper running slowly** — pass `--cpu-only` if MPS is causing issues, or use a smaller model (`base.en`).

## License

MIT — see [LICENSE](LICENSE).
