#!/usr/bin/env python3
"""
Speech-to-Speech service.

Listens on /tmp/s2s.sock for newline-delimited JSON requests:
    {"event": "process_audio", "audio_b64": "<base64 16kHz mono WAV>"}

Returns:
    {
        "transcript": "...",
        "response_text": "...",
        "response_audio_b64": "<base64 24kHz WAV>",
        "latency_ms": {"stt": 120, "llm": 840, "tts": 95, "total": 1055}
    }

On startup, prints {"status": "ready"} to stdout once all models are warm.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SOCKET_PATH = "/tmp/s2s.sock"

SYSTEM_PROMPT = (
    "You are Wisdom Tree, a helpful and concise voice assistant. "
    "Respond in natural spoken language only. "
    "No markdown, no bullet points, no lists, no special characters. "
    "Keep responses under 3 sentences unless the user explicitly asks for more detail. "
    "Never say 'As an AI' or similar phrases."
)


# ---------------------------------------------------------------------------
# Service class — holds all model state
# ---------------------------------------------------------------------------

class S2SService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.whisper = None
        self.kokoro = None
        self.llm = None

    # ── Model loading ────────────────────────────────────────────────────────

    def load_models(self) -> None:
        self._load_whisper()
        self._load_kokoro()
        self._init_llm_client()

    def _load_whisper(self) -> None:
        from faster_whisper import WhisperModel

        device = "cpu" if self.args.cpu_only else "auto"
        # CTranslate2 doesn't support MPS; "auto" safely falls back to CPU on macOS.
        log.info("Loading Whisper %s (device=%s) …", self.args.whisper_model, device)
        self.whisper = WhisperModel(
            self.args.whisper_model,
            device=device,
            compute_type="int8",
        )
        log.info("Whisper ready")

    def _load_kokoro(self) -> None:
        from kokoro import KPipeline

        log.info("Loading Kokoro TTS (voice=%s) …", self.args.voice)
        self.kokoro = KPipeline(lang_code="a")  # 'a' = American English
        log.info("Kokoro ready")

    def _init_llm_client(self) -> None:
        import requests

        self.llm = requests.Session()
        self.llm.headers.update({
            "Authorization": f"Bearer {self.args.api_key}",
            "Content-Type": "application/json",
        })
        log.info("LLM client → %s (workspace=%s)", self.args.anythingllm_url, self.args.workspace)

    # ── Warmup ───────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        log.info("Warming up models …")

        # Whisper: transcribe 1s of silence
        t0 = time.perf_counter()
        silence = np.zeros(16_000, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, silence, 16_000)
            tmp = f.name
        try:
            segs, _ = self.whisper.transcribe(tmp, vad_filter=True, language="en")
            list(segs)  # exhaust generator
        finally:
            os.unlink(tmp)
        log.info("Whisper warmup: %.0f ms", (time.perf_counter() - t0) * 1000)

        # Kokoro: synthesize and immediately play "Ready." — warmup + startup cue
        t0 = time.perf_counter()
        ready_audio = self._synthesize("Ready.")
        log.info("Kokoro warmup: %.0f ms", (time.perf_counter() - t0) * 1000)
        self._play_audio(ready_audio)

    # ── Pipeline steps ───────────────────────────────────────────────────────

    def _transcribe(self, wav_path: str) -> str:
        segments, _ = self.whisper.transcribe(
            wav_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def _llm_response(self, transcript: str) -> str:
        url = f"{self.args.anythingllm_url}/api/v1/workspace/{self.args.workspace}/chat"
        payload = {"message": f"@agent {transcript}", "mode": "chat"}
        resp = self.llm.post(url, json=payload, timeout=self.args.timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"AnythingLLM error: {data['error']}")
        raw = data["textResponse"]
        reply = _strip_thinking(raw)
        if raw != reply:
            log.debug("Stripped thinking block (%d chars)", len(raw) - len(reply))
        return reply

    def _synthesize(self, text: str) -> bytes:
        """Synthesize text → 24 kHz WAV bytes."""
        chunks = []
        for _, _, audio in self.kokoro(
            text,
            voice=self.args.voice,
            speed=1.0,
            split_pattern=r"[.!?]+",
        ):
            if audio is not None and len(audio) > 0:
                # kokoro yields torch.Tensor; convert to numpy for soundfile
                arr = audio.cpu().numpy() if hasattr(audio, "cpu") else audio
                chunks.append(arr)

        if not chunks:
            return b""

        full = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, full, 24_000, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _play_audio(self, wav_bytes: bytes) -> None:
        """Play WAV bytes directly through the default audio output."""
        if not wav_bytes:
            return
        import sounddevice as sd
        buf = io.BytesIO(wav_bytes)
        audio, sr = sf.read(buf, dtype="float32")
        sd.play(audio, samplerate=sr)
        sd.wait()

    # ── Request handling ─────────────────────────────────────────────────────

    def _llm_response_gemini(self, transcript: str) -> str:
        import subprocess
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {transcript}\nAssistant:"
        result = subprocess.run(
            ["gemini", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gemini CLI error: {result.stderr.strip()}")
        return result.stdout.strip()

    def process_request(self, req: dict) -> dict:
        t_total = time.perf_counter()
        lat: dict[str, int] = {}
        backend = req.get("backend", "anythingllm")

        # Decode audio to temp file
        wav_bytes = base64.b64decode(req.get("audio_b64", ""))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name

        try:
            # STT
            t0 = time.perf_counter()
            transcript = self._transcribe(tmp)
            lat["stt"] = int((time.perf_counter() - t0) * 1000)
            log.info("STT (%.0f ms): %r", lat["stt"], transcript)

            if not transcript:
                audio = self._synthesize("Sorry, I didn't catch that.")
                return {
                    "error": "no_speech",
                    "response_audio_b64": _b64(audio),
                }

            # LLM
            log.info("LLM backend: %s", backend)
            t0 = time.perf_counter()
            try:
                if backend == "gemini":
                    response_text = self._llm_response_gemini(transcript)
                else:
                    response_text = self._llm_response(transcript)
            except Exception as exc:
                log.error("LLM error: %s", exc)
                audio = self._synthesize("My brain seems to be offline.")
                return {
                    "error": "llm_unavailable",
                    "response_audio_b64": _b64(audio),
                }
            lat["llm"] = int((time.perf_counter() - t0) * 1000)
            log.info("LLM (%.0f ms): %r", lat["llm"], response_text[:80])

            # TTS
            t0 = time.perf_counter()
            try:
                response_audio = self._synthesize(response_text)
                audio_b64_out: Optional[str] = _b64(response_audio)
            except Exception as exc:
                log.error("TTS error: %s", exc)
                audio_b64_out = None
            lat["tts"] = int((time.perf_counter() - t0) * 1000)
            log.info("TTS (%.0f ms)", lat["tts"])

            lat["total"] = int((time.perf_counter() - t_total) * 1000)
            return {
                "transcript": transcript,
                "response_text": response_text,
                "response_audio_b64": audio_b64_out,
                "latency_ms": lat,
            }
        finally:
            os.unlink(tmp)

    # ── Self-test ────────────────────────────────────────────────────────────

    def run_self_test(self) -> None:
        """Inject a hardcoded transcript and verify LLM + TTS without a mic."""
        log.info("Self-test: injecting transcript 'what time is it'")
        try:
            t0 = time.perf_counter()
            text = self._llm_response("what time is it")
            llm_ms = int((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            audio = self._synthesize(text)
            tts_ms = int((time.perf_counter() - t0) * 1000)

            print(json.dumps({
                "test": "pass",
                "transcript": "what time is it",
                "response_text": text,
                "has_audio": bool(audio),
                "latency_ms": {"llm": llm_ms, "tts": tts_ms},
            }, indent=2), flush=True)
        except Exception as exc:
            print(json.dumps({"test": "fail", "error": str(exc)}), flush=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Async socket server
# ---------------------------------------------------------------------------

async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    service: S2SService,
) -> None:
    log.info("Client connected")
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                req = json.loads(line.decode().strip())
            except json.JSONDecodeError as exc:
                log.warning("Bad JSON: %s", exc)
                continue

            event = req.get("event")

            if event == "intro":
                backend = req.get("backend", "anythingllm")
                if backend == "gemini":
                    text = "Hi, I'm Wisdom Tree, your local AI assistant. You are using Google Gemini for reasoning."
                else:
                    text = "Hi, I'm Wisdom Tree, your local AI assistant. You are using your local reasoning engine."
                try:
                    audio = await loop.run_in_executor(None, service._synthesize, text)
                    response = {"response_audio_b64": _b64(audio)}
                except Exception as exc:
                    log.error("Intro TTS error: %s", exc)
                    response = {"error": str(exc)}

            elif event == "process_audio":
                try:
                    response = await loop.run_in_executor(None, service.process_request, req)
                except Exception as exc:
                    log.error("Pipeline error: %s", exc, exc_info=True)
                    response = {"error": str(exc)}

            else:
                log.warning("Unknown event: %s", event)
                continue

            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        log.info("Client disconnected")
        writer.close()


async def run_server(service: S2SService) -> None:
    sock = Path(SOCKET_PATH)
    if sock.exists():
        sock.unlink()

    # Default limit is 64 KB — raise it to 50 MB to handle large audio payloads.
    # 30 s of 16 kHz 16-bit mono = ~960 KB raw → ~1.3 MB base64 → well under 50 MB.
    server = await asyncio.start_unix_server(
        lambda r, w: _handle_client(r, w, service),
        path=SOCKET_PATH,
        limit=50 * 1024 * 1024,
    )
    log.info("Listening on %s", SOCKET_PATH)
    # Signal ready to stdout so run_all.sh / the Rust daemon can proceed.
    print(json.dumps({"status": "ready"}), flush=True)

    async with server:
        await server.serve_forever()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Speech-to-Speech pipeline service")
    p.add_argument("--whisper-model",  default="medium.en",
                   help="faster-whisper model name (default: medium.en)")
    p.add_argument("--voice",          default="af_sky",
                   help="Kokoro voice ID (default: af_sky)")
    p.add_argument("--anythingllm-url", default=os.environ.get("ANYTHINGLLM_URL", "http://localhost:3001"),
                   help="AnythingLLM base URL. Can also be set via the ANYTHINGLLM_URL environment variable. "
                        "(default: http://localhost:3001)")
    p.add_argument("--api-key",         default=os.environ.get("ANYTHINGLLM_API_KEY", ""),
                   help="AnythingLLM API key (from Settings → API Keys). "
                        "Can also be set via the ANYTHINGLLM_API_KEY environment variable.")
    p.add_argument("--workspace",       default="chimpoo",
                   help="AnythingLLM workspace slug to use as the model (default: chimpoo)")
    p.add_argument("--cpu-only",       action="store_true",
                   help="Force CPU for Whisper (use if MPS causes issues)")
    p.add_argument("--timeout",        type=int, default=90,
                   help="LLM request timeout in seconds (default: 90)")
    p.add_argument("--test",           action="store_true",
                   help="Run self-test with hardcoded transcript then exit")
    return p


def main() -> None:
    args = build_parser().parse_args()
    service = S2SService(args)
    service.load_models()
    service.warmup()

    if args.test:
        service.run_self_test()
        return

    asyncio.run(run_server(service))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(data: bytes) -> Optional[str]:
    return base64.b64encode(data).decode() if data else None


def _strip_thinking(text: str) -> str:
    """Strip reasoning chain from models that emit </think> (e.g. DeepSeek-R1 via LM Studio).

    LM Studio omits the opening <think> tag, so the output looks like:
        "...chain of thought...\n</think>\n\nActual answer."
    We keep only what comes after </think>. If the tag is absent the text is
    returned unchanged (handles both reasoning and non-reasoning models).
    """
    marker = "</think>"
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    return text.strip()


if __name__ == "__main__":
    main()
