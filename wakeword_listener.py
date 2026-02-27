#!/usr/bin/env python3
"""
Wake word sidecar for the Wisdom Tree Rust daemon.

Captures microphone audio, runs wake word inference, and emits JSON
detection events over a Unix domain socket at /tmp/wakeword.sock.

Event format (newline-delimited JSON):
    {"event": "detected", "model": "<name>", "score": <float>, "timestamp": <unix_ms>}

Usage:
    python3 wakeword_listener.py [--model PATH] [--threshold FLOAT]

    --model PATH         Path to a .onnx wake word model (default: models/chimpoo.onnx).
                         A matching <stem>_norm.npz normalisation file must sit alongside it.
    --threshold FLOAT    Minimum score to trigger (default: 0.5).
"""

import argparse
import base64
import io
import json
import logging
import socket
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio
import onnxruntime as ort
import torch
import torchaudio
import webrtcvad
from openwakeword.model import Model as OWWModel
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOCKET_PATH = "/tmp/wakeword.sock"

# openwakeword requires 16 kHz mono 16-bit PCM.
SAMPLE_RATE = 16_000
# 1280 samples = 80 ms — the recommended chunk size for openwakeword.
CHUNK_SAMPLES = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1

RECONNECT_DELAY_S = 2.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

# Feature extraction params for custom flat mel-spec models (e.g. chimpoo)
_SR        = 16_000
_N_MELS    = 32
_FRAME_LEN = int(_SR * 25 / 1000)   # 400 samples
_HOP_LEN   = int(_SR * 10 / 1000)   # 160 samples
_N_FRAMES  = 76
_AUDIO_LEN = _HOP_LEN * (_N_FRAMES - 1) + _FRAME_LEN   # 12 400 samples


class ChimpooDetector:
    """Detector for custom flat log-mel models (input: [batch, n_mels*n_frames])."""

    def __init__(self, model_path: str) -> None:
        path = Path(model_path).resolve()
        norm_path = path.parent / (path.stem + "_norm.npz")

        log.info("Loading custom mel model: %s", path)
        self.session    = ort.InferenceSession(str(path))
        self.input_name = self.session.get_inputs()[0].name
        self.model_name = path.stem

        if norm_path.exists():
            norm       = np.load(str(norm_path))
            self.mean  = norm["mean"].astype(np.float32)
            self.std   = norm["std"].astype(np.float32)
        else:
            log.warning("No norm file found at %s — running without normalisation", norm_path)
            self.mean  = np.zeros(_N_MELS * _N_FRAMES, dtype=np.float32)
            self.std   = np.ones (_N_MELS * _N_FRAMES, dtype=np.float32)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=_SR, n_fft=_FRAME_LEN,
            win_length=_FRAME_LEN, hop_length=_HOP_LEN, n_mels=_N_MELS,
        )
        self._buffer = np.zeros(_AUDIO_LEN, dtype=np.float32)
        log.info("Detector ready — model=%s, input_dim=%d", self.model_name, _N_MELS * _N_FRAMES)

    def predict(self, audio_chunk: np.ndarray) -> dict:
        chunk_f32 = audio_chunk.astype(np.float32) / 32768.0
        self._buffer = np.roll(self._buffer, -len(chunk_f32))
        self._buffer[-len(chunk_f32):] = chunk_f32

        t = torch.from_numpy(self._buffer).unsqueeze(0)
        with torch.no_grad():
            mel     = self.mel_transform(t)
            log_mel = torch.log(mel + 1e-6).squeeze(0)

        if log_mel.shape[1] > _N_FRAMES:
            log_mel = log_mel[:, :_N_FRAMES]
        elif log_mel.shape[1] < _N_FRAMES:
            log_mel = torch.nn.functional.pad(log_mel, (0, _N_FRAMES - log_mel.shape[1]))

        features = log_mel.numpy().astype(np.float32).flatten()
        features = (features - self.mean) / (self.std + 1e-8)
        features = features[np.newaxis, :]

        score = float(np.array(self.session.run(None, {self.input_name: features})[0]).flat[0])
        return {self.model_name: score}


class OWWDetector:
    """Thin wrapper around openwakeword's Model for standard OWW .onnx models."""

    def __init__(self, model_path: str) -> None:
        # If it's an existing file path use it as-is; otherwise pass the bare
        # model name so openwakeword looks it up in its bundled resources.
        p = Path(model_path)
        model_arg = str(p.resolve()) if p.exists() else model_path
        log.info("Loading OWW model: %s", model_arg)
        self._model = OWWModel(wakeword_models=[model_arg], inference_framework="onnx")
        log.info("Model loaded — wake words: %s", list(self._model.models.keys()))

    def predict(self, audio_chunk: np.ndarray) -> dict:
        """Accept a 1280-sample int16 chunk, return {model_name: score}."""
        return self._model.predict(audio_chunk)


def load_model(model_path: str) -> "ChimpooDetector | OWWDetector":
    try:
        p = Path(model_path)
        if p.exists():
            # Peek at input rank: 2D flat → ChimpooDetector, 3D OWW → OWWDetector
            sess = ort.InferenceSession(str(p.resolve()))
            rank = len(sess.get_inputs()[0].shape)
            if rank == 2:
                return ChimpooDetector(model_path)
        return OWWDetector(model_path)
    except Exception as exc:
        log.error("Failed to load model %r: %s", model_path, exc)
        sys.exit(1)


def open_microphone() -> pyaudio.PyAudio:
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )
    log.info(
        "Microphone opened: %d Hz, %d-sample chunks (%.0f ms)",
        SAMPLE_RATE,
        CHUNK_SAMPLES,
        CHUNK_SAMPLES / SAMPLE_RATE * 1000,
    )
    return pa, stream


def connect_to_daemon() -> Optional[socket.socket]:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        return sock
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        log.debug("Connect failed: %s", exc)
        return None


# How many 80 ms chunks to read right after detection and ship as preroll.
# 6 chunks = 480 ms — covers the gap between detection and capture.py opening the mic.
PREROLL_CHUNKS = 6

COOLDOWN_S = 2.0  # seconds to suppress repeat detections after a trigger


def _chunks_to_b64_wav(chunks: list) -> str:
    """Encode a list of int16 numpy chunks as a base64 WAV string."""
    audio = np.concatenate(chunks)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def send_event(sock: socket.socket, model_name: str, score: float,
               preroll_b64: Optional[str] = None) -> None:
    event = {
        "event":       "detected",
        "model":       model_name,
        "score":       round(float(score), 4),
        "timestamp":   int(time.time() * 1000),
        "preroll_b64": preroll_b64,
    }
    payload = json.dumps(event) + "\n"
    sock.sendall(payload.encode())


# webrtcvad frame size: 10 ms at 16 kHz = 160 samples = 320 bytes
_VAD_FRAME_SAMPLES = 160
_VAD_FRAME_BYTES   = _VAD_FRAME_SAMPLES * 2


def _chunk_has_speech(vad: webrtcvad.Vad, pcm_bytes: bytes) -> bool:
    """Return True if any 10 ms sub-frame in the 80 ms chunk contains speech."""
    for i in range(0, len(pcm_bytes), _VAD_FRAME_BYTES):
        frame = pcm_bytes[i:i + _VAD_FRAME_BYTES]
        if len(frame) == _VAD_FRAME_BYTES and vad.is_speech(frame, SAMPLE_RATE):
            return True
    return False


def detection_loop(
    audio_stream: pyaudio.Stream,
    oww_model: OWWDetector,
    sock: socket.socket,
    threshold: float,
    min_hits: int = 3,
    debug: bool = False,
) -> None:
    """Read audio chunks, run inference, emit events. Raises on socket error."""
    log.info(
        "Detection loop running (threshold=%.2f, min_hits=%d, cooldown=%.1fs)",
        threshold, min_hits, COOLDOWN_S,
    )
    vad = webrtcvad.Vad(2)  # aggressiveness 0–3; 2 is a good balance

    last_detected: dict[str, float] = {}   # model_name -> timestamp of last trigger
    hit_counts:    dict[str, int]   = {}   # model_name -> consecutive frames above threshold

    while True:
        raw = audio_stream.read(CHUNK_SAMPLES, exception_on_overflow=False)

        # Gate: skip wake-word inference entirely when there is no speech
        if not _chunk_has_speech(vad, raw):
            for model_name in hit_counts:
                hit_counts[model_name] = 0
            continue

        audio = np.frombuffer(raw, dtype=np.int16)
        predictions: dict[str, float] = oww_model.predict(audio)

        now = time.time()
        for model_name, score in predictions.items():
            if debug:
                bar = "█" * int(score * 20)
                log.info("score=%.3f  [%-20s]  hits=%d", score, bar, hit_counts.get(model_name, 0))

            if score >= threshold:
                hit_counts[model_name] = hit_counts.get(model_name, 0) + 1
            else:
                hit_counts[model_name] = 0
                continue

            if hit_counts[model_name] < min_hits:
                continue  # not enough consecutive frames yet
            if now - last_detected.get(model_name, 0) < COOLDOWN_S:
                continue  # still in cooldown window

            hit_counts[model_name] = 0
            last_detected[model_name] = now
            log.info("Detected: %s  score=%.3f", model_name, score)

            # Read a few more chunks immediately after detection — these are
            # the first words the user says right after the wake phrase.
            # We ship them as preroll so capture.py doesn't miss them.
            preroll = []
            for _ in range(PREROLL_CHUNKS):
                pr = audio_stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                preroll.append(np.frombuffer(pr, dtype=np.int16))
            preroll_b64 = _chunks_to_b64_wav(preroll)

            send_event(sock, model_name, score, preroll_b64)


def run(args: argparse.Namespace) -> None:
    oww_model = load_model(args.model)
    _pa, audio_stream = open_microphone()

    log.info("Waiting to connect to daemon at %s …", SOCKET_PATH)

    while True:
        sock = connect_to_daemon()
        if sock is None:
            log.warning(
                "Could not connect to daemon at %s — retrying in %.0fs …",
                SOCKET_PATH,
                RECONNECT_DELAY_S,
            )
            time.sleep(RECONNECT_DELAY_S)
            continue

        log.info("Connected to Wisdom Tree daemon")
        try:
            detection_loop(audio_stream, oww_model, sock, args.threshold, args.min_hits, args.debug)
        except (BrokenPipeError, ConnectionResetError):
            log.warning("Connection lost — reconnecting …")
        except KeyboardInterrupt:
            log.info("Interrupted — exiting")
            break
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected error: %s", exc, exc_info=True)
            time.sleep(1)
        finally:
            sock.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenWakeWord sidecar — emits detections over a Unix socket.",
    )
    parser.add_argument(
        "--model",
        default="models/chimpoo.onnx",
        help="Path to .onnx wake word model, or a bundled OWW model name (e.g. hey_jarvis_v0.1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection score threshold 0–1 (default: 0.5).",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        dest="min_hits",
        help="Consecutive frames above threshold required to trigger (default: 3 = 240 ms).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print live scores every frame (useful for calibrating threshold).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
