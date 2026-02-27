#!/usr/bin/env python3
"""
VAD-gated audio capture.

Records from the default microphone at 16 kHz mono using silero-VAD for
end-of-speech detection.  Writes a single line of JSON to stdout:

    {"audio_b64": "<base64 WAV>", "duration_ms": 1234, "speech_detected": true}

If no speech is detected within --silence-timeout seconds:

    {"audio_b64": null, "duration_ms": 5000, "speech_detected": false}

Usage:
    python3 capture.py [--silence-duration SECS] [--max-duration SECS]
                       [--min-speech SECS]       [--silence-timeout SECS]
"""

import argparse
import base64
import collections
import io
import json
import logging
import sys
import time
import wave
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import torch

# Silence all non-critical output so only the result JSON reaches stdout.
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

SAMPLE_RATE = 16_000
# Silero VAD requires exactly 512 samples per chunk at 16 kHz (32 ms).
CHUNK_SAMPLES = 512
VAD_THRESHOLD = 0.5
# Keep ~300 ms of audio before speech starts so we don't clip the first syllable.
PRE_BUFFER_CHUNKS = int(0.3 * SAMPLE_RATE / CHUNK_SAMPLES)  # ≈ 9 chunks


# ---------------------------------------------------------------------------
# VAD model (cached across calls within one process)
# ---------------------------------------------------------------------------

_VAD_MODEL = None


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            verbose=False,
            trust_repo=True,
        )
        _VAD_MODEL.eval()
    return _VAD_MODEL


def _vad_prob(model, chunk_int16: np.ndarray) -> float:
    chunk_f32 = chunk_int16.astype(np.float32) / 32_768.0
    with torch.no_grad():
        return float(model(torch.from_numpy(chunk_f32), SAMPLE_RATE))


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record(
    silence_duration: float,
    max_duration: float,
    min_speech: float,
    silence_timeout: float,
) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    Returns (audio_int16 | None, duration_ms, speech_detected).
    """
    model = _get_vad_model()
    model.reset_states()

    silence_chunks_needed = max(1, int(silence_duration * SAMPLE_RATE / CHUNK_SAMPLES))
    min_speech_chunks     = max(1, int(min_speech * SAMPLE_RATE / CHUNK_SAMPLES))
    max_chunks            = int(max_duration * SAMPLE_RATE / CHUNK_SAMPLES)
    timeout_chunks        = int(silence_timeout * SAMPLE_RATE / CHUNK_SAMPLES)

    pre_buffer: collections.deque = collections.deque(maxlen=PRE_BUFFER_CHUNKS)
    recorded: list[np.ndarray] = []

    speech_started    = False
    silence_counter   = 0
    speech_chunk_count = 0

    t_start = time.perf_counter()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
    ) as stream:
        for i in range(max_chunks):
            raw, _ = stream.read(CHUNK_SAMPLES)
            chunk = np.frombuffer(bytes(raw), dtype=np.int16).copy()
            prob = _vad_prob(model, chunk)
            is_speech = prob > VAD_THRESHOLD

            if not speech_started:
                pre_buffer.append(chunk)
                if is_speech:
                    speech_started = True
                    recorded.extend(list(pre_buffer))
                    recorded.append(chunk)
                    speech_chunk_count = 1
                    silence_counter = 0
                elif i >= timeout_chunks:
                    break  # No speech in time window — give up
            else:
                recorded.append(chunk)
                if is_speech:
                    speech_chunk_count += 1
                    silence_counter = 0
                else:
                    silence_counter += 1
                    if silence_counter >= silence_chunks_needed:
                        break  # Enough silence after speech — done

    duration_ms = int((time.perf_counter() - t_start) * 1000)

    if not speech_started or speech_chunk_count < min_speech_chunks:
        return None, duration_ms, False

    # Trim the trailing silence we recorded after detecting speech end.
    trim = min(silence_counter, len(recorded) - 1)
    if trim > 0:
        recorded = recorded[:-trim]

    audio = np.concatenate(recorded)
    return audio, duration_ms, True


def _to_wav_bytes(audio_int16: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _decode_preroll(preroll_b64: str) -> np.ndarray:
    """Decode a base64 WAV string into an int16 numpy array."""
    raw = base64.b64decode(preroll_b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def main() -> None:
    p = argparse.ArgumentParser(description="VAD-gated mic capture → stdout JSON")
    p.add_argument("--silence-duration", type=float, default=1.5,
                   help="Seconds of silence to mark end-of-speech (default: 1.5)")
    p.add_argument("--max-duration",     type=float, default=30.0,
                   help="Hard recording limit in seconds (default: 30)")
    p.add_argument("--min-speech",       type=float, default=0.3,
                   help="Minimum speech to accept, seconds (default: 0.3)")
    p.add_argument("--silence-timeout",  type=float, default=5.0,
                   help="Give up if no speech starts within N seconds (default: 5)")
    p.add_argument("--preroll",          type=str,   default=None,
                   help="Base64 WAV audio captured immediately after wake word detection")
    args = p.parse_args()

    try:
        audio, duration_ms, detected = record(
            silence_duration=args.silence_duration,
            max_duration=args.max_duration,
            min_speech=args.min_speech,
            silence_timeout=args.silence_timeout,
        )

        # Prepend the preroll so words spoken right after the wake phrase
        # aren't lost during the gap before this process opened the mic.
        if detected and audio is not None and args.preroll:
            preroll = _decode_preroll(args.preroll)
            audio = np.concatenate([preroll, audio])

        if detected and audio is not None:
            wav = _to_wav_bytes(audio)
            result = {
                "audio_b64":      base64.b64encode(wav).decode(),
                "duration_ms":    duration_ms,
                "speech_detected": True,
            }
        else:
            result = {
                "audio_b64":      None,
                "duration_ms":    duration_ms,
                "speech_detected": False,
            }
    except Exception as exc:
        result = {
            "audio_b64":      None,
            "duration_ms":    0,
            "speech_detected": False,
            "error":          str(exc),
        }
        print(json.dumps(result), flush=True)
        sys.exit(1)

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
