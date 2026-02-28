#!/usr/bin/env python3
"""
Remote-button sidecar for the Rust voice assistant daemon.

Intercepts Key.media_play_pause from a Satechi remote at the macOS Quartz
event-tap level, **consuming** the event so it does NOT reach other apps
(browsers, Apple Music, etc.).  Sends a JSON detection event over the
Unix socket at /tmp/wakeword.sock.

Event format (newline-delimited JSON):
    {"event": "detected", "model": "remote_button", "score": 1.0,
     "timestamp": <unix_ms>, "preroll_b64": null}

macOS permissions:
    Needs "Input Monitoring" in System Settings → Privacy & Security.
    If the tap can't be created, the script warns and falls back to
    pynput (key is NOT suppressed in that mode).
"""

import json
import logging
import signal
import socket
import sys
import threading
import time
from typing import Optional

import Quartz
import AppKit

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SOCKET_PATH       = "/tmp/wakeword.sock"
RECONNECT_DELAY_S = 2.0
COOLDOWN_S        = 1.0   # minimum seconds between triggers

# macOS media key constants
_NX_KEYTYPE_PLAY        = 16      # play / pause key code
_NX_KEYSTATE_DOWN_MASK  = 0x0A00  # high byte of data1 flags when key is pressed
_NS_SYSDEFINED_TYPE     = 14      # NSEventTypeSystemDefined

# Regular key constants
_KEYPAD_PLUS_KEYCODE    = 69      # kVK_ANSI_KeypadPlus (Dell / extended keyboard)
_EQUAL_KEYCODE          = 24      # kVK_ANSI_Equal — Shift+= gives '+' on Apple keyboard

# ---------------------------------------------------------------------------
# Socket state (accessed from the tap callback thread and main thread)
# ---------------------------------------------------------------------------

_sock:      Optional[socket.socket] = None
_sock_lock: threading.Lock          = threading.Lock()
_last_sent: float                   = 0.0


def _get_socket() -> Optional[socket.socket]:
    """Return cached socket, or try to create a new one."""
    global _sock
    with _sock_lock:
        if _sock is not None:
            return _sock
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(SOCKET_PATH)
            log.info("Connected to daemon at %s", SOCKET_PATH)
            _sock = s
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            log.debug("Daemon not reachable: %s", exc)
        return _sock


def _drop_socket() -> None:
    global _sock
    with _sock_lock:
        if _sock:
            try:
                _sock.close()
            except OSError:
                pass
            _sock = None


# ---------------------------------------------------------------------------
# Send detection event
# ---------------------------------------------------------------------------

def _send_detection() -> None:
    global _last_sent
    now = time.time()
    if now - _last_sent < COOLDOWN_S:
        log.debug("Cooldown — ignoring rapid press")
        return
    _last_sent = now

    payload = (json.dumps({
        "event":       "detected",
        "model":       "remote_button",
        "score":       1.0,
        "timestamp":   int(now * 1000),
        "preroll_b64": None,
    }) + "\n").encode()

    # Try up to twice (once with existing socket, once after reconnect).
    for attempt in range(2):
        sock = _get_socket()
        if sock is None:
            log.warning("Daemon not reachable — will retry on next press")
            return
        try:
            sock.sendall(payload)
            log.info("Remote button pressed — detection event sent")
            return
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            log.warning("Socket error (%s) — reconnecting…", exc)
            _drop_socket()

    log.error("Could not send event after reconnect")


# ---------------------------------------------------------------------------
# CGEventTap callback
# ---------------------------------------------------------------------------

def _event_callback(proxy, event_type, event, _refcon):
    """
    Invoked for every NSSystemDefined or kCGEventKeyDown event.

    Returns None  → event is consumed (not delivered to other apps).
    Returns event → event passes through unchanged.
    """
    # Media keys (play/pause on Satechi remote)
    if event_type == _NS_SYSDEFINED_TYPE:
        ns_ev = AppKit.NSEvent.eventWithCGEvent_(event)
        if ns_ev is None or ns_ev.subtype() != 8:   # subtype 8 = media keys
            return event

        data1      = ns_ev.data1()
        key_code   = (data1 & 0xFFFF0000) >> 16
        key_state  =  data1 & 0xFF00          # high byte of the low word

        if key_code == _NX_KEYTYPE_PLAY and key_state == _NX_KEYSTATE_DOWN_MASK:
            threading.Thread(target=_send_detection, daemon=True).start()
            return None   # ← consume: other apps never see this event

        return event

    # Regular keydown — keypad plus (Dell/extended) or Shift+= / '+' (Apple keyboard)
    if event_type == Quartz.kCGEventKeyDown:
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        flags   = Quartz.CGEventGetFlags(event)
        shift   = bool(flags & Quartz.kCGEventFlagMaskShift)
        if keycode == _KEYPAD_PLUS_KEYCODE or (keycode == _EQUAL_KEYCODE and shift):
            threading.Thread(target=_send_detection, daemon=True).start()
            return None   # ← consume

    return event


# ---------------------------------------------------------------------------
# Run loop (primary path)
# ---------------------------------------------------------------------------

def _run_cgeventtap() -> bool:
    """
    Install a Quartz event tap and run the current thread's CFRunLoop.
    Returns False if the tap could not be created (permission denied).
    """
    mask = (Quartz.CGEventMaskBit(_NS_SYSDEFINED_TYPE) |
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown))
    tap  = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,       # session-level tap
        Quartz.kCGHeadInsertEventTap,    # inserted before everything else
        Quartz.kCGEventTapOptionDefault, # active (can consume events)
        mask,
        _event_callback,
        None,
    )
    if tap is None:
        return False

    src  = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    loop = Quartz.CFRunLoopGetCurrent()
    Quartz.CFRunLoopAddSource(loop, src, Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)
    log.info("CGEventTap active — play/pause key is intercepted exclusively")
    Quartz.CFRunLoopRun()   # blocks until SIGTERM / Ctrl-C
    return True


# ---------------------------------------------------------------------------
# Fallback (pynput) — key reaches other apps but assistant still fires
# ---------------------------------------------------------------------------

def _run_pynput_fallback() -> None:
    from pynput import keyboard

    log.warning(
        "CGEventTap could not be created.\n"
        "  → play/pause key will ALSO reach other apps in this mode.\n"
        "  → To suppress it: System Settings → Privacy & Security → Input Monitoring\n"
        "     and add your terminal app (Terminal / iTerm2 / etc.)."
    )

    def on_press(key):
        if key == keyboard.Key.media_play_pause:
            _send_detection()
        elif hasattr(key, 'vk') and key.vk in (_KEYPAD_PLUS_KEYCODE, _EQUAL_KEYCODE):
            _send_detection()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Remote keypress listener starting (socket=%s)", SOCKET_PATH)

    # Attempt an eager connection so the first button press has no delay.
    _get_socket()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if not _run_cgeventtap():
        _run_pynput_fallback()


if __name__ == "__main__":
    main()
