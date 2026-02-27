#!/usr/bin/env python3
"""Run this and press buttons on your Satechi remote to see what keys it sends."""

from pynput import keyboard

print("Press buttons on your remote (Ctrl+C to quit)...")
print()

def on_press(key):
    print(f"  PRESSED:  {key}")

def on_release(key):
    print(f"  RELEASED: {key}")
    print()

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
