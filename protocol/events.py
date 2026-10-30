from __future__ import annotations

import io
import math
import struct
import wave

# Binary WebSocket Frame Header Bytes (1-byte tag + raw PCM16 S16LE mono payload)
BINARY_UPLINK_AUDIO_PCM16 = 0x01    # Edge Mic -> Brain (16kHz, S16LE, mono)
BINARY_DOWNLINK_AUDIO_PCM16 = 0x02  # Brain -> Edge Speaker (24kHz, S16LE, mono)


def pack_uplink_audio(pcm_bytes: bytes) -> bytes:
    """Pack raw 16kHz S16LE mono PCM bytes with 1-byte uplink header."""
    return bytes([BINARY_UPLINK_AUDIO_PCM16]) + pcm_bytes


def pack_downlink_audio(pcm_bytes: bytes) -> bytes:
    """Pack raw 24kHz S16LE mono PCM bytes with 1-byte downlink header."""
    return bytes([BINARY_DOWNLINK_AUDIO_PCM16]) + pcm_bytes


def unpack_binary_frame(frame: bytes) -> tuple[int, bytes]:
    """Unpack a binary WebSocket message into (frame_type, payload)."""
    if not frame:
        return 0, b""
    return frame[0], frame[1:]


def compute_pcm16_rms(pcm_bytes: bytes) -> float:
    """Compute normalized RMS amplitude [0.0, 1.0] of S16LE PCM audio."""
    n_samples = len(pcm_bytes) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes[: n_samples * 2])
    sum_sq = sum(s * s for s in samples)
    rms = math.sqrt(sum_sq / n_samples) / 32768.0
    return min(1.0, rms)


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw S16LE PCM bytes into an in-memory RIFF WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
