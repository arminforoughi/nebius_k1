"""
PyAudio-compatible wrapper using sounddevice.
Use this when PyAudio fails to build or load on macOS (e.g. _PaMacCore_SetupChannelMap).
"""
import numpy as np
import sounddevice as sd


# Match PyAudio format constant so existing code (e.g. pyaudio.paInt16) works
paInt16 = 8


class _Stream:
    """Stream that mimics PyAudio's Stream read/write/stop_stream/close."""

    def __init__(self, stream, is_input: bool):
        self._stream = stream
        self._is_input = is_input

    def read(self, num_frames, exception_on_overflow=False):
        data, overflowed = self._stream.read(num_frames)
        if exception_on_overflow and overflowed:
            raise OSError("Audio input overflowed")
        return bytes(data) if not isinstance(data, bytes) else data

    def write(self, data):
        if isinstance(data, bytes):
            data = np.frombuffer(data, dtype=np.int16)
        self._stream.write(data)

    def stop_stream(self):
        self._stream.stop()

    def close(self):
        self._stream.close()


class PyAudio:
    """Drop-in replacement for pyaudio.PyAudio using sounddevice."""

    paInt16 = paInt16

    def open(
        self,
        format=None,
        channels=1,
        rate=16000,
        input=False,
        output=False,
        frames_per_buffer=1024,
        input_device_index=None,
        output_device_index=None,
        **kwargs,
    ):
        device = input_device_index if input else output_device_index
        stream_kw = dict(
            samplerate=rate,
            channels=channels,
            dtype="int16",
            blocksize=frames_per_buffer,
        )
        if device is not None:
            stream_kw["device"] = device
        if input:
            stream = sd.RawInputStream(**stream_kw)
        else:
            stream = sd.RawOutputStream(**stream_kw)
        stream.start()
        return _Stream(stream, is_input=input)

    def terminate(self):
        pass  # sounddevice doesn't require cleanup
