import pyaudio
import numpy as np
import threading


class AudioMonitor:
    """
    Continuously samples the default microphone in a background daemon thread
    and flags a violation whenever the RMS energy exceeds `threshold`.

    Usage:
        monitor = AudioMonitor(threshold=800)
        monitor.start()
        ...
        violation = monitor.get_violation()   # "suspicious_audio_detected" or None
        monitor.stop()
    """

    def __init__(self, threshold: int = 800):
        self.chunk    = 1024
        self.format   = pyaudio.paInt16
        self.channels = 1
        self.rate     = 16_000
        self.threshold = threshold

        self._p      = pyaudio.PyAudio()
        self._stream = self._p.open(
            format            = self.format,
            channels          = self.channels,
            rate              = self.rate,
            input             = True,
            frames_per_buffer = self.chunk,
        )

        self._is_monitoring   = False
        self._latest_violation: str | None = None
        self._lock = threading.Lock()   # protect violation state across threads

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Starts audio monitoring in a background daemon thread."""
        t = threading.Thread(target=self._listen, name="AudioMonitor", daemon=True)
        t.start()

    def get_violation(self) -> str | None:
        """
        Returns the current violation string if audio is above threshold,
        or None if audio level is normal.
        Thread-safe.
        """
        with self._lock:
            return self._latest_violation

    def stop(self) -> None:
        """Gracefully shuts down the audio stream."""
        self._is_monitoring = False
        try:
            self._stream.stop_stream()
            self._stream.close()
            self._p.terminate()
        except Exception as e:
            print(f"[AUDIO] Error during shutdown: {e}")

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _listen(self) -> None:
        self._is_monitoring = True
        while self._is_monitoring:
            try:
                raw        = self._stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                rms        = float(np.sqrt(np.mean(np.square(audio_data))))

                with self._lock:
                    if rms > self.threshold:
                        self._latest_violation = "suspicious_audio_detected"
                    else:
                        self._latest_violation = None

            except Exception as e:
                print(f"[AUDIO] Processing error: {e}")