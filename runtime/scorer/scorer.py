"""Wake-word rolling-window scorer.

Turns a mono 16 kHz WAV into a rolling detection-score curve for a trained
LiveKit wake-word ``.onnx`` classifier, at a configurable time resolution.

Two scan modes:

* ``full``  — score the model exactly as it runs on a continuous stream: a
  2.0 s window slides over the audio and every step scores the last 16 speech
  embeddings. The heavy mel + embedding stages run ONCE over the whole clip
  (a dense embedding bank); only the cheap classifier head repeats per step,
  so 10 ms resolution costs a fraction of a second. Reproduces the model's
  native 80 ms grid exactly at the shared points.

* ``reset`` — at each step, keep only the last ``keep_ms`` of real audio and
  pad silence in front to fill the 2.0 s window, mimicking the isolated,
  silence-padded clips the trainer builds. This recovers detection of a phrase
  spoken mid-utterance (see docs/streaming notes). Costs one model pass per
  step, batched for speed.

The window is 2.0 s; internally the model emits one embedding every 8 mel
frames (~80 ms) and the classifier consumes the last 16. Those are fixed by the
trained model. The *scan* step (how often we ask for a score) is free to choose.
"""

from __future__ import annotations

import numpy as np

from livekit.wakeword.inference.model import (
    WakeWordModel,
    EMBEDDING_WINDOW,   # 76 mel frames per embedding
    EMBEDDING_STRIDE,   # 8 mel frames between embeddings (~80 ms)
    MIN_EMBEDDINGS,     # 16 embeddings -> classifier input
)

SAMPLE_RATE = 16000
MEL_HOP = 160                      # samples per mel frame (10 ms)
WINDOW_SAMPLES = 32000             # 2.0 s classifier window
SPAN_FRAMES = (MIN_EMBEDDINGS - 1) * EMBEDDING_STRIDE  # 120 mel frames


class Scorer:
    def __init__(self, onnx_path: str, model_name: str | None = None):
        self._model = WakeWordModel(models=[onnx_path])
        self._name, (self._sess, self._inp) = next(iter(self._model._classifiers.items()))
        if model_name:
            self._name = model_name

    @property
    def name(self) -> str:
        return self._name

    # ---- primitives ----------------------------------------------------
    def _mel(self, audio: np.ndarray) -> np.ndarray:
        mel = self._model._mel_frontend(audio.astype(np.float32))
        return mel[0] if mel.ndim == 3 else mel

    def _embeddings(self, mel: np.ndarray) -> np.ndarray:
        """Dense embedding bank: one embedding per mel frame (stride 1)."""
        k = mel.shape[0] - EMBEDDING_WINDOW + 1
        if k < 1:
            return np.zeros((0, 96), np.float32)
        return np.stack([
            self._model._speech_embedding(mel[i:i + EMBEDDING_WINDOW][np.newaxis])[0]
            for i in range(k)
        ])

    def _tail_embeddings(self, mel: np.ndarray) -> np.ndarray:
        """The last <=16 embeddings on the native stride-8 grid, as predict()
        selects them. Only ~16 embedding calls per window (not the dense bank)."""
        k = mel.shape[0] - EMBEDDING_WINDOW + 1
        if k < 1:
            return np.zeros((0, 96), np.float32)
        starts = list(range(0, k, EMBEDDING_STRIDE))[-MIN_EMBEDDINGS:]
        return np.stack([
            self._model._speech_embedding(mel[i:i + EMBEDDING_WINDOW][np.newaxis])[0]
            for i in starts
        ])

    def _classify(self, seqs: np.ndarray) -> np.ndarray:
        """seqs: (B, 16, 96) -> (B,) scores in [0,1]."""
        out = self._sess.run(None, {self._inp: seqs.astype(np.float32)})[0]
        return out[:, 0].astype(np.float64)

    # ---- full continuous scan -----------------------------------------
    def score_full(self, audio: np.ndarray, step_ms: int = 10) -> tuple[np.ndarray, np.ndarray]:
        step = max(1, round(step_ms / 10))  # mel frames per scan step (hop=10ms)
        mel = self._mel(audio)
        E = self._embeddings(mel)
        k = E.shape[0]
        if k <= SPAN_FRAMES:
            return np.zeros(0), np.zeros(0)
        ends = list(range(SPAN_FRAMES, k, step))
        seqs = np.stack([E[e - SPAN_FRAMES:e + 1:EMBEDDING_STRIDE] for e in ends])
        scores = self._classify(seqs)
        times_ms = np.array([(e + EMBEDDING_WINDOW) * (MEL_HOP / SAMPLE_RATE) * 1000 for e in ends])
        return times_ms, scores

    # ---- silence-reset scan -------------------------------------------
    def _window_at(self, audio: np.ndarray, end: int, keep: int) -> np.ndarray:
        seg = audio[max(0, end - keep):end]
        if len(seg) >= WINDOW_SAMPLES:
            return seg[-WINDOW_SAMPLES:]
        buf = np.zeros(WINDOW_SAMPLES, np.float32)
        buf[WINDOW_SAMPLES - len(seg):] = seg
        return buf

    def score_reset(self, audio: np.ndarray, keep_ms: int = 700, step_ms: int = 10,
                    emb_batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        keep = int(keep_ms / 1000 * SAMPLE_RATE)
        step = max(MEL_HOP, int(step_ms / 1000 * SAMPLE_RATE))
        ends = [min(e, len(audio)) for e in range(step, len(audio) + step, step)]

        # Collect the 16 stride-8 mel windows for every step, then run the
        # embedding model once over all of them (vectorized) instead of
        # 16 tiny calls per step. Classifier is batched the same way.
        mel_windows: list[np.ndarray] = []   # each (76, D)
        per_step: list[list[int]] = []       # indices into mel_windows, per step
        for e in ends:
            mel = self._mel(self._window_at(audio, e, keep))
            k = mel.shape[0] - EMBEDDING_WINDOW + 1
            starts = list(range(0, max(0, k), EMBEDDING_STRIDE))[-MIN_EMBEDDINGS:]
            idx = []
            for s in starts:
                idx.append(len(mel_windows))
                mel_windows.append(mel[s:s + EMBEDDING_WINDOW])
            per_step.append(idx)

        if not mel_windows:
            return np.zeros(0), np.zeros(0)
        stacked = np.stack(mel_windows).astype(np.float32)
        embs = np.concatenate([
            self._model._speech_embedding(stacked[i:i + emb_batch])
            for i in range(0, len(stacked), emb_batch)
        ])

        seqs = np.zeros((len(ends), MIN_EMBEDDINGS, 96), np.float32)
        for j, idx in enumerate(per_step):
            e_seq = embs[idx]                       # (<=16, 96), oldest..newest
            seqs[j, MIN_EMBEDDINGS - len(e_seq):] = e_seq   # left-pad short windows
        scores = self._classify(seqs)
        times_ms = np.array([e / SAMPLE_RATE * 1000 for e in ends])
        return times_ms, scores
