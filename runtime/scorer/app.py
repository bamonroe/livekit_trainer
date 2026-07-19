"""HTTP wake-word scoring service.

Mirrors the Whisper-service pattern the sync-server already uses: a small,
stateless HTTP service the Rust server (or the app, during development) calls to
score audio. Turns an uploaded 16 kHz mono WAV into a rolling detection-score
curve for a trained wake-word model, at a chosen time resolution and scan mode.

Endpoints
---------
GET  /health          -> {"status","models_dir","loaded":[...]}
POST /score           multipart form:
    file       WAV (16 kHz mono, 16-bit PCM)
    slug       wake-word slug; model resolved at <MODELS_DIR>/<slug>/<slug>.onnx
    mode       "full" (continuous rolling window) | "reset" (silence-pad)
    step_ms    scan resolution (default 10)
    keep_ms    reset mode: real audio kept before the silence pad (default 700)
  -> {"model","mode","duration_ms","window_ms":2000,"step_ms","keep_ms",
      "times_ms":[...],"scores":[...]}
"""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from scorer import Scorer

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/output"))

app = Flask(__name__)
_scorers: dict[str, Scorer] = {}


def get_scorer(slug: str) -> Scorer:
    if slug not in _scorers:
        onnx = MODELS_DIR / slug / f"{slug}.onnx"
        if not onnx.exists():
            raise FileNotFoundError(f"no model for slug {slug!r} at {onnx}")
        _scorers[slug] = Scorer(str(onnx))
    return _scorers[slug]


def read_wav(raw: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(raw)) as w:
        if w.getnchannels() != 1 or w.getframerate() != 16000 or w.getsampwidth() != 2:
            raise ValueError(
                f"expected 16 kHz mono 16-bit WAV, got {w.getframerate()} Hz "
                f"{w.getnchannels()}ch {w.getsampwidth()*8}-bit"
            )
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0


@app.get("/health")
def health():
    available = sorted(p.name for p in MODELS_DIR.glob("*/") if (p / f"{p.name}.onnx").exists()) \
        if MODELS_DIR.exists() else []
    return jsonify(status="ok", models_dir=str(MODELS_DIR),
                   loaded=sorted(_scorers), available=available)


@app.post("/score")
def score():
    if "file" not in request.files:
        return jsonify(error="missing 'file'"), 400
    slug = request.form.get("slug", "").strip()
    if not slug:
        return jsonify(error="missing 'slug'"), 400
    mode = request.form.get("mode", "full")
    step_ms = int(request.form.get("step_ms", 10))
    keep_ms = int(request.form.get("keep_ms", 700))
    try:
        audio = read_wav(request.files["file"].read())
        scorer = get_scorer(slug)
    except (FileNotFoundError, ValueError) as e:
        return jsonify(error=str(e)), 400

    if mode == "reset":
        times, scores = scorer.score_reset(audio, keep_ms=keep_ms, step_ms=step_ms)
    else:
        times, scores = scorer.score_full(audio, step_ms=step_ms)

    return jsonify(
        model=scorer.name, mode=mode, window_ms=2000, step_ms=step_ms,
        keep_ms=keep_ms if mode == "reset" else None,
        duration_ms=round(len(audio) / 16000 * 1000, 1),
        times_ms=[round(float(t), 1) for t in times],
        scores=[round(float(s), 4) for s in scores],
    )


if __name__ == "__main__":
    app.run(host=os.environ.get("BIND_HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8770")))
