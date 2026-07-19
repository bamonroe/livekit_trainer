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
POST /compare         multipart form:
    file       WAV (16 kHz mono, 16-bit PCM)
    models     comma-separated model directory names (or a JSON array). Each is
               resolved to the single .onnx inside <MODELS_DIR>/<name>/. Empty
               means every available model.
    mode/step_ms/keep_ms  as /score
  Scores the ONE uploaded take through every listed model on a shared time grid
  so trained models can be diffed head-to-head. ->
      {"mode","window_ms":2000,"step_ms","keep_ms","duration_ms",
       "times_ms":[...],"results":{"<name>":{"onnx","scores":[...]}, ...},
       "errors":{"<name>":"..."}}
"""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from scorer import Scorer

import json

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/output"))

app = Flask(__name__)
_scorers: dict[str, Scorer] = {}


def resolve_onnx(model: str) -> Path:
    """Locate the .onnx for a model *directory* name. Prefers the
    conventional <model>/<model>.onnx; otherwise the single .onnx in the dir.
    The dir name (not the file name, which can collide across variants) is the
    stable identifier callers use."""
    d = MODELS_DIR / model
    if not d.is_dir():
        raise FileNotFoundError(f"no model directory {model!r} under {MODELS_DIR}")
    preferred = d / f"{model}.onnx"
    if preferred.exists():
        return preferred
    onnxes = sorted(d.glob("*.onnx"))
    if not onnxes:
        raise FileNotFoundError(f"no .onnx in model directory {model!r}")
    if len(onnxes) > 1:
        raise FileNotFoundError(
            f"ambiguous model {model!r}: {[p.name for p in onnxes]}; "
            f"expected {model}.onnx"
        )
    return onnxes[0]


def get_scorer(model: str) -> Scorer:
    if model not in _scorers:
        # Key the scorer's public name by the caller's dir name so /compare
        # results stay distinct even when two variants share an .onnx filename.
        _scorers[model] = Scorer(str(resolve_onnx(model)), model_name=model)
    return _scorers[model]


def available_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    out = []
    for d in sorted(MODELS_DIR.glob("*/")):
        if any(d.glob("*.onnx")):
            out.append(d.name)
    return out


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
    return jsonify(status="ok", models_dir=str(MODELS_DIR),
                   loaded=sorted(_scorers), available=available_models())


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


def _parse_models(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return available_models()
    if raw.startswith("["):
        return [str(m).strip() for m in json.loads(raw) if str(m).strip()]
    return [m.strip() for m in raw.split(",") if m.strip()]


@app.post("/compare")
def compare():
    if "file" not in request.files:
        return jsonify(error="missing 'file'"), 400
    mode = request.form.get("mode", "full")
    step_ms = int(request.form.get("step_ms", 10))
    keep_ms = int(request.form.get("keep_ms", 700))
    try:
        audio = read_wav(request.files["file"].read())
        models = _parse_models(request.form.get("models", ""))
    except (ValueError, json.JSONDecodeError) as e:
        return jsonify(error=str(e)), 400
    if not models:
        return jsonify(error="no models available to compare"), 400

    times_ref = None
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for name in models:
        try:
            scorer = get_scorer(name)
            if mode == "reset":
                times, scores = scorer.score_reset(audio, keep_ms=keep_ms, step_ms=step_ms)
            else:
                times, scores = scorer.score_full(audio, step_ms=step_ms)
        except (FileNotFoundError, ValueError) as e:
            errors[name] = str(e)
            continue
        # Every model shares the same audio and scan grid, so times match; keep
        # the first as the reference and only diverge if a model truncates.
        if times_ref is None:
            times_ref = times
        results[name] = {
            "onnx": resolve_onnx(name).name,
            "scores": [round(float(s), 4) for s in scores],
            "n": int(len(scores)),
        }

    if times_ref is None:
        return jsonify(error="no model produced a curve", errors=errors), 400

    return jsonify(
        mode=mode, window_ms=2000, step_ms=step_ms,
        keep_ms=keep_ms if mode == "reset" else None,
        duration_ms=round(len(audio) / 16000 * 1000, 1),
        times_ms=[round(float(t), 1) for t in times_ref],
        results=results, errors=errors,
    )


if __name__ == "__main__":
    app.run(host=os.environ.get("BIND_HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8770")))
