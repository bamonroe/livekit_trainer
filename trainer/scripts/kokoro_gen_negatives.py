#!/usr/bin/env python3
"""Generate IMPOSTOR wake-word negatives with Kokoro TTS.

Runs INSIDE the speech-kokoro container (it has requests + soundfile + scipy and
reaches the Kokoro FastAPI at localhost:8880). It synthesizes the wake phrase in
a rotation of NOT-THE-USER, female Kokoro voices and writes one 16 kHz mono wav
per clip. These are pooled into the trainer's NEGATIVES so a personal wake-word
model learns to reject the phrase in other people's voices — the speaker-
discriminative signal that plain near-miss negatives don't provide.

Why Kokoro (not the user's cloned F5/Zonos, and not the trainer's built-in pool):
Kokoro exposes explicit NAMED voices, so we can pick female ones deliberately
("definitely not me"). The built-in trainer pool blends 900+ mixed-gender voices
with no way to select female, and it can't report a live per-file count.

Design mirrors the positive generators where it matters: the phrase lands at the
TAIL (an optional spoken lead-in carrier can precede it) so clips are tail-
aligned like real utterances, and voice/speed are jittered per clip so the batch
spreads across a range rather than repeating one delivery.
"""
import argparse
import io
import os
import random

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

# A deliberately female-only voice pool (Kokoro af_* American female, bf_* British
# female). Every one is unambiguously not the (male) user. Kept as a default so
# callers usually pass nothing; override with --voices to widen or narrow it.
DEFAULT_VOICES = (
    "af_alloy,af_aoede,af_bella,af_heart,af_jessica,af_kore,af_nicole,"
    "af_nova,af_river,af_sarah,af_sky,bf_alice,bf_emma,bf_isabella,bf_lily"
)


def synth(url, voice, text, speed, timeout=60):
    """POST one utterance to Kokoro, returning (float32 mono samples, sample_rate)."""
    r = requests.post(
        f"{url}/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": speed,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    if data.ndim > 1:            # collapse any stereo to mono
        data = data.mean(axis=1)
    return data, sr


def to_out_sr(data, sr, out_sr):
    """Resample float32 mono to out_sr with a polyphase filter (24k -> 16k = 2/3)."""
    if out_sr <= 0 or sr == out_sr:
        return data
    g = np.gcd(int(sr), int(out_sr))
    return resample_poly(data, out_sr // g, sr // g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kokoro-url", default="http://localhost:8880")
    ap.add_argument("--voices", default=DEFAULT_VOICES,
                    help="Comma-separated Kokoro voice ids (rotated per clip).")
    ap.add_argument("--gen-text", required=True, help="The wake phrase to speak.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Say the phrase this many times per clip.")
    ap.add_argument("--speed-min", type=float, default=0.85,
                    help="Min Kokoro speed multiplier (jittered per clip).")
    ap.add_argument("--speed-max", type=float, default=1.15)
    # Lead-in carriers so the phrase isn't a hard onset (tail-aligned context).
    ap.add_argument("--lead-carriers",
                    default="okay|so|hey|and then|right|now|um|alright|well|yeah",
                    help="Pipe-separated fragments; one is sometimes prepended so "
                         "the phrase carries coarticulated context. Phrase lands last.")
    ap.add_argument("--lead-frac", type=float, default=0.5,
                    help="Fraction of clips that get a lead-in carrier.")
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--out-sr", type=int, default=16000,
                    help="Resample each clip to this rate (mono, 16-bit PCM).")
    args = ap.parse_args()

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    if not voices:
        raise SystemExit("no voices given")
    carriers = [c.strip() for c in args.lead_carriers.split("|") if c.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed_base)
    phrase = " ".join([args.gen_text.strip()] * max(1, args.repeat))

    skipped = 0
    for i in range(args.count):
        voice = voices[i % len(voices)]
        speed = rng.uniform(args.speed_min, args.speed_max)
        lead = ""
        if carriers and rng.random() < args.lead_frac:
            lead = rng.choice(carriers) + " "
        text = f"{lead}{phrase}"
        out = os.path.join(args.out_dir, f"kokoro_{i:05d}.wav")
        try:
            data, sr = synth(args.kokoro_url, voice, text, speed)
            data = to_out_sr(data, sr, args.out_sr)
            # Clip guard, then write 16-bit PCM.
            data = np.clip(data, -1.0, 1.0)
            sf.write(out, data, args.out_sr or sr, subtype="PCM_16")
        except Exception as e:
            skipped += 1
            if os.path.exists(out):
                os.remove(out)
            print(f"[{i+1}/{args.count}] SKIP voice={voice} text={text!r}: {e}", flush=True)
            continue
        print(f"[{i+1}/{args.count}] voice={voice} speed={speed:.2f} "
              f"text={text!r} -> {out}", flush=True)

    if skipped:
        print(f"done: {args.count - skipped}/{args.count} written, {skipped} skipped",
              flush=True)


if __name__ == "__main__":
    main()
