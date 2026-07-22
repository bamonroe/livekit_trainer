#!/usr/bin/env python3
"""Generate voice-cloned wake-word positives with F5-TTS.

Runs INSIDE the speech-f5tts container so the F5 model loads once and stays
resident for the whole batch (loading per clip would dominate wall-clock).

It clones the user's timbre from a handful of clean reference clips and writes
one 24 kHz wav per generated take. The host wrapper (f5_gen_positives.sh) stages
the references in, runs this, then resamples the output to the 16 kHz mono the
trainer expects.

Design notes
------------
- Positives are the phrase ALONE, so the clip is tail-aligned like a real
  positive; the trainer's align_clip_to_end + context-fix augmentation supplies
  the leading context. We do not bake in a carrier sentence.
- Timbre variety: rotate over every reference clip so the batch isn't one fixed
  render of one clip.
- Prosody variety: jitter speed and vary the seed per clip so 500 positives
  aren't 500 identical renders (which the wake-word model would overfit to).
- F5 can be unstable on ultra-short gen_text. If a bare phrase renders poorly,
  set --repeat 2 to say it twice with a gap; the slicer/energy fallback can cut
  the repetitions, or keep them as short multi-hit positives.
"""
import argparse
import glob
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir", required=True,
                    help="Dir of clean reference positive clips (the user saying the phrase).")
    ap.add_argument("--ref-text", required=True,
                    help="Default transcript of the reference clips, e.g. 'all set'. "
                         "Overridden per clip by a sibling <name>.txt if one exists "
                         "(enrollment references carry their exact passage this way).")
    ap.add_argument("--gen-text", required=True,
                    help="Phrase to synthesize, usually the same as --ref-text.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Say the phrase N times per clip (helps F5 stability on short text).")
    ap.add_argument("--speed-min", type=float, default=0.85)
    ap.add_argument("--speed-max", type=float, default=1.15)
    ap.add_argument("--model", default="F5TTS_v1_Base")
    ap.add_argument("--seed-base", type=int, default=1234)
    args = ap.parse_args()

    refs = sorted(glob.glob(os.path.join(args.refs_dir, "*.wav")))
    if not refs:
        raise SystemExit(f"no reference wavs in {args.refs_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    from f5_tts.api import F5TTS
    tts = F5TTS(model=args.model)

    rng = random.Random(args.seed_base)
    gen_text = " ".join([args.gen_text.strip()] * args.repeat)

    def ref_text_for(ref_path):
        # An enrollment reference ships its exact passage in a sibling .txt; fall
        # back to the phrase for plain positive-clip references.
        sidecar = os.path.splitext(ref_path)[0] + ".txt"
        if os.path.isfile(sidecar):
            with open(sidecar, encoding="utf-8") as fh:
                text = fh.read().strip()
            if text:
                return text
        return args.ref_text

    for i in range(args.count):
        ref = refs[i % len(refs)]                 # rotate timbre across refs
        speed = rng.uniform(args.speed_min, args.speed_max)
        seed = args.seed_base + i                  # deterministic but varied
        out = os.path.join(args.out_dir, f"f5_{i:05d}.wav")
        tts.infer(
            ref_file=ref,
            ref_text=ref_text_for(ref),
            gen_text=gen_text,
            speed=speed,
            seed=seed,
            remove_silence=True,
            file_wave=out,
        )
        print(f"[{i+1}/{args.count}] {os.path.basename(ref)} speed={speed:.2f} -> {out}",
              flush=True)


if __name__ == "__main__":
    main()
