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
- The phrase always lands at the TAIL so the clip is tail-aligned like a real
  positive. Most clips get a short spoken lead-in fragment before the phrase
  (--lead-carriers / --lead-frac) so F5 renders natural coarticulation into the
  wake word instead of a hard onset/cut right before it, and so the leading
  window carries real speech context. This complements the trainer's
  align_clip_to_end + context-fix augmentation rather than replacing it; some
  clips stay bare for pure tail-aligned examples.
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
    ap.add_argument("--lead-carriers",
                    default="okay|well|so|alright|and then|you know|i think|"
                            "right|let me see|hold on|one sec|here we go|"
                            "anyway|yeah|hmm let me|just a moment|so then",
                    help="Pipe-separated neutral lead-in fragments spoken BEFORE the "
                         "phrase so F5 renders real coarticulation into the wake word "
                         "instead of a hard onset. The phrase still lands at the tail, "
                         "so the clip stays tail-aligned; the lead-in is the leading "
                         "context. Keep these bland and non-triggering.")
    ap.add_argument("--lead-frac", type=float, default=0.7,
                    help="Fraction of clips that get a spoken lead-in; the rest are "
                         "the bare phrase, preserving some pure tail-aligned examples "
                         "and lead-in variety (0 disables lead-ins entirely).")
    ap.add_argument("--speed-min", type=float, default=0.85)
    ap.add_argument("--speed-max", type=float, default=1.15)
    ap.add_argument("--model", default="F5TTS_v1_Base")
    ap.add_argument("--nfe-step", type=int, default=32,
                    help="Denoising steps F5 runs per clip. Higher = more faithful "
                         "to your voice and cleaner, but slower (F5 default 32; "
                         "48-64 noticeably sharper, diminishing returns past ~64).")
    ap.add_argument("--cfg-strength", type=float, default=2.0,
                    help="Classifier-free guidance: how hard F5 adheres to the "
                         "reference voice and target text vs. drifting (F5 default "
                         "2.0; raise toward 3.0 to hew closer to your timbre, too "
                         "high can sound tense/artifacty).")
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--out-sr", type=int, default=0,
                    help="If >0, resample each clip to this rate (mono, 16-bit PCM) "
                         "in-container, so no host-side ffmpeg pass is needed. The "
                         "trainer wants 16000; 0 keeps F5's native 24 kHz output.")
    args = ap.parse_args()

    refs = sorted(glob.glob(os.path.join(args.refs_dir, "*.wav")))
    if not refs:
        raise SystemExit(f"no reference wavs in {args.refs_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    from f5_tts.api import F5TTS
    tts = F5TTS(model=args.model)

    rng = random.Random(args.seed_base)
    phrase_text = " ".join([args.gen_text.strip()] * args.repeat)
    carriers = [c.strip() for c in args.lead_carriers.split("|") if c.strip()]

    def to_out_sr(path):
        # Downsample F5's 24 kHz render to the trainer's 16 kHz mono 16-bit in
        # place, so callers that ask for --out-sr get training-ready clips with
        # no external ffmpeg step.
        if args.out_sr <= 0:
            return
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != args.out_sr:
            wav = torchaudio.functional.resample(wav, sr, args.out_sr)
        torchaudio.save(path, wav, args.out_sr, encoding="PCM_S", bits_per_sample=16)

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

        # Prepend a spoken lead-in so the wake word has real preceding context
        # and F5 doesn't render a hard onset right before it. The phrase stays
        # LAST, so the clip is still tail-aligned; the lead-in is windowed as the
        # leading context the streaming model needs. Some clips stay bare for
        # variety and pure tail examples.
        lead = ""
        if carriers and rng.random() < args.lead_frac:
            lead = rng.choice(carriers) + " "
        gen_text = f"{lead}{phrase_text}"

        out = os.path.join(args.out_dir, f"f5_{i:05d}.wav")
        tts.infer(
            ref_file=ref,
            ref_text=ref_text_for(ref),
            gen_text=gen_text,
            speed=speed,
            seed=seed,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            remove_silence=True,
            file_wave=out,
        )
        to_out_sr(out)
        print(f"[{i+1}/{args.count}] {os.path.basename(ref)} speed={speed:.2f} "
              f"text={gen_text!r} -> {out}", flush=True)


if __name__ == "__main__":
    main()
