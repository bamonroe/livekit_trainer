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
- Timbre variety: the batch is generated in SETS. Each set concatenates a small
  window of real reference clips (--concat-size) into one longer priming
  reference so F5 clones from several utterances of the user's voice at once, then
  renders --seeds-per-set clips off that one priming (each with its own seed)
  before ROTATING the window to the next block of references and repeating. This
  gives F5 richer timbre context per render while still spreading the batch
  across all of the user's real positives.
- Prosody variety: jitter speed and vary the seed per clip so 500 positives
  aren't 500 identical renders (which the wake-word model would overfit to). The
  seeds within a set are distinct, so the N clips off one priming still differ.
- F5 can be unstable on ultra-short gen_text. If a bare phrase renders poorly,
  set --repeat 2 to say it twice with a gap; the slicer/energy fallback can cut
  the repetitions, or keep them as short multi-hit positives.
"""
import argparse
import contextlib
import glob
import os
import random
import tempfile
import wave


def concat_refs(paths, out_path, gap_secs):
    """Concatenate reference wavs into one priming clip, returning its length in
    seconds. Clips whose format doesn't match the first are skipped (the user's
    real positives are uniform 16 kHz mono, so this is just a guard). A short
    silence gap is inserted between clips so utterances don't run together."""
    frames = []
    params = None
    for p in paths:
        with contextlib.closing(wave.open(p)) as w:
            cur = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = cur
            elif cur != params:
                continue
            frames.append(w.readframes(w.getnframes()))
    if not frames:
        raise RuntimeError(f"no compatible reference clips among {paths}")
    nch, width, rate = params
    gap = b"\x00" * (int(gap_secs * rate) * width * nch)
    data = gap.join(frames)
    with contextlib.closing(wave.open(out_path, "wb")) as w:
        w.setnchannels(nch)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(data)
    return len(data) / float(rate * width * nch)


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
    ap.add_argument("--sec-per-word", type=float, default=0.42,
                    help="Seconds F5 is told to allocate PER WORD of gen_text. F5 "
                         "otherwise sizes the spoken part from the reference's "
                         "rate (ref_audio_len/ref_text_len), which starves short "
                         "phrases when the reference is a long enrollment passage "
                         "and crushes/drops the lead-in. We instead pass an explicit "
                         "fix_duration = ref_len + words*sec_per_word/speed + margin, "
                         "so the words always get real time. remove_silence trims any "
                         "slack, so erring generous is safe.")
    ap.add_argument("--dur-margin", type=float, default=0.5,
                    help="Extra seconds added to the gen-portion duration budget "
                         "(trimmed back by remove_silence).")
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--concat-size", type=int, default=5,
                    help="How many real reference clips to concatenate into one "
                         "priming reference per set. F5 clones from all of them at "
                         "once, so it hears several utterances of the user's voice "
                         "before rendering. Wraps around if the user has fewer refs.")
    ap.add_argument("--seeds-per-set", type=int, default=5,
                    help="How many clips to render off each concatenated priming "
                         "before rotating the reference window to the next block. "
                         "Each of these gets a distinct seed.")
    ap.add_argument("--concat-gap", type=float, default=0.2,
                    help="Seconds of silence inserted between concatenated "
                         "reference clips so they don't run together.")
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

    # Generate in sets: concatenate a rotating window of real refs into one
    # priming clip, render `seeds_per_set` clips off it, then rotate the window.
    concat_size = max(1, args.concat_size)
    seeds_per_set = max(1, args.seeds_per_set)
    priming_dir = tempfile.mkdtemp(prefix="f5priming_")
    current_set = -1
    priming_path = None
    priming_text = None
    priming_secs = 0.0

    skipped = 0
    for i in range(args.count):
        # Rebuild the priming reference when we roll over to a new set. The window
        # advances by concat_size each set and wraps, so the batch spreads across
        # all of the user's real positives.
        set_idx = i // seeds_per_set
        if set_idx != current_set:
            current_set = set_idx
            window = [refs[(set_idx * concat_size + k) % len(refs)]
                      for k in range(concat_size)]
            priming_path = os.path.join(priming_dir, f"priming_{set_idx:04d}.wav")
            priming_secs = concat_refs(window, priming_path, args.concat_gap)
            # The priming audio now holds concat_size utterances of the phrase, so
            # its transcript is the phrase repeated to match.
            priming_text = " ".join([args.ref_text.strip()] * len(window))
            print(f"[set {set_idx}] priming from "
                  f"{[os.path.basename(x) for x in window]} "
                  f"({priming_secs:.2f}s)", flush=True)

        speed = rng.uniform(args.speed_min, args.speed_max)
        seed = args.seed_base + i                  # distinct per clip within a set

        # Prepend a spoken lead-in so the wake word has real preceding context
        # and F5 doesn't render a hard onset right before it. The phrase stays
        # LAST, so the clip is still tail-aligned; the lead-in is windowed as the
        # leading context the streaming model needs. Some clips stay bare for
        # variety and pure tail examples.
        lead = ""
        if carriers and rng.random() < args.lead_frac:
            lead = rng.choice(carriers) + " "
        gen_text = f"{lead}{phrase_text}"

        # Give the words real time. Without this F5 sizes the spoken part from
        # the reference's rate, so a short phrase against the longer concatenated
        # priming renders far too short and the lead-in gets dropped. Budget the
        # gen portion by word count; fix_duration is priming_len + that (silence
        # trimmed back afterward).
        n_words = max(1, len(gen_text.split()))
        gen_secs = n_words * args.sec_per_word / speed + args.dur_margin
        fix_duration = priming_secs + gen_secs

        out = os.path.join(args.out_dir, f"f5_{i:05d}.wav")
        # F5 occasionally emits a bad/undecodeable render; skip that one clip
        # rather than aborting the whole batch (one glitch must not cost 100).
        try:
            tts.infer(
                ref_file=priming_path,
                ref_text=priming_text,
                gen_text=gen_text,
                speed=speed,
                seed=seed,
                nfe_step=args.nfe_step,
                cfg_strength=args.cfg_strength,
                fix_duration=fix_duration,
                remove_silence=True,
                file_wave=out,
            )
            to_out_sr(out)
        except Exception as e:
            skipped += 1
            if os.path.exists(out):
                os.remove(out)
            print(f"[{i+1}/{args.count}] SKIP set={current_set} "
                  f"text={gen_text!r}: {e}", flush=True)
            continue
        print(f"[{i+1}/{args.count}] set={current_set} speed={speed:.2f} "
              f"text={gen_text!r} dur={gen_secs:.2f}s -> {out}", flush=True)

    if skipped:
        print(f"done: {args.count - skipped}/{args.count} written, {skipped} skipped",
              flush=True)


if __name__ == "__main__":
    main()
