#!/usr/bin/env python3
"""Generate voice-cloned wake-word positives with Zonos (Zyphra).

Prototype sibling of f5_gen_positives.py. Runs INSIDE the speech-zonos container
so the Zonos model loads once and stays resident for the whole batch. It clones
the user's timbre from their real reference clips and writes one wav per take,
resampled to the trainer's 16 kHz mono if --out-sr is given.

Why Zonos instead of F5 here: Zonos exposes explicit prosody LEVERS on top of a
cloned speaker embedding — speaking_rate, pitch_std (prosodic variability), and
an 8-way emotion vector. The plateau we hit with F5 looked like too-NARROW a
range within the user's own voice (every clone sounded like the same calm read),
so this generator deliberately jitters those levers per clip to spread the batch
across the user's plausible speaking range while keeping ONE cloned timbre.

Design mirrors F5's generator where it matters:
- Phrase lands at the TAIL (optional spoken lead-in carrier before it) so clips
  are tail-aligned like real positives and carry leading speech context.
- Speaker embedding is built per SET from a rotating window of real refs, so the
  clone hears several of the user's utterances and the batch spreads across all
  of them.
- Per-clip seed + lever jitter so N clips off one embedding aren't identical.

Emotion vector order (Zonos): [happiness, sadness, disgust, fear, surprise,
anger, other, neutral].
"""
import argparse
import contextlib
import glob
import os
import random
import tempfile
import wave


def concat_refs(paths, out_path, gap_secs):
    """Concatenate reference wavs into one clip for speaker embedding, returning
    its length in seconds. Mismatched-format clips are skipped (the user's real
    positives are uniform 16 kHz mono). A short silence gap separates utterances."""
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


# Zonos's default emotion vector (roughly neutral-with-warmth). We jitter around
# this rather than swinging to extremes, which distort timbre and hurt cloning.
BASE_EMOTION = [0.30, 0.03, 0.03, 0.03, 0.03, 0.03, 0.25, 0.30]


def jittered_emotion(rng, amount):
    """Perturb the base emotion vector by +/- amount on the expressive dims and
    renormalize, so each clip carries a slightly different affect."""
    e = list(BASE_EMOTION)
    for idx in (0, 6, 7):  # happiness, other, neutral — the safe, timbre-neutral dims
        e[idx] = max(0.0, e[idx] + rng.uniform(-amount, amount))
    total = sum(e) or 1.0
    return [x / total for x in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--ref-text", default="",
                    help="Accepted for interface parity with the F5 generator; "
                         "Zonos clones from the audio embedding, not ref text.")
    ap.add_argument("--gen-text", required=True, help="The wake phrase to speak.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--language", default="en-us")
    ap.add_argument("--model", default="Zyphra/Zonos-v0.1-transformer")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Say the phrase this many times per clip (ultra-short "
                         "gen text can render unstably; 2 gives the model more to "
                         "work with and the slicer can cut the repeats).")
    # Prosody levers — the whole point of using Zonos. Each is jittered per clip.
    ap.add_argument("--rate-min", type=float, default=11.0,
                    help="Min speaking_rate (Zonos ~15 default; lower = slower).")
    ap.add_argument("--rate-max", type=float, default=19.0)
    ap.add_argument("--pitch-std-min", type=float, default=20.0,
                    help="Min pitch_std (prosodic variability; ~20 flat, ~45 lively).")
    ap.add_argument("--pitch-std-max", type=float, default=45.0)
    ap.add_argument("--emotion-jitter", type=float, default=0.08,
                    help="Max per-dim perturbation of the emotion vector.")
    ap.add_argument("--fmax", type=float, default=24000.0)
    # Lead-in carriers so the phrase isn't a hard onset (tail-aligned context).
    ap.add_argument("--lead-carriers",
                    default="okay|so|hey|and then|right|now|um|alright",
                    help="Pipe-separated fragments; one is sometimes prepended "
                         "before the phrase so the wake word has coarticulated "
                         "context. Phrase still lands last.")
    ap.add_argument("--lead-frac", type=float, default=0.5,
                    help="Fraction of clips that get a lead-in carrier.")
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--concat-size", type=int, default=5,
                    help="Real refs concatenated into one speaker-embedding source "
                         "per set (wraps if the user has fewer).")
    ap.add_argument("--seeds-per-set", type=int, default=5,
                    help="Clips rendered per speaker embedding before rotating the "
                         "reference window.")
    ap.add_argument("--concat-gap", type=float, default=0.2)
    ap.add_argument("--out-sr", type=int, default=0,
                    help="If >0, resample each clip to this rate (mono, 16-bit PCM). "
                         "The trainer wants 16000; 0 keeps Zonos's 44.1 kHz output.")
    args = ap.parse_args()

    refs = sorted(glob.glob(os.path.join(args.refs_dir, "*.wav")))
    if not refs:
        raise SystemExit(f"no reference wavs in {args.refs_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    import torchaudio
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Zonos.from_pretrained(args.model, device=device)

    rng = random.Random(args.seed_base)
    phrase_text = " ".join([args.gen_text.strip()] * max(1, args.repeat))
    carriers = [c.strip() for c in args.lead_carriers.split("|") if c.strip()]

    def to_out_sr(path):
        if args.out_sr <= 0:
            return
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != args.out_sr:
            wav = torchaudio.functional.resample(wav, sr, args.out_sr)
        torchaudio.save(path, wav, args.out_sr, encoding="PCM_S", bits_per_sample=16)

    concat_size = max(1, args.concat_size)
    seeds_per_set = max(1, args.seeds_per_set)
    priming_dir = tempfile.mkdtemp(prefix="zonospriming_")
    current_set = -1
    speaker = None

    skipped = 0
    for i in range(args.count):
        # Rebuild the speaker embedding when we roll into a new set: concatenate a
        # rotating window of real refs and embed it, so the clone hears several
        # utterances and the batch spreads across all of the user's positives.
        set_idx = i // seeds_per_set
        if set_idx != current_set:
            current_set = set_idx
            window = [refs[(set_idx * concat_size + k) % len(refs)]
                      for k in range(concat_size)]
            priming_path = os.path.join(priming_dir, f"priming_{set_idx:04d}.wav")
            secs = concat_refs(window, priming_path, args.concat_gap)
            wav, sr = torchaudio.load(priming_path)
            speaker = model.make_speaker_embedding(wav, sr)
            print(f"[set {set_idx}] speaker embedding from "
                  f"{[os.path.basename(x) for x in window]} ({secs:.2f}s)", flush=True)

        seed = args.seed_base + i
        torch.manual_seed(seed)
        rate = rng.uniform(args.rate_min, args.rate_max)
        pitch_std = rng.uniform(args.pitch_std_min, args.pitch_std_max)
        emotion = jittered_emotion(rng, args.emotion_jitter)

        lead = ""
        if carriers and rng.random() < args.lead_frac:
            lead = rng.choice(carriers) + " "
        gen_text = f"{lead}{phrase_text}"

        out = os.path.join(args.out_dir, f"zonos_{i:05d}.wav")
        # One bad render must not abort the batch.
        try:
            cond = make_cond_dict(
                text=gen_text,
                speaker=speaker,
                language=args.language,
                emotion=emotion,
                pitch_std=pitch_std,
                speaking_rate=rate,
                fmax=args.fmax,
            )
            conditioning = model.prepare_conditioning(cond)
            codes = model.generate(conditioning)
            audio = model.autoencoder.decode(codes).cpu()
            if audio.dim() == 3:      # [batch, channels, samples] -> first item
                audio = audio[0]
            torchaudio.save(out, audio, model.autoencoder.sampling_rate,
                            encoding="PCM_S", bits_per_sample=16)
            to_out_sr(out)
        except Exception as e:
            skipped += 1
            if os.path.exists(out):
                os.remove(out)
            print(f"[{i+1}/{args.count}] SKIP set={current_set} "
                  f"text={gen_text!r}: {e}", flush=True)
            continue
        print(f"[{i+1}/{args.count}] set={current_set} rate={rate:.1f} "
              f"pitch_std={pitch_std:.1f} text={gen_text!r} -> {out}", flush=True)

    if skipped:
        print(f"done: {args.count - skipped}/{args.count} written, {skipped} skipped",
              flush=True)


if __name__ == "__main__":
    main()
