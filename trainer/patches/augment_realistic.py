"""Audio augmentation pipeline — realistic-positive compositing.

This replaces the stock ``livekit.wakeword.data.augment`` module (mount it over
the installed file, same mechanism as ``augment_ctxfix.py``).

Motivation (see memory: streaming-recall-gap, context-fix-augmentation)
----------------------------------------------------------------------
The stock trainer synthesizes a bare wake phrase, drops it at the end of a 2 s
window, and zero-fills the ~1 s in front. Pure digital silence before the
phrase is a cue that never occurs on a live mic, so the model scored ~99.8% on
silence-padded eval clips but ~1% on the same phrase spoken mid-sentence.

The first fix (``augment_ctxfix.py``) re-mixed background across the whole
window so the lead-in carried sound. But it *overlapped* real speech on top of
the synthetic phrase at similar level — two voices at once — which is not how a
real trigger reaches a microphone.

This module composites each positive to resemble the real use case:

  * One clear voice on a background bed, never two voices layered at equal level.
  * The voice track is built in *sequence*: optional filler words, a natural
    background-only gap, then the wake word (end-token), or the wake word then
    filler (start-token). Filler is either the user's own recorded speech
    (``voice_lead_paths``) or already-synthesized negative clips (same TTS
    engine) — concatenated, not mixed.
  * A background-only margin on the token's open edge (trailing for end-token,
    leading for start-token), because real speech has room tone before/after the
    phrase, not a hard cutoff.
  * A background bed spanning the full window at a *wide* SNR, from quiet all the
    way up to equal loudness with the voice (0 dB) — but never a second voice at
    equal level. The bed itself is pre-augmented (EQ + reverb) for room
    diversity, drawn mostly from the user's recorded ambience.

Knobs live in a sidecar YAML pointed to by the ``AUG_REALISTIC_CONFIG`` env var
(see ``RealisticCfg``); absent that, sane defaults apply and behaviour degrades
to "bare phrase on a background bed with a trailing margin" — still never zeros.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import WakeWordConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# Realistic-compositing configuration (sidecar YAML)                          #
# --------------------------------------------------------------------------- #
@dataclass
class RealisticCfg:
    """Tunables for realistic positive compositing.

    Loaded from the YAML file named by ``AUG_REALISTIC_CONFIG`` (keys map 1:1 to
    these fields). Every field has a default, so a missing/partial file is fine.
    """

    # Where the wake word sits in the window: "end" (tail-aligned, default) or
    # "start" (head-aligned). Start-token models want filler *after* the phrase.
    token_position: str = "end"

    # Directories of the user's own recorded speech to use as *sequential*
    # lead-in / tail filler (NOT ambience). Distinct from background_paths.
    voice_lead_paths: list[str] = field(default_factory=list)

    # Probability a positive gets any filler voice before/after the phrase.
    lead_probability: float = 0.7
    # Of the clips that get filler, fraction that use real recorded speech
    # rather than a synthesized negative clip.
    real_lead_fraction: float = 0.5
    # Allow synthesized negative clips (model_dir/negative_train) as filler.
    synthetic_lead: bool = True

    # Filler length cap and the background-only gap between filler and phrase.
    max_lead_ms: float = 900.0
    lead_gap_ms: tuple[float, float] = (40.0, 300.0)

    # Background-only margin on the token's open edge.
    margin_ms: tuple[float, float] = (100.0, 700.0)

    # Background bed level relative to the voice, in dB SNR. 0 => bed as loud as
    # the voice (a busy room); large => nearly silent room. Drawn uniformly.
    snr_db_range: tuple[float, float] = (0.0, 18.0)
    # Pre-augment the bed (per-sample EQ/distortion + a room impulse response)
    # so positives see many rooms, not one.
    background_augment: bool = True

    # Peak the composed voice track is normalized to (headroom below clip).
    voice_peak: float = 0.7
    # Short raised-cosine fade at every splice to avoid clicks (ms).
    splice_fade_ms: float = 6.0

    @staticmethod
    def load() -> "RealisticCfg":
        path = os.environ.get("AUG_REALISTIC_CONFIG")
        cfg = RealisticCfg()
        if not path:
            logger.info("No AUG_REALISTIC_CONFIG set; using realistic defaults")
            return cfg
        p = Path(path)
        if not p.exists():
            logger.warning("AUG_REALISTIC_CONFIG=%s not found; using defaults", path)
            return cfg
        import yaml

        with open(p) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(cfg, k):
                logger.warning("Unknown realistic-aug key ignored: %s", k)
                continue
            # Coerce list -> tuple for the *_range / *_ms pair fields.
            if isinstance(getattr(cfg, k), tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(cfg, k, v)
        logger.info("Loaded realistic-aug config from %s: %s", path, cfg)
        return cfg


# --------------------------------------------------------------------------- #
# Small audio helpers                                                         #
# --------------------------------------------------------------------------- #
def _ms_to_samples(ms: float) -> int:
    return int(round(ms * SAMPLE_RATE / 1000.0))


def _read_mono_16k(path: Path) -> np.ndarray | None:
    """Read a wav as mono float32 at 16 kHz, or None on failure/empty."""
    import soundfile as sf

    try:
        audio, sr = sf.read(str(path))
    except Exception as exc:  # noqa: BLE001 - corrupt files must not kill a run
        logger.warning("Skipping unreadable audio %s: %s", path, exc)
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if audio.size == 0:
        return None
    return audio


def _peak_normalize(audio: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.max(np.abs(audio))) if audio.size else 0.0
    if m < 1e-8:
        return audio
    return (audio * (peak / m)).astype(np.float32)


def _fade_edges(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply a short raised-cosine fade in/out to hide splice discontinuities."""
    n = len(audio)
    if fade_samples <= 0 or n < 2 * fade_samples:
        return audio
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_samples)))
    out = audio.copy()
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out


class AudioAugmentor:
    """Augmentation pipeline for wake word clips.

    Applies per-sample augmentations, RIR convolution, background beds, and
    realistic positive compositing.
    """

    def __init__(
        self,
        background_paths: list[Path],
        rir_paths: list[Path],
        sample_rate: int = SAMPLE_RATE,
        voice_lead_paths: list[Path] | None = None,
        synth_filler_paths: list[Path] | None = None,
    ):
        self.sample_rate = sample_rate
        self.background_files = self._collect_wavs(background_paths)
        self.rir_files = self._collect_wavs(rir_paths)
        self.voice_lead_files = self._collect_wavs(voice_lead_paths or [])
        self.synth_filler_files = self._collect_wavs(synth_filler_paths or [])
        self._per_sample_aug = None

    @staticmethod
    def _collect_wavs(dirs: list[Path]) -> list[Path]:
        wavs: list[Path] = []
        for d in dirs:
            if d.exists():
                wavs.extend(d.glob("**/*.wav"))
        return wavs

    def _get_per_sample_augmentations(self) -> Any:
        """Lazy-load audiomentations transforms."""
        if self._per_sample_aug is None:
            from audiomentations import Compose, SevenBandParametricEQ, TanhDistortion

            self._per_sample_aug = Compose(
                [
                    SevenBandParametricEQ(p=0.25),
                    TanhDistortion(p=0.25),
                ]
            )
        return self._per_sample_aug

    def apply_rir(self, audio: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Convolve audio with a random room impulse response."""
        if random.random() > p or not self.rir_files:
            return audio
        import soundfile as sf
        from scipy.signal import fftconvolve

        rir_path = random.choice(self.rir_files)
        rir, sr = sf.read(str(rir_path))
        if rir.ndim > 1:
            rir = rir[:, 0]
        rir = rir / (np.max(np.abs(rir)) + 1e-8)
        convolved = fftconvolve(audio, rir, mode="full")[: len(audio)]
        return convolved.astype(np.float32)

    def augment_clip(self, audio: np.ndarray) -> np.ndarray:
        """Apply per-sample augmentations to a single clip."""
        aug = self._get_per_sample_augmentations()
        return aug(samples=audio, sample_rate=self.sample_rate)

    def mix_with_background(
        self,
        audio: np.ndarray,
        snr_db_range: tuple[float, float] = (5.0, 15.0),
    ) -> np.ndarray:
        """Mix audio with random background noise at a random SNR (whole-clip).

        Used for negatives/backgrounds. Positives use ``compose_positive`` which
        computes SNR against the *voice region* only.
        """
        if not self.background_files:
            return audio
        bg = self._random_background(len(audio), augment=False)
        if bg is None:
            return audio
        snr_db = random.uniform(*snr_db_range)
        audio_power = float(np.mean(audio**2)) + 1e-8
        bg_power = float(np.mean(bg**2)) + 1e-8
        scale = np.sqrt(audio_power / (bg_power * 10 ** (snr_db / 10)))
        return (audio + scale * bg).astype(np.float32)

    # --- realistic compositing helpers ----------------------------------- #
    def _random_background(self, length: int, augment: bool) -> np.ndarray | None:
        """A background segment of exactly *length* samples, looped/cropped.

        When *augment* is set, the bed passes through EQ/distortion and a room
        impulse response so repeated draws of the same source file still yield
        acoustically distinct rooms.
        """
        if not self.background_files:
            return None
        for _ in range(4):  # retry a few unreadable files before giving up
            bg = _read_mono_16k(random.choice(self.background_files))
            if bg is not None and bg.size:
                break
        else:
            return None
        if len(bg) < length:
            reps = (length // len(bg)) + 1
            bg = np.tile(bg, reps)
        start = random.randint(0, max(0, len(bg) - length))
        bg = bg[start : start + length].astype(np.float32)
        if augment:
            bg = self.augment_clip(bg)
            bg = self.apply_rir(bg, p=0.5)
            bg = bg[:length] if len(bg) >= length else np.pad(bg, (0, length - len(bg)))
        return bg.astype(np.float32)

    def _pick_filler(self, real: bool, max_samples: int, fade: int) -> np.ndarray | None:
        """One filler-voice segment (real recorded speech or a synth negative).

        Cropped to at most *max_samples* (a random contiguous span of a longer
        clip) and faded at both edges so the splice is inaudible.
        """
        pool = self.voice_lead_files if real else self.synth_filler_files
        if not pool:
            return None
        for _ in range(4):
            clip = _read_mono_16k(random.choice(pool))
            if clip is not None and clip.size:
                break
        else:
            return None
        if len(clip) > max_samples:
            start = random.randint(0, len(clip) - max_samples)
            clip = clip[start : start + max_samples]
        return _fade_edges(clip.astype(np.float32), fade)


# --------------------------------------------------------------------------- #
# Positive compositing                                                        #
# --------------------------------------------------------------------------- #
def compose_positive(
    phrase: np.ndarray,
    augmentor: AudioAugmentor,
    target_length: int,
    cfg: RealisticCfg,
) -> np.ndarray:
    """Build one realistic 2 s positive window from a bare phrase clip.

    Layout (end-token): [ bed ][ filler? ][ gap ][ PHRASE ][ trailing margin ]
    Layout (start-token): [ leading margin ][ PHRASE ][ gap ][ filler? ][ bed ]
    with a single continuous background bed under the whole window. SNR is
    measured against the placed voice, so the bed level is relative to the
    speaker, not to the silent margins.
    """
    fade = _ms_to_samples(cfg.splice_fade_ms)
    at_end = cfg.token_position != "start"

    phrase = _fade_edges(phrase.astype(np.float32), fade)

    # --- assemble the sequential voice track --------------------------------
    pieces: list[np.ndarray] = []
    if random.random() < cfg.lead_probability:
        use_real = random.random() < cfg.real_lead_fraction
        if not use_real and not cfg.synthetic_lead:
            use_real = True
        filler = augmentor._pick_filler(
            real=use_real, max_samples=_ms_to_samples(cfg.max_lead_ms), fade=fade
        )
        # Fall back to the other source if the preferred one is empty.
        if filler is None:
            filler = augmentor._pick_filler(
                real=not use_real, max_samples=_ms_to_samples(cfg.max_lead_ms), fade=fade
            )
    else:
        filler = None

    gap = np.zeros(random.randint(*(_ms_to_samples(m) for m in cfg.lead_gap_ms)), np.float32)

    if filler is not None:
        if at_end:
            pieces = [filler, gap, phrase]
        else:
            pieces = [phrase, gap, filler]
    else:
        pieces = [phrase]

    voice = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    voice = _peak_normalize(voice, cfg.voice_peak)

    # --- place the voice track in the window with an open-edge margin -------
    window = np.zeros(target_length, np.float32)
    margin = random.randint(*(_ms_to_samples(m) for m in cfg.margin_ms))
    margin = min(margin, max(0, target_length - len(voice)))

    if at_end:
        end_pos = target_length - margin
        start_pos = end_pos - len(voice)
        if start_pos < 0:  # voice longer than window: keep its tail (the phrase)
            voice = voice[-end_pos:]
            start_pos = 0
        vstart, vend = start_pos, start_pos + len(voice)
    else:
        start_pos = margin
        end_pos = start_pos + len(voice)
        if end_pos > target_length:  # keep the head (the phrase)
            voice = voice[: target_length - start_pos]
            end_pos = target_length
        vstart, vend = start_pos, end_pos
    window[vstart:vend] = voice

    # --- one continuous background bed under the whole window ---------------
    bed = augmentor._random_background(target_length, augment=cfg.background_augment)
    if bed is not None:
        voice_power = float(np.mean(window[vstart:vend] ** 2)) + 1e-8
        bed_power = float(np.mean(bed**2)) + 1e-8
        snr_db = random.uniform(*cfg.snr_db_range)
        scale = np.sqrt(voice_power / (bed_power * 10 ** (snr_db / 10)))
        window = window + scale * bed

    # Guard against clipping after summing voice + bed.
    peak = float(np.max(np.abs(window))) if window.size else 0.0
    if peak > 0.99:
        window = window * (0.99 / peak)
    return window.astype(np.float32)


def align_clip_to_end(
    audio: np.ndarray,
    target_length: int,
    jitter_samples: int = 3200,  # 200ms at 16kHz
) -> np.ndarray:
    """Tail-align a clip in the window with random jitter (legacy / negatives)."""
    result = np.zeros(target_length, dtype=np.float32)
    jitter = random.randint(0, jitter_samples)
    end_pos = target_length - jitter
    start_pos = max(0, end_pos - len(audio))
    clip_start = max(0, len(audio) - (end_pos - start_pos))
    result[start_pos:end_pos] = audio[clip_start : clip_start + (end_pos - start_pos)]
    return result


_ALL_SPLITS = [
    "positive_train", "positive_test",
    "negative_train", "negative_test",
    "background_train", "background_test",
]


def run_augment(config: WakeWordConfig) -> None:
    """Run augmentation pipeline on generated clips."""
    import re

    target_duration = config.augmentation.clip_duration
    model_dir = config.model_output_dir
    realistic = RealisticCfg.load()

    # Clean up old augmented files before starting fresh augmentation.
    _aug_re = re.compile(r"^clip_\d{6}_r\d+\.wav$")
    for split in _ALL_SPLITS:
        clip_dir = model_dir / split
        if not clip_dir.exists():
            continue
        old_augs = [p for p in clip_dir.glob("*.wav") if _aug_re.match(p.name)]
        if old_augs:
            logger.info(f"Cleaning {len(old_augs)} old augmented files from {split}")
            for p in old_augs:
                p.unlink()

    background_paths = [Path(p) for p in config.augmentation.background_paths]
    rir_paths = [Path(p) for p in config.augmentation.rir_paths]
    voice_lead_paths = [Path(p) for p in realistic.voice_lead_paths]
    # Synthetic filler = the negative clips we already synthesized (same TTS).
    synth_filler_paths = [model_dir / "negative_train"] if realistic.synthetic_lead else []
    workers = config.augmentation.workers

    logger.info(
        "Realistic augment: token=%s voice_lead_dirs=%d synth_filler=%s snr=%s",
        realistic.token_position,
        len(voice_lead_paths),
        bool(synth_filler_paths),
        realistic.snr_db_range,
    )

    for round_idx in range(config.augmentation.rounds):
        logger.info(f"Augmentation round {round_idx + 1}/{config.augmentation.rounds}")
        for split in _ALL_SPLITS:
            clip_dir = model_dir / split
            if not clip_dir.exists():
                logger.warning(f"Skipping {split}: directory not found")
                continue
            _augment_directory(
                clip_dir,
                background_paths=background_paths,
                rir_paths=rir_paths,
                voice_lead_paths=voice_lead_paths,
                synth_filler_paths=synth_filler_paths,
                realistic=realistic,
                is_positive="positive" in split,
                round_idx=round_idx,
                target_duration_s=target_duration,
                workers=workers,
            )


# Per-worker augmentor, built once per process by the pool initializer below.
_WORKER_AUGMENTOR: AudioAugmentor | None = None
_WORKER_REALISTIC: RealisticCfg | None = None


def _init_augment_worker(
    background_paths: list[Path],
    rir_paths: list[Path],
    voice_lead_paths: list[Path],
    synth_filler_paths: list[Path],
    realistic: RealisticCfg,
    sample_rate: int,
) -> None:
    """Process-pool initializer: build one augmentor per worker and reseed RNG.

    Without reseeding, forked workers inherit the parent's ``random``/``numpy``
    state and replay identical augmentations, collapsing dataset diversity."""
    global _WORKER_AUGMENTOR, _WORKER_REALISTIC
    _WORKER_AUGMENTOR = AudioAugmentor(
        background_paths,
        rir_paths,
        sample_rate,
        voice_lead_paths=voice_lead_paths,
        synth_filler_paths=synth_filler_paths,
    )
    _WORKER_REALISTIC = realistic
    seed = (os.getpid() * 2654435761) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)


def _augment_one_clip(task: tuple) -> None:
    """Pool task: augment a single clip using this worker's augmentor."""
    wav_path_str, is_positive, round_idx, target_length, sample_rate = task
    assert _WORKER_AUGMENTOR is not None and _WORKER_REALISTIC is not None
    _do_augment_clip(
        Path(wav_path_str),
        _WORKER_AUGMENTOR,
        _WORKER_REALISTIC,
        is_positive,
        round_idx,
        target_length,
        sample_rate,
    )


def _do_augment_clip(
    wav_path: Path,
    augmentor: AudioAugmentor,
    realistic: RealisticCfg,
    is_positive: bool,
    round_idx: int,
    target_length: int,
    sample_rate: int,
) -> None:
    """Augment one clip end-to-end. Independent of every other clip."""
    import re

    import soundfile as sf

    audio = _read_mono_16k(wav_path)
    if audio is None:
        return

    # Per-sample effects + reverb apply to every clip; the voice/bed compositing
    # differs by class below.
    audio = augmentor.augment_clip(audio)
    audio = augmentor.apply_rir(audio)

    if round_idx == 0:
        if is_positive:
            # Full realistic compositing: one clear voice (optionally preceded /
            # followed by filler) on a continuous background bed, with an
            # open-edge margin. Never zero-padded silence in front of the phrase.
            audio = compose_positive(audio, augmentor, target_length, realistic)
        else:
            # Center-pad or crop negatives/backgrounds, then a whole-clip bed so
            # they are not dead silent either.
            if len(audio) < target_length:
                padded = np.zeros(target_length, dtype=np.float32)
                start = (target_length - len(audio)) // 2
                padded[start : start + len(audio)] = audio
                audio = padded
            elif len(audio) > target_length:
                start = (len(audio) - target_length) // 2
                audio = audio[start : start + target_length]
            audio = augmentor.mix_with_background(audio, snr_db_range=realistic.snr_db_range)
    else:
        # Later rounds compound light per-sample variation on the already-2 s
        # composited clip; no re-placement (that would double-place the phrase).
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

    orig_stem = re.sub(r"_r\d+$", "", wav_path.stem)
    out_path = wav_path.with_name(f"{orig_stem}_r{round_idx}.wav")
    sf.write(str(out_path), audio, sample_rate)


def _augment_directory(
    clip_dir: Path,
    background_paths: list[Path],
    rir_paths: list[Path],
    voice_lead_paths: list[Path],
    synth_filler_paths: list[Path],
    realistic: RealisticCfg,
    is_positive: bool,
    target_duration_s: float = 2.0,
    sample_rate: int = SAMPLE_RATE,
    round_idx: int = 0,
    workers: int = 0,
) -> None:
    """Augment all original WAV files in a directory, fanned across CPU cores.

    Round 0 reads the original TTS clips (``clip_000000.wav``); round N reads
    round N-1's output. Clips within a round are independent (process pool);
    rounds stay sequential because each depends on the previous.
    """
    import os as _os
    import re
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    target_length = int(target_duration_s * sample_rate)

    if round_idx == 0:
        _src_re = re.compile(r"^clip_\d{6}\.wav$")
    else:
        _src_re = re.compile(rf"^clip_\d{{6}}_r{round_idx - 1}\.wav$")

    wav_files = sorted(p for p in clip_dir.glob("*.wav") if _src_re.match(p.name))
    if not wav_files:
        return

    n_workers = workers if workers > 0 else (_os.cpu_count() or 1)
    desc = f"Augmenting {clip_dir.name} r{round_idx}"

    if n_workers <= 1:
        augmentor = AudioAugmentor(
            background_paths,
            rir_paths,
            sample_rate,
            voice_lead_paths=voice_lead_paths,
            synth_filler_paths=synth_filler_paths,
        )
        for wav_path in tqdm(wav_files, desc=desc, unit="clip"):
            _do_augment_clip(
                wav_path, augmentor, realistic, is_positive, round_idx, target_length, sample_rate
            )
        return

    tasks = [
        (str(p), is_positive, round_idx, target_length, sample_rate) for p in wav_files
    ]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_augment_worker,
        initargs=(
            background_paths,
            rir_paths,
            voice_lead_paths,
            synth_filler_paths,
            realistic,
            sample_rate,
        ),
    ) as executor:
        list(
            tqdm(
                executor.map(_augment_one_clip, tasks, chunksize=16),
                total=len(tasks),
                desc=desc,
                unit="clip",
            )
        )
