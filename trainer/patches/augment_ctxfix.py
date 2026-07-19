"""Audio augmentation pipeline."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from ..config import WakeWordConfig

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """Augmentation pipeline for wake word clips.

    Applies per-sample augmentations, RIR convolution,
    and background noise mixing.
    """

    def __init__(
        self,
        background_paths: list[Path],
        rir_paths: list[Path],
        sample_rate: int = 16000,
    ):
        self.sample_rate = sample_rate
        self.background_files = self._collect_wavs(background_paths)
        self.rir_files = self._collect_wavs(rir_paths)
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
        # Normalize RIR
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
        """Mix audio with random background noise at given SNR."""
        if not self.background_files:
            return audio
        import soundfile as sf

        bg_path = random.choice(self.background_files)
        bg, sr = sf.read(str(bg_path))
        if bg.ndim > 1:
            bg = bg[:, 0]

        # Loop or crop background to match audio length
        if len(bg) < len(audio):
            repeats = (len(audio) // len(bg)) + 1
            bg = np.tile(bg, repeats)
        start = random.randint(0, max(0, len(bg) - len(audio)))
        bg = bg[start : start + len(audio)]

        # Compute SNR mixing
        snr_db = random.uniform(*snr_db_range)
        audio_power = np.mean(audio**2) + 1e-8
        bg_power = np.mean(bg**2) + 1e-8
        scale = np.sqrt(audio_power / (bg_power * 10 ** (snr_db / 10)))
        mixed = audio + scale * bg
        return mixed.astype(np.float32)


def align_clip_to_end(
    audio: np.ndarray,
    target_length: int,
    jitter_samples: int = 3200,  # 200ms at 16kHz
) -> np.ndarray:
    """Align a clip to the END of the target window with random jitter.

    Positive clips are placed at the end of the window with 0-200ms jitter.
    """
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

    # Clean up old augmented files before starting fresh augmentation.
    # This prevents stale _rN.wav files from previous runs piling up.
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
    workers = config.augmentation.workers

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
                is_positive="positive" in split,
                round_idx=round_idx,
                target_duration_s=target_duration,
                workers=workers,
            )


# Per-worker augmentor, built once per process by the pool initializer below.
_WORKER_AUGMENTOR: AudioAugmentor | None = None


def _init_augment_worker(
    background_paths: list[Path], rir_paths: list[Path], sample_rate: int
) -> None:
    """Process-pool initializer: build one augmentor per worker and give each
    process an independent RNG seed. Without reseeding, forked workers inherit the
    parent's ``random``/``numpy`` state and would replay identical augmentations,
    collapsing dataset diversity — the whole point of augmentation."""
    import os

    global _WORKER_AUGMENTOR
    _WORKER_AUGMENTOR = AudioAugmentor(background_paths, rir_paths, sample_rate)
    seed = (os.getpid() * 2654435761) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)


def _augment_one_clip(task: tuple) -> None:
    """Pool task: augment a single clip using this worker's augmentor."""
    wav_path_str, is_positive, round_idx, target_length, sample_rate = task
    assert _WORKER_AUGMENTOR is not None
    _do_augment_clip(
        Path(wav_path_str),
        _WORKER_AUGMENTOR,
        is_positive,
        round_idx,
        target_length,
        sample_rate,
    )


def _do_augment_clip(
    wav_path: Path,
    augmentor: AudioAugmentor,
    is_positive: bool,
    round_idx: int,
    target_length: int,
    sample_rate: int,
) -> None:
    """Augment one clip end-to-end: read -> effects -> RIR -> background -> write.
    Independent of every other clip, so it is safe to run in a process pool."""
    import re

    import soundfile as sf

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    audio = augmentor.augment_clip(audio)
    audio = augmentor.apply_rir(audio)
    audio = augmentor.mix_with_background(audio)

    # Align to target duration only on round 0 (raw TTS clips vary in length;
    # later rounds already have the correct duration).
    if round_idx == 0:
        if is_positive:
            audio = align_clip_to_end(audio, target_length)
            # CTXFIX: align_clip_to_end zero-fills the ~1s in front of the
            # phrase. Pure digital silence before the wake word is a trivially
            # learnable cue that never occurs in a live mic stream, so the model
            # scores ~99% on silence-padded eval clips but ~1% on the same phrase
            # spoken mid-sentence. Re-mix background across the full 2s window so
            # that leading region carries real ambience (and, when the configured
            # background_paths include ordinary-speech clips, real speech) instead
            # of zeros. This teaches the phrase to fire when preceded by sound.
            audio = augmentor.mix_with_background(audio)
        else:
            # Center-pad or crop negatives
            if len(audio) < target_length:
                padded = np.zeros(target_length, dtype=np.float32)
                start = (target_length - len(audio)) // 2
                padded[start : start + len(audio)] = audio
                audio = padded
            elif len(audio) > target_length:
                start = (len(audio) - target_length) // 2
                audio = audio[start : start + target_length]

    # Derive output name from the original stem (strip any _rN suffix)
    orig_stem = re.sub(r"_r\d+$", "", wav_path.stem)
    out_path = wav_path.with_name(f"{orig_stem}_r{round_idx}.wav")
    sf.write(str(out_path), audio, sample_rate)


def _augment_directory(
    clip_dir: Path,
    background_paths: list[Path],
    rir_paths: list[Path],
    is_positive: bool,
    target_duration_s: float = 2.0,
    sample_rate: int = 16000,
    round_idx: int = 0,
    workers: int = 0,
) -> None:
    """Augment all WAV files in a directory, fanned across CPU cores.

    Round 0 reads the original TTS clips (``clip_000000.wav``); round N reads
    round N-1's output so augmentation compounds. Every round writes its own
    ``_rN.wav`` so the originals are preserved. Clips within a round are
    independent and run in a process pool (``workers``: 0 = all cores, 1 =
    inline); rounds stay sequential because each depends on the previous.
    """
    import os
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

    n_workers = workers if workers > 0 else (os.cpu_count() or 1)
    desc = f"Augmenting {clip_dir.name} r{round_idx}"

    # Single-process path: cheap for small sets and easier to debug.
    if n_workers <= 1:
        augmentor = AudioAugmentor(background_paths, rir_paths, sample_rate)
        for wav_path in tqdm(wav_files, desc=desc, unit="clip"):
            _do_augment_clip(
                wav_path, augmentor, is_positive, round_idx, target_length, sample_rate
            )
        return

    tasks = [
        (str(p), is_positive, round_idx, target_length, sample_rate) for p in wav_files
    ]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_augment_worker,
        initargs=(background_paths, rir_paths, sample_rate),
    ) as executor:
        list(
            tqdm(
                executor.map(_augment_one_clip, tasks, chunksize=16),
                total=len(tasks),
                desc=desc,
                unit="clip",
            )
        )
