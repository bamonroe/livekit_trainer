package com.bam.livekittrainer

/**
 * Turns the raw rolling score curve into discrete detection *events*.
 *
 * The scanner scores every ~10 ms, so a single spoken wake word paints a whole
 * run of consecutive above-threshold ticks (a plateau), while spurious blips are
 * only a tick or two wide. An "event" is one contiguous above-threshold run; the
 * [minWidthMs] gate drops runs narrower than that, so a real plateau survives but
 * a one-tick spike is ignored. This is the width filter the Test tab exposes.
 */
object ScoreEvents {
    data class Event(
        val startMs: Double,
        val endMs: Double,
        val peakMs: Double,
        val peak: Double,
    )

    /** Model firing can land up to ~1s from Whisper's reported word time. */
    private const val MAX_DRIFT_MS = 1200.0

    /**
     * One detection search window per target, in time order — matches the
     * server's `target_windows`. Whisper word timings drift up to ~1s from where
     * the tail-aligned model fires, so each window spans `end ± MAX_DRIFT_MS`,
     * clamped to the midpoints between adjacent phrase ends so a dense script's
     * repeats can't claim each other's firing.
     */
    fun windows(targets: List<ScoreTarget>): List<Pair<Double, Double>> =
        targets.indices.map { k ->
            val e = targets[k].endMs
            var lo = e - MAX_DRIFT_MS
            var hi = e + MAX_DRIFT_MS
            if (k > 0) lo = maxOf(lo, (targets[k - 1].endMs + e) / 2.0)
            if (k + 1 < targets.size) hi = minOf(hi, (e + targets[k + 1].endMs) / 2.0)
            maxOf(lo, 0.0) to hi
        }

    /**
     * Contiguous runs of `score >= threshold` whose time span is at least
     * [minWidthMs]. Times/scores are parallel arrays from the score curve.
     */
    fun events(
        timesMs: List<Double>,
        scores: List<Double>,
        threshold: Double,
        minWidthMs: Double,
    ): List<Event> {
        val out = ArrayList<Event>()
        val n = minOf(timesMs.size, scores.size)
        var i = 0
        while (i < n) {
            if (scores[i] < threshold) {
                i++
                continue
            }
            var j = i
            var peak = scores[i]
            var peakMs = timesMs[i]
            while (j < n && scores[j] >= threshold) {
                if (scores[j] > peak) {
                    peak = scores[j]
                    peakMs = timesMs[j]
                }
                j++
            }
            val startMs = timesMs[i]
            val endMs = timesMs[j - 1]
            if (endMs - startMs >= minWidthMs) {
                out.add(Event(startMs, endMs, peakMs, peak))
            }
            i = j
        }
        return out
    }

    private fun overlaps(event: Event, band: Pair<Double, Double>): Boolean =
        event.startMs <= band.second && event.endMs >= band.first

    /** Which targets have a qualifying event inside their drift window. */
    fun detectedFlags(
        targets: List<ScoreTarget>,
        events: List<Event>,
    ): List<Boolean> {
        val windows = windows(targets)
        return windows.map { window -> events.any { overlaps(it, window) } }
    }

    /**
     * A fire that lands outside every Whisper window but peaks this high is
     * almost certainly a real wake word Whisper failed to transcribe, not noise —
     * the model rarely reaches this confidence on non-wake audio. These are the
     * evidence that the LiveKit model beats Whisper at spotting the phrase.
     */
    const val MODEL_ONLY_CONFIDENCE = 0.90

    data class Tally(
        val detected: Int,
        val missed: Int,
        val modelOnly: Int,
        val falseAlarms: Int,
    )

    /**
     * Full breakdown against the Whisper targets: how many phrases the model
     * caught vs missed, and — for fires with no Whisper phrase under them — how
     * many are high-confidence (likely a wake word Whisper missed) vs low
     * (genuine false alarms).
     */
    fun tally(targets: List<ScoreTarget>, events: List<Event>): Tally {
        val windows = windows(targets)
        val detected = windows.count { window -> events.any { overlaps(it, window) } }
        var modelOnly = 0
        var falseAlarms = 0
        for (event in events) {
            if (windows.any { overlaps(event, it) }) continue
            if (event.peak >= MODEL_ONLY_CONFIDENCE) modelOnly++ else falseAlarms++
        }
        return Tally(
            detected = detected,
            missed = targets.size - detected,
            modelOnly = modelOnly,
            falseAlarms = falseAlarms,
        )
    }

    /** Events that fall outside every target's window — the false alarms. */
    fun falseAlarms(
        targets: List<ScoreTarget>,
        events: List<Event>,
    ): Int {
        val windows = windows(targets)
        return events.count { event -> windows.none { overlaps(event, it) } }
    }
}
