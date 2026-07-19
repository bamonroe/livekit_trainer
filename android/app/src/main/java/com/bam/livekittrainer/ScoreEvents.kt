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

    /** Search band a target's detection may fall in, trailing the phrase end. */
    fun band(target: ScoreTarget): Pair<Double, Double> =
        (target.endMs - 100.0) to (target.endMs + 400.0)

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

    /** Which targets have a qualifying event inside their search band. */
    fun detectedFlags(
        targets: List<ScoreTarget>,
        events: List<Event>,
    ): List<Boolean> = targets.map { target ->
        val b = band(target)
        events.any { overlaps(it, b) }
    }

    /** Events that fall outside every target's band — the false alarms. */
    fun falseAlarms(
        targets: List<ScoreTarget>,
        events: List<Event>,
    ): Int {
        val bands = targets.map { band(it) }
        return events.count { event -> bands.none { overlaps(event, it) } }
    }
}
