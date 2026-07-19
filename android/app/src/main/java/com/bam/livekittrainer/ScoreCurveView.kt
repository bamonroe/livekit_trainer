package com.bam.livekittrainer

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.view.View

/**
 * Plots a wake-word detection score curve over time with a movable threshold
 * line and per-utterance markers. Nothing else in the app draws to a Canvas, so
 * this is a self-contained custom View: the activity feeds it data and colors,
 * then nudges the threshold/playhead in place without a full page re-render.
 */
class ScoreCurveView(context: Context) : View(context) {
    private var timesMs: List<Double> = emptyList()
    private var scores: List<Double> = emptyList()
    private var targets: List<ScoreTarget> = emptyList()
    private var domainMs: Double = 1.0
    private var threshold: Double = 0.5
    private var windowMs: Double = 0.0
    private var playheadMs: Double = -1.0

    private var curveColor = Color.rgb(37, 110, 112)
    private var gridColor = Color.LTGRAY
    private var hitColor = Color.rgb(30, 132, 73)
    private var missColor = Color.rgb(190, 45, 45)
    private var thresholdColor = Color.rgb(190, 45, 45)
    private var labelColor = Color.DKGRAY
    // Shaded band where the LiveKit engine would actually fire at the current
    // threshold + width, colored by whether Whisper agrees, and the marker for
    // where Whisper located the phrase.
    private var fireColor = Color.argb(64, 245, 158, 66) // agreement (Whisper too)
    private var modelOnlyColor = Color.argb(95, 150, 70, 205) // model caught, Whisper missed
    private var falseAlarmColor = Color.argb(45, 140, 140, 140) // low-confidence noise
    private var whisperColor = Color.rgb(90, 100, 210)

    private val density = resources.displayMetrics.density
    private fun dp(value: Float): Float = value * density

    private val curvePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1f)
    }
    private val markerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1.5f)
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val firePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val whisperPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1.5f)
    }
    private val whisperFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val trianglePath = Path()
    private val thresholdPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1.5f)
        pathEffect = DashPathEffect(floatArrayOf(dp(6f), dp(4f)), 0f)
    }
    private val playheadPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1.5f)
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = dp(10f) }

    private val curvePath = Path()

    fun setColors(curve: Int, grid: Int, hit: Int, miss: Int, threshold: Int, label: Int) {
        curveColor = curve
        gridColor = grid
        hitColor = hit
        missColor = miss
        thresholdColor = threshold
        labelColor = label
        invalidate()
    }

    fun setData(timesMs: List<Double>, scores: List<Double>, targets: List<ScoreTarget>, durationMs: Double) {
        this.timesMs = timesMs
        this.scores = scores
        this.targets = targets
        val lastTime = timesMs.lastOrNull() ?: 0.0
        this.domainMs = maxOf(durationMs, lastTime, 1.0)
        invalidate()
    }

    fun setThreshold(value: Double) {
        threshold = value
        invalidate()
    }

    /** Sliding max-pool window width (ms) for the firing decision. */
    fun setWindow(value: Double) {
        windowMs = value
        invalidate()
    }

    fun setPlayhead(ms: Double) {
        playheadMs = ms
        invalidate()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val height = dp(200f).toInt()
        setMeasuredDimension(width, height)
    }

    override fun onDraw(canvas: Canvas) {
        val padL = dp(30f)
        val padR = dp(8f)
        val padT = dp(8f)
        val padB = dp(20f)
        val plotW = width - padL - padR
        val plotH = height - padT - padB
        if (plotW <= 0f || plotH <= 0f) return

        fun xForMs(ms: Double): Float = padL + (ms / domainMs).toFloat().coerceIn(0f, 1f) * plotW
        fun yForScore(score: Double): Float = padT + (1.0 - score.coerceIn(0.0, 1.0)).toFloat() * plotH

        // Horizontal gridlines + y labels at 0.0 / 0.5 / 1.0.
        gridPaint.color = gridColor
        labelPaint.color = labelColor
        for (level in listOf(0.0, 0.5, 1.0)) {
            val y = yForScore(level)
            canvas.drawLine(padL, y, padL + plotW, y, gridPaint)
            canvas.drawText(String.format("%.1f", level), dp(4f), y + dp(3.5f), labelPaint)
        }

        // Shaded band wherever the LiveKit engine would actually fire at the
        // current threshold + width — the events that drive the counts. Drawn
        // behind the curve so the trace still reads on top. This is where a real
        // trigger would land, lag and all.
        val events = ScoreEvents.events(timesMs, scores, threshold, windowMs)
        val windows = ScoreEvents.windows(targets)
        for (event in events) {
            val inWhisper = windows.any { event.startMs <= it.second && event.endMs >= it.first }
            firePaint.color = when {
                inWhisper -> fireColor
                event.peak >= ScoreEvents.MODEL_ONLY_CONFIDENCE -> modelOnlyColor
                else -> falseAlarmColor
            }
            val xl = xForMs(event.startMs)
            val xr = xForMs(event.endMs)
            canvas.drawRect(xl, padT, maxOf(xr, xl + dp(1.5f)), padT + plotH, firePaint)
        }

        // Score curve.
        if (scores.size >= 2 && timesMs.size == scores.size) {
            curvePath.reset()
            for (index in scores.indices) {
                val x = xForMs(timesMs[index])
                val y = yForScore(scores[index])
                if (index == 0) curvePath.moveTo(x, y) else curvePath.lineTo(x, y)
            }
            curvePaint.color = curveColor
            canvas.drawPath(curvePath, curvePaint)
        }

        // Threshold line, drawn over the curve so its level is always readable.
        thresholdPaint.color = thresholdColor
        val thrY = yForScore(threshold)
        canvas.drawLine(padL, thrY, padL + plotW, thrY, thresholdPaint)

        // Whisper markers: a downward triangle at the top edge where Whisper
        // located each spoken wake phrase. Compare against the orange fire band
        // to see where the two engines agree, and where the model fired on a
        // phrase Whisper missed (orange with no triangle) or vice versa.
        whisperPaint.color = whisperColor
        whisperFill.color = whisperColor
        val tri = dp(5f)
        for (target in targets) {
            val x = xForMs(target.endMs)
            trianglePath.reset()
            trianglePath.moveTo(x - tri, padT)
            trianglePath.lineTo(x + tri, padT)
            trianglePath.lineTo(x, padT + tri * 1.4f)
            trianglePath.close()
            canvas.drawPath(trianglePath, whisperFill)
            canvas.drawLine(x, padT, x, padT + plotH, whisperPaint)
        }

        // Per-utterance model markers: peak dot, colored by whether a
        // qualifying-width plateau clears the threshold in the target's window
        // (green = detected, red = missed) — same rule as the counts.
        val detectedFlags = ScoreEvents.detectedFlags(targets, events)
        for ((index, target) in targets.withIndex()) {
            val detected = detectedFlags.getOrElse(index) { false }
            val color = if (detected) hitColor else missColor
            val x = xForMs(target.peakTimeMs)
            dotPaint.color = color
            canvas.drawCircle(x, yForScore(target.peakScore), dp(3.5f), dotPaint)
        }

        // Playhead during replay.
        if (playheadMs >= 0.0) {
            playheadPaint.color = labelColor
            val x = xForMs(playheadMs)
            canvas.drawLine(x, padT, x, padT + plotH, playheadPaint)
        }
    }
}
