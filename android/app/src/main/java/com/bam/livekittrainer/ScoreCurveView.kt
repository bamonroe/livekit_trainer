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
    private var minWidthMs: Double = 0.0
    private var playheadMs: Double = -1.0

    private var curveColor = Color.rgb(37, 110, 112)
    private var gridColor = Color.LTGRAY
    private var hitColor = Color.rgb(30, 132, 73)
    private var missColor = Color.rgb(190, 45, 45)
    private var thresholdColor = Color.rgb(190, 45, 45)
    private var labelColor = Color.DKGRAY

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

    /** Minimum plateau width (ms) a run must span to mark a target detected. */
    fun setMinWidth(value: Double) {
        minWidthMs = value
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

        // Per-utterance markers: vertical tick + peak dot, colored by whether a
        // qualifying-width plateau clears the threshold in the target's band
        // (green = detected, red = missed) — same rule as the counts.
        val events = ScoreEvents.events(timesMs, scores, threshold, minWidthMs)
        val detectedFlags = ScoreEvents.detectedFlags(targets, events)
        for ((index, target) in targets.withIndex()) {
            val detected = detectedFlags.getOrElse(index) { false }
            val color = if (detected) hitColor else missColor
            val x = xForMs(target.peakTimeMs)
            markerPaint.color = color
            canvas.drawLine(x, padT, x, padT + plotH, markerPaint)
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
