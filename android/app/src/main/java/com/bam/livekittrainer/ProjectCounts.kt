package com.bam.livekittrainer

/**
 * Per-wake-word clip tallies reported by the server's /projects endpoint.
 *
 * [pooledNegative] is the number of negatives available from every *other*
 * project (their negatives plus their positives reused as hard negatives) that
 * training folds in for this wake word. Because negatives are shared across
 * projects, they accumulate far faster than [positive], so the pooled figure
 * helps show how lopsided the real positive set has become.
 */
data class ProjectCounts(
    val positive: Int,
    val negative: Int,
    val background: Int,
    val pooledNegative: Int,
)
