package com.bam.livekittrainer

data class WakeWordProject(
    val id: String,
    val phrase: String,
    val slug: String,
    val createdAtMillis: Long,
)

fun slugifyPhrase(input: String): String {
    val slug = input
        .trim()
        .lowercase()
        .replace(Regex("[^a-z0-9]+"), "_")
        .trim('_')
    return slug.ifBlank { "wake_word" }
}
