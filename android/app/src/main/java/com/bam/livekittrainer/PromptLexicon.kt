package com.bam.livekittrainer

import android.content.Context
import kotlin.math.min

class PromptLexicon private constructor(
    private val words: List<String>,
    private val phonesByWord: Map<String, List<String>>,
) {
    constructor(context: Context) : this(
        words = context.assets.open("prompt_words.txt").bufferedReader().useLines { lines ->
            lines
                .map { it.trim().lowercase() }
                .filter { it.isNotBlank() && !it.startsWith("#") }
                .toList()
        },
        phonesByWord = context.assets.open("cmudict_phones.tsv").bufferedReader().useLines { lines ->
            lines
                .map { it.trim() }
                .filter { it.isNotBlank() && !it.startsWith("#") }
                .mapNotNull { line ->
                    val parts = line.split('\t')
                    if (parts.size == 2) {
                        parts[0] to parts[1].split(' ').map { phone -> phone.filterNot { it.isDigit() } }
                    } else {
                        null
                    }
                }
                .toMap()
        },
    )

    fun phraseInContext(phrase: String, random: kotlin.random.Random): String {
        val excluded = phraseWords(phrase)
        val before = randomWords(random, excluded, random.nextInt(3)).joinToString(" ")
        val after = randomWords(random, excluded, random.nextInt(3)).joinToString(" ")
        return listOf(before, phrase.trim(), after)
            .filter { it.isNotBlank() }
            .joinToString(" ")
            .sentenceCase()
    }

    fun randomWordSentence(random: kotlin.random.Random, phrase: String): String {
        return randomContentWords(random, phrase, 2 + random.nextInt(3))
            .joinToString(" ")
            .sentenceCase()
    }

    fun randomContentWords(random: kotlin.random.Random, phrase: String, count: Int): List<String> {
        return randomWords(random, phraseWords(phrase), count)
    }

    fun phoneticHardNegatives(phrase: String, random: kotlin.random.Random, limit: Int): List<String> {
        val words = phrase.trim().lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.isEmpty()) return emptyList()

        val phraseWordSet = words.toSet()
        val variants = mutableListOf<String>()
        words.forEachIndexed { index, word ->
            val nearWords = nearestWords(word, phraseWordSet, 16).shuffled(random).take(4)
            nearWords.forEach { replacement ->
                variants.add(words.replaceAt(index, replacement).joinToString(" "))
            }
        }

        if (words.size == 1) {
            nearestWords(words.first(), phraseWordSet, 10).shuffled(random).take(4).forEach { near ->
                variants.add("$near ${randomWords(random, phraseWordSet, 1).firstOrNull() ?: "now"}")
            }
        }

        return variants
            .map { it.trim().replace(Regex("\\s+"), " ") }
            .filter { it.isNotBlank() && it != phrase.trim().lowercase() }
            .distinct()
            .take(limit)
    }

    private fun nearestWords(word: String, excluded: Set<String>, limit: Int): List<String> {
        val target = phonesByWord[word] ?: return emptyList()
        return phonesByWord.asSequence()
            .filter { (candidate, phones) ->
                candidate !in excluded &&
                    candidate.length > 2 &&
                    phones.size in ((target.size - 2).coerceAtLeast(1)..target.size + 2)
            }
            .map { (candidate, phones) ->
                candidate to phonemeDistance(target, phones)
            }
            .filter { (_, distance) -> distance in 1..4 }
            .sortedWith(compareBy<Pair<String, Int>> { it.second }.thenBy { wordFrequencyRank(it.first) })
            .take(limit)
            .map { it.first }
            .toList()
    }

    private fun randomWords(random: kotlin.random.Random, excluded: Set<String>, count: Int): List<String> {
        val pool = contentWords.filterNot { it in excluded }
        if (pool.isEmpty()) return fallbackWords.shuffled(random).take(count)
        return pool.shuffled(random).take(count)
    }

    private val contentWords: List<String> by lazy {
        words
            .filter { it.length in 4..10 }
            .filterNot { it in STOP_WORDS }
            .filterNot { it.endsWith("ed") || it.endsWith("ing") }
            .take(8_000)
    }

    private val wordRanks: Map<String, Int> by lazy {
        words.withIndex().associate { it.value to it.index }
    }

    private fun wordFrequencyRank(word: String): Int {
        return wordRanks[word] ?: Int.MAX_VALUE
    }

    private fun phraseWords(phrase: String): Set<String> {
        return phrase.trim().lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }.toSet()
    }

    private fun phonemeDistance(left: List<String>, right: List<String>): Int {
        val previous = IntArray(right.size + 1) { it }
        val current = IntArray(right.size + 1)

        for (i in 1..left.size) {
            current[0] = i
            for (j in 1..right.size) {
                val cost = if (left[i - 1] == right[j - 1]) 0 else 1
                current[j] = min(
                    min(current[j - 1] + 1, previous[j] + 1),
                    previous[j - 1] + cost,
                )
            }
            for (j in previous.indices) {
                previous[j] = current[j]
            }
        }

        return previous[right.size]
    }

    private fun List<String>.replaceAt(index: Int, value: String): List<String> {
        return mapIndexed { itemIndex, item -> if (itemIndex == index) value else item }
    }

    private fun String.sentenceCase(): String {
        return trim()
            .replace(Regex("\\s+"), " ")
            .replaceFirstChar { it.uppercase() }
            .let { if (it.endsWith(".")) it else "$it." }
    }

    companion object {
        private val fallbackWords = listOf(
            "paper",
            "window",
            "silver",
            "garden",
            "river",
            "button",
            "market",
            "yellow",
            "pencil",
            "coffee",
            "planet",
            "folder",
            "bridge",
            "orange",
            "camera",
            "blanket",
            "ticket",
            "station",
            "circle",
            "cabinet",
            "mirror",
            "forest",
            "battery",
            "notebook",
            "plastic",
            "library",
            "carpet",
            "pocket",
            "remote",
            "basket",
        )

        val Empty = PromptLexicon(
            words = fallbackWords,
            phonesByWord = emptyMap(),
        )

        private val STOP_WORDS = setOf(
            "about",
            "after",
            "also",
            "because",
            "been",
            "before",
            "being",
            "between",
            "could",
            "does",
            "from",
            "have",
            "into",
            "more",
            "most",
            "only",
            "other",
            "over",
            "same",
            "some",
            "than",
            "that",
            "their",
            "them",
            "then",
            "there",
            "these",
            "they",
            "this",
            "through",
            "very",
            "were",
            "what",
            "when",
            "where",
            "which",
            "while",
            "with",
            "would",
            "your",
        )
    }
}
