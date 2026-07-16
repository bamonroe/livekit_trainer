package com.bam.livekittrainer

enum class ClipLabel {
    POSITIVE,
    NEGATIVE,
    HARD_NEGATIVE,
    BACKGROUND,
    FALSE_POSITIVE,
    FALSE_NEGATIVE,
}

data class RecordingPrompt(
    val label: ClipLabel,
    val spokenPhrase: String,
    val instruction: String,
)

object PromptGenerator {
    fun initialBatch(project: WakeWordProject): List<RecordingPrompt> {
        val phrase = project.phrase
        val hardNegatives = hardNegativeCandidates(phrase)
        val ordinarySentences = listOf(
            "This is just an ordinary sentence.",
            "I am testing the recording workflow.",
            "The quick note is ready now.",
            "Please ignore this background phrase.",
        )

        return buildList {
            repeat(4) {
                add(
                    RecordingPrompt(
                        label = ClipLabel.POSITIVE,
                        spokenPhrase = phrase,
                        instruction = "Say: $phrase",
                    ),
                )
            }
            hardNegatives.take(4).forEach { candidate ->
                add(
                    RecordingPrompt(
                        label = ClipLabel.HARD_NEGATIVE,
                        spokenPhrase = candidate,
                        instruction = "Say near miss: $candidate",
                    ),
                )
            }
            ordinarySentences.forEach { sentence ->
                add(
                    RecordingPrompt(
                        label = ClipLabel.NEGATIVE,
                        spokenPhrase = sentence,
                        instruction = "Say: $sentence",
                    ),
                )
            }
            add(
                RecordingPrompt(
                    label = ClipLabel.BACKGROUND,
                    spokenPhrase = "",
                    instruction = "Stay silent and capture the room.",
                ),
            )
            add(
                RecordingPrompt(
                    label = ClipLabel.BACKGROUND,
                    spokenPhrase = "",
                    instruction = "Capture normal room noise.",
                ),
            )
        }.shuffled(stableRandom(project.slug))
    }

    private fun hardNegativeCandidates(phrase: String): List<String> {
        val words = phrase.trim().lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }
        val variants = mutableListOf<String>()

        if (words.isNotEmpty()) {
            if (words.size == 1) {
                variants.addAll(soundAlikes(words.first()))
                variants.add("${words.first()} now")
            } else {
                variants.add(words.joinToString(" ") { word -> soundAlikes(word).firstOrNull() ?: word })
                words.forEachIndexed { index, word ->
                    soundAlikes(word).take(2).forEach { replacement ->
                        variants.add(words.replaceAt(index, replacement).joinToString(" "))
                    }
                }
                variants.add(words.first())
                variants.add(words.last())
            }

            if (words.size == 2 && words[0] == words[1]) {
                val word = words.first()
                variants.add("$word ${soundAlikes(word).firstOrNull() ?: "now"}")
                variants.add("${soundAlikes(word).firstOrNull() ?: "now"} $word")
                variants.add("$word $word $word")
            }

            variants.add(words.drop(1).joinToString(" ").ifBlank { words.first() })
            variants.add(words.dropLast(1).joinToString(" ").ifBlank { words.first() })
            variants.add("$phrase now")
            variants.add("not $phrase")
            variants.add("$phrase please")
        }

        return variants
            .map { it.trim().replace(Regex("\\s+"), " ") }
            .filter { it.isNotBlank() && it != phrase.lowercase() }
            .distinct()
    }

    private fun List<String>.replaceAt(index: Int, value: String): List<String> {
        return mapIndexed { itemIndex, item -> if (itemIndex == index) value else item }
    }

    private fun soundAlikes(word: String): List<String> {
        val known = mapOf(
            "beep" to listOf("peep", "deep", "bleep", "boop"),
            "boop" to listOf("beep", "loop", "scoop"),
            "hey" to listOf("hi", "okay", "hey there"),
            "hello" to listOf("yellow", "hollow", "hello there"),
            "buddy" to listOf("body", "bunny", "muddy"),
            "computer" to listOf("commuter", "calculator", "compute"),
            "assistant" to listOf("assistance", "resistant", "assistant now"),
            "bump" to listOf("pump", "dump", "bumped"),
            "wake" to listOf("wait", "bake", "awake"),
            "word" to listOf("world", "work", "words"),
        )
        known[word]?.let { return it }

        return when {
            word.endsWith("ing") && word.length > 5 -> listOf(word.removeSuffix("ing"), "$word now")
            word.endsWith("er") && word.length > 4 -> listOf(word.removeSuffix("er"), "${word} now")
            word.length <= 3 -> listOf("$word now", "not $word")
            else -> listOf("$word now", "not $word", "$word please")
        }
    }

    private fun stableRandom(seed: String): kotlin.random.Random {
        return kotlin.random.Random(seed.fold(17) { acc, char -> acc * 31 + char.code })
    }
}
