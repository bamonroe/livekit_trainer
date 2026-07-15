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
        val words = phrase.trim().split(Regex("\\s+")).filter { it.isNotBlank() }
        val variants = mutableListOf<String>()
        if (words.isNotEmpty()) {
            variants.add(words.drop(1).joinToString(" ").ifBlank { words.first() })
            variants.add(words.joinToString(" ") { word -> word.drop(1).ifBlank { word } })
            variants.add(words.joinToString(" ") { word -> "${word.first()}${word}" })
        }
        variants.add("$phrase now")
        variants.add("not $phrase")
        return variants.map { it.trim() }.filter { it.isNotBlank() && it != phrase }.distinct()
    }

    private fun stableRandom(seed: String): kotlin.random.Random {
        return kotlin.random.Random(seed.fold(17) { acc, char -> acc * 31 + char.code })
    }
}
