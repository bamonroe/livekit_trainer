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

data class BulkScriptContent(
    val text: String,
    val hardNegatives: List<String>,
)

object PromptGenerator {
    fun bulkScript(
        project: WakeWordProject,
        lexicon: PromptLexicon = PromptLexicon.Empty,
        batchNumber: Int = 0,
        revision: Int = 0,
        wakePlacements: Int = DEFAULT_BULK_WAKE_PLACEMENTS,
    ): String {
        return bulkScriptContent(project, lexicon, batchNumber, revision, wakePlacements).text
    }

    fun bulkScriptContent(
        project: WakeWordProject,
        lexicon: PromptLexicon = PromptLexicon.Empty,
        batchNumber: Int = 0,
        revision: Int = 0,
        wakePlacements: Int = DEFAULT_BULK_WAKE_PLACEMENTS,
    ): BulkScriptContent {
        val phrase = project.phrase
        val placementCount = wakePlacements.coerceIn(1, 48)
        val random = stableRandom("${project.slug}:bulk:$batchNumber:$revision")
        val hardNegatives = hardNegativeCandidates(
            phrase,
            lexicon,
            stableRandom("${project.slug}:bulk:$batchNumber:$revision:hard-negatives"),
        ).shuffled(random).toMutableList()
        val usedHardNegatives = mutableListOf<String>()
        val script = buildList {
            add(bulkNeutralSentence(lexicon, random, phrase))
            add(bulkNeutralSentence(lexicon, random, phrase))
            repeat(placementCount) { index ->
                add(bulkWakeSentence(phrase, lexicon, random, index))
                if (index % 4 == 1 && hardNegatives.isNotEmpty()) {
                    val nearMiss = hardNegatives.removeAt(0)
                    usedHardNegatives.add(nearMiss)
                    add(bulkHardNegativeSentence(nearMiss, lexicon, random, phrase))
                } else {
                    add(bulkNeutralSentence(lexicon, random, phrase))
                }
                if (index % 3 == 0) {
                    add(bulkNeutralSentence(lexicon, random, phrase))
                }
            }
        }.joinToString(" ")
        return BulkScriptContent(script, usedHardNegatives)
    }

    const val DEFAULT_BULK_WAKE_PLACEMENTS = 8

    fun initialBatch(
        project: WakeWordProject,
        lexicon: PromptLexicon = PromptLexicon.Empty,
        batchNumber: Int = 0,
    ): List<RecordingPrompt> {
        val phrase = project.phrase
        val batchSeed = "${project.slug}:batch:$batchNumber"
        val random = stableRandom(batchSeed)
        val hardNegatives = hardNegativeCandidates(phrase, lexicon, stableRandom("$batchSeed:hard-negatives"))
        val contextualPositives = contextualPositiveCandidates(batchSeed, phrase, lexicon)
        val generalNegatives = generalNegativeCandidates(batchSeed, phrase, lexicon)

        return buildList {
            repeat(2) {
                add(
                    RecordingPrompt(
                        label = ClipLabel.POSITIVE,
                        spokenPhrase = phrase,
                        instruction = "Say: $phrase",
                    ),
                )
            }
            contextualPositives.take(3).forEach { utterance ->
                add(
                    RecordingPrompt(
                        label = ClipLabel.POSITIVE,
                        spokenPhrase = utterance,
                        instruction = "Say with wake phrase: $utterance",
                    ),
                )
            }
            hardNegatives.take(5).forEach { candidate ->
                add(
                    RecordingPrompt(
                        label = ClipLabel.HARD_NEGATIVE,
                        spokenPhrase = candidate,
                        instruction = "Say near miss: $candidate",
                    ),
                )
            }
            generalNegatives.forEach { sentence ->
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
        }.shuffled(random)
    }

    private fun contextualPositiveCandidates(
        batchSeed: String,
        phrase: String,
        lexicon: PromptLexicon,
    ): List<String> {
        val random = stableRandom("$batchSeed:context-positives")
        return buildList {
            repeat(8) {
                add(lexicon.phraseInContext(phrase, random))
            }
        }.distinct()
    }

    private fun generalNegativeCandidates(
        batchSeed: String,
        phrase: String,
        lexicon: PromptLexicon,
    ): List<String> {
        val random = stableRandom("$batchSeed:general-negatives")

        val ordinarySentences = listOf(
            "The kitchen light is still on.",
            "I need to check the calendar later.",
            "Please put the small box on the shelf.",
            "The quick note is ready now.",
            "This is just an ordinary sentence.",
            "A blue folder is beside the keyboard.",
            "The window is open in the next room.",
            "We should leave after lunch today.",
            "The charger is next to the notebook.",
            "I found the receipt in my jacket.",
            "The meeting moved to Friday morning.",
            "Please ignore this unrelated phrase.",
            "The coffee mug is near the sink.",
            "I am testing the recording workflow.",
            "The package arrived before dinner.",
            "Move the paper bag under the table.",
        )

        val unrelatedCommands = listOf(
            "Start the timer for five minutes.",
            "Turn the volume down a little.",
            "Read the next message out loud.",
            "Open the list from yesterday.",
            "Remind me to water the plants.",
            "Set an alarm for seven thirty.",
            "Pause the music in the hallway.",
            "Search for the nearest grocery store.",
        )

        val numberPrompts = listOf(
            "Seven four nine two one.",
            "Thirty eight plus twelve equals fifty.",
            "Room two sixteen is down the hall.",
            "The code is eight zero three five.",
        )

        return buildList {
            addAll(ordinarySentences.shuffled(random).take(4))
            addAll(unrelatedCommands.shuffled(random).take(2))
            addAll(numberPrompts.shuffled(random).take(1))
            repeat(4) {
                add(lexicon.randomWordSentence(random, phrase))
            }
        }.distinct()
    }

    private fun bulkNeutralSentence(
        lexicon: PromptLexicon,
        random: kotlin.random.Random,
        phrase: String,
    ): String {
        val words = lexicon.randomContentWords(random, phrase, 4)
        val first = words.getOrElse(0) { "notebook" }
        val second = words.getOrElse(1) { "counter" }
        val third = words.getOrElse(2) { "folder" }
        val fourth = words.getOrElse(3) { "window" }
        return when (random.nextInt(10)) {
            0 -> "The $first is beside the $second."
            1 -> "Please move the $first near the $second."
            2 -> "I checked the $first before the $second."
            3 -> "The $first and the $second are on the table."
            4 -> "Put the $first behind the $second for now."
            5 -> "The $first is ready, but the $second is missing."
            6 -> "I wrote $first and $second on the note."
            7 -> "The $first, $second, and $third are in the room."
            8 -> "Before dinner, I moved the $first near the $second."
            else -> "The $first is between the $second and the $third near the $fourth."
        }.sentenceCase()
    }

    private fun bulkWakeSentence(
        phrase: String,
        lexicon: PromptLexicon,
        random: kotlin.random.Random,
        index: Int,
    ): String {
        val words = lexicon.randomContentWords(random, phrase, 3)
        val first = words.getOrElse(0) { "item" }
        val second = words.getOrElse(1) { "list" }
        return when ((index + random.nextInt(8)) % 8) {
            0 -> "After the $first is ready, say $phrase."
            1 -> "When the $second is finished, the next words are $phrase."
            2 -> "I will pause for a moment, then say $phrase."
            3 -> "The reminder is complete, so now I say $phrase."
            4 -> "Once the room is quiet, the wake phrase is $phrase."
            5 -> "After checking the $first, I can say $phrase."
            6 -> "The task is done, and the correct phrase is $phrase."
            else -> "When everything is ready, I say $phrase without rushing."
        }.sentenceCase()
    }

    private fun bulkHardNegativeSentence(
        nearMiss: String,
        lexicon: PromptLexicon,
        random: kotlin.random.Random,
        phrase: String,
    ): String {
        val word = lexicon.randomContentWords(random, phrase, 1).firstOrNull() ?: "note"
        return when (random.nextInt(4)) {
            0 -> "This is not the wake phrase, but I will say $nearMiss."
            1 -> "The near match for this $word is $nearMiss."
            2 -> "Ignore the similar phrase $nearMiss and keep reading."
            else -> "For a hard negative example, say $nearMiss."
        }.sentenceCase()
    }

    private fun hardNegativeCandidates(
        phrase: String,
        lexicon: PromptLexicon,
        random: kotlin.random.Random,
    ): List<String> {
        val words = phrase.trim().lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }
        val variants = mutableListOf<String>()
        variants.addAll(lexicon.phoneticHardNegatives(phrase, random, 10))

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

    private fun String.sentenceCase(): String {
        return trim()
            .replace(Regex("\\s+"), " ")
            .replaceFirstChar { it.uppercase() }
            .let { if (it.endsWith(".")) it else "$it." }
    }
}
