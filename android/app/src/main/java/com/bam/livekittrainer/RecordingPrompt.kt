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
        style: String = STYLE_PROSE,
    ): BulkScriptContent {
        val phrase = project.phrase
        val placementCount = wakePlacements.coerceIn(1, 48)
        val random = stableRandom("${project.slug}:bulk:$batchNumber:$revision")
        val hardNegatives = hardNegativeCandidates(
            phrase,
            lexicon,
            stableRandom("${project.slug}:bulk:$batchNumber:$revision:hard-negatives"),
        ).shuffled(random).toMutableList()
        when (style) {
            STYLE_STREAM -> return wordStreamScript(phrase, lexicon, random, placementCount, hardNegatives)
            STYLE_DENSE -> return denseBulkScript(phrase, lexicon, random, placementCount, hardNegatives)
        }
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

    /**
     * A positive-dense script: mostly short, prosodically varied ways to say the
     * wake phrase itself, with near-miss hard negatives folded in frequently and,
     * between each one, a *short* run of genuinely random words drawn from the
     * whole lexicon. Unlike the full word stream, the filler here is only a few
     * words per gap, so the take stays dense with positives and hard negatives
     * instead of turning into a long march of random words. The random filler
     * still gives each wake utterance real surrounding speech (good context for
     * the streaming-recall fix) without any fixed carrier sentence repeating.
     */
    private fun denseBulkScript(
        phrase: String,
        lexicon: PromptLexicon,
        random: kotlin.random.Random,
        placementCount: Int,
        hardNegativeSeed: List<String>,
    ): BulkScriptContent {
        val usedHardNegatives = mutableListOf<String>()
        // Cycle the limited near-miss pool, reshuffling when it runs dry so a long
        // script keeps producing hard negatives instead of stopping after ~10.
        val hardNegatives = hardNegativeSeed.toMutableList()
        fun nextHardNegative(): String? {
            if (hardNegatives.isEmpty()) {
                if (hardNegativeSeed.isEmpty()) return null
                hardNegatives.addAll(hardNegativeSeed.shuffled(random))
            }
            return hardNegatives.removeAt(0)
        }

        val lines = mutableListOf<String>()
        // A few genuinely random words between wake words — enough to surround the
        // wake with real speech, but nowhere near the long word-stream style.
        fun emitShortRun(wordCount: Int) {
            val words = lexicon.frequencyWeightedStream(phrase, random, wordCount)
            if (words.isNotEmpty()) lines.add(words.joinToString(" ").sentenceCase())
        }

        // One short warmup run so mic level settles before the first positive.
        emitShortRun(3 + random.nextInt(3))
        repeat(placementCount) { index ->
            lines.add(compactWakeLine(phrase, random, index))
            emitShortRun(3 + random.nextInt(4))
            // A near-miss after roughly every other positive, with its own short
            // random tail so it doesn't butt straight into the next wake.
            if (index % 2 == 1) {
                nextHardNegative()?.let { nearMiss ->
                    usedHardNegatives.add(nearMiss)
                    lines.add(compactHardNegativeLine(nearMiss, random))
                    emitShortRun(2 + random.nextInt(3))
                }
            }
        }
        return BulkScriptContent(lines.joinToString(" "), usedHardNegatives.distinct())
    }

    private fun compactWakeLine(
        phrase: String,
        random: kotlin.random.Random,
        index: Int,
    ): String {
        val templates = listOf(
            phrase,
            "Hey, $phrase.",
            "$phrase?",
            "Okay, $phrase.",
            "$phrase, please.",
            "$phrase!",
            "Say $phrase.",
            "Um, $phrase.",
            "$phrase, right now.",
            "Ready? $phrase.",
            "$phrase, thanks.",
            "Well, $phrase.",
        )
        return templates[(index + random.nextInt(templates.size)) % templates.size].compactCase()
    }

    private fun compactHardNegativeLine(
        nearMiss: String,
        random: kotlin.random.Random,
    ): String {
        val templates = listOf(
            nearMiss,
            "$nearMiss?",
            "$nearMiss!",
            "Not quite, $nearMiss.",
            "$nearMiss, almost.",
            "Hmm, $nearMiss.",
        )
        return templates[random.nextInt(templates.size)].compactCase()
    }

    const val DEFAULT_BULK_WAKE_PLACEMENTS = 8

    const val STYLE_PROSE = "prose"
    const val STYLE_DENSE = "dense"
    const val STYLE_STREAM = "stream"

    /**
     * A word-stream script: the speaker reads a long, ever-changing run of
     * ordinary words drawn frequency-weighted from the whole lexicon, with the
     * wake phrase (and the occasional near miss) dropped in at intervals. Because
     * every word is sampled fresh from thousands of candidates, no filler word
     * gets over-represented across takes the way the fixed carrier templates
     * did. Each wake utterance is surrounded by continuous real speech, which is
     * good context for the streaming-recall fix. Words are grouped into short
     * breath lines; the wake phrase and near misses stand alone so the aligner
     * can slice them cleanly.
     */
    private fun wordStreamScript(
        phrase: String,
        lexicon: PromptLexicon,
        random: kotlin.random.Random,
        placementCount: Int,
        hardNegativeSeed: List<String>,
    ): BulkScriptContent {
        val usedHardNegatives = mutableListOf<String>()
        val hardNegatives = hardNegativeSeed.toMutableList()
        fun nextHardNegative(): String? {
            if (hardNegatives.isEmpty()) {
                if (hardNegativeSeed.isEmpty()) return null
                hardNegatives.addAll(hardNegativeSeed.shuffled(random))
            }
            return hardNegatives.removeAt(0)
        }

        val lines = mutableListOf<String>()
        fun emitWordRun(wordCount: Int) {
            val words = lexicon.frequencyWeightedStream(phrase, random, wordCount)
            if (words.isEmpty()) return
            // Break the run into short breath groups so it reads at a natural
            // pace instead of one long unpunctuated mumble.
            var i = 0
            while (i < words.size) {
                val groupSize = 4 + random.nextInt(3)
                val group = words.subList(i, minOf(i + groupSize, words.size))
                lines.add(group.joinToString(" ").sentenceCase())
                i += groupSize
            }
        }

        // Lead-in speech so the mic settles and the first wake has real context.
        emitWordRun(10 + random.nextInt(8))
        repeat(placementCount) { index ->
            lines.add(phrase.trim().sentenceCase())
            emitWordRun(10 + random.nextInt(10))
            if (index % 2 == 1) {
                nextHardNegative()?.let { nearMiss ->
                    usedHardNegatives.add(nearMiss)
                    lines.add(nearMiss.trim().sentenceCase())
                    emitWordRun(6 + random.nextInt(6))
                }
            }
        }
        return BulkScriptContent(lines.joinToString(" "), usedHardNegatives.distinct())
    }

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
        val templates = listOf(
            "The $first is beside the $second.",
            "Please move the $first near the $second.",
            "I checked the $first before the $second.",
            "The $first and the $second are on the table.",
            "Put the $first behind the $second for now.",
            "The $first is ready, but the $second is missing.",
            "I wrote $first and $second on the note.",
            "The $first, $second, and $third are in the room.",
            "Before dinner, I moved the $first near the $second.",
            "The $first is between the $second and the $third near the $fourth.",
            "Where did you leave the $first yesterday?",
            "My $first fell behind the $second again.",
            "She carried the $first across the $second.",
            "Nobody remembered to close the $first.",
            "That $first has been sitting by the $second all week.",
            "Can you hand me the $first from the $second?",
            "Every morning I move the $first past the $second.",
            "He traded his old $first for a newer $second.",
            "Two $first and one $second went missing overnight.",
            "Somewhere under the $first, I found the $second.",
            "It rained so hard the $first soaked the $second.",
            "I can't believe the $first outlasted the $second.",
            "We stacked the $first on top of the $second.",
            "The $first rolled off the $second and stopped.",
        )
        return templates[random.nextInt(templates.size)].sentenceCase()
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
        val third = words.getOrElse(2) { "shelf" }
        val templates = listOf(
            "After the $first is ready, say $phrase.",
            "When the $second is finished, the next words are $phrase.",
            "I will pause for a moment, then say $phrase.",
            "The reminder is complete, so now I say $phrase.",
            "Once the room is quiet, the wake phrase is $phrase.",
            "After checking the $first, I can say $phrase.",
            "The task is done, and the correct phrase is $phrase.",
            "When everything is ready, I say $phrase without rushing.",
            "$phrase, I said, right after the $first settled.",
            "I leaned toward the $first and said $phrase.",
            "Sometimes I just say $phrase for no clear reason.",
            "$phrase is what comes out when the $second is near.",
            "Halfway through sorting the $first, I said $phrase.",
            "My favorite thing to try is $phrase, honestly.",
            "Right before lunch, the phrase I need is $phrase.",
            "I whispered $phrase while holding the $first.",
            "$phrase came out clearly once the $second stopped.",
            "Between the $first and the $third, I said $phrase.",
        )
        return templates[(index + random.nextInt(templates.size)) % templates.size].sentenceCase()
    }

    private fun bulkHardNegativeSentence(
        nearMiss: String,
        lexicon: PromptLexicon,
        random: kotlin.random.Random,
        phrase: String,
    ): String {
        val word = lexicon.randomContentWords(random, phrase, 1).firstOrNull() ?: "note"
        val templates = listOf(
            "This is not the wake phrase, but I will say $nearMiss.",
            "The near match for this $word is $nearMiss.",
            "Ignore the similar phrase $nearMiss and keep reading.",
            "For a hard negative example, say $nearMiss.",
            "It almost sounds right, but I am only saying $nearMiss.",
            "Careful, $nearMiss is close but not the one.",
            "The tricky look-alike here is $nearMiss.",
            "Don't be fooled, this is just $nearMiss.",
        )
        return templates[random.nextInt(templates.size)].sentenceCase()
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
        }

        return variants
            .map { it.trim().replace(Regex("\\s+"), " ") }
            .filter { it.isNotBlank() && it != phrase.lowercase() }
            .filterNot { candidate -> containsPhraseRun(candidate, words) }
            .distinct()
    }

    private fun containsPhraseRun(candidate: String, phraseWords: List<String>): Boolean {
        if (phraseWords.isEmpty()) return false
        val candidateWords = candidate.trim().lowercase()
            .split(Regex("\\s+"))
            .filter { it.isNotBlank() }
        if (candidateWords.size < phraseWords.size) return false
        for (start in 0..candidateWords.size - phraseWords.size) {
            if (candidateWords.subList(start, start + phraseWords.size) == phraseWords) {
                return true
            }
        }
        return false
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

    // Like sentenceCase but keeps existing terminal punctuation (? or !) so short
    // carrier lines can vary their intonation instead of all ending in a period.
    private fun String.compactCase(): String {
        val trimmed = trim().replace(Regex("\\s+"), " ").replaceFirstChar { it.uppercase() }
        return if (trimmed.isEmpty() || trimmed.last() in ".?!") trimmed else "$trimmed."
    }
}
