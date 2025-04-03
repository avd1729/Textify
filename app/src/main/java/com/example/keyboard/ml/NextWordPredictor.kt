package com.example.keyboard.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.min

class NextWordPredictor(private val context: Context) {
    private val TAG = "NextWordPredictor"
    private var interpreter: Interpreter? = null
    private val vocabSize = 10000  // Size of vocabulary - adjust based on your model
    private val maxSuggestions = 3  // Number of suggestions to provide
    private val vocabMap: MutableMap<String, Int> = HashMap()  // Word to index mapping
    private val indexMap: MutableMap<Int, String> = HashMap()  // Index to word mapping
    private val inputLength = 5  // Number of previous words to consider

    // User's typing history for local learning
    private val userHistory = mutableListOf<String>()
    private val historyLimit = 1000  // Limit history size

    // Frequency dictionary to track user's most common words
    private val frequencyDict = mutableMapOf<String, Int>()
    private val commonFollowingWords = mutableMapOf<String, MutableMap<String, Int>>()

    // Reusable buffers to reduce allocation
    private val inputBuffer = ByteBuffer.allocateDirect(inputLength * 4).order(ByteOrder.nativeOrder())
    private val outputBuffer = ByteBuffer.allocateDirect(vocabSize * 4).order(ByteOrder.nativeOrder())
    private val results = FloatArray(vocabSize)

    // Fallback common words
    private val fallbackWords = listOf(
        "the", "and", "to", "a", "of", "is", "in", "that", "it", "was",
        "for", "on", "with", "he", "as", "you", "do", "at", "this", "but",
        "his", "by", "from", "they", "we", "say", "her", "she", "or", "an",
        "will", "my", "all", "would", "there", "their", "what", "so", "up", "if"
    )

    init {
        try {
            loadModel()
            loadVocabulary()
            Log.d(TAG, "Initialization complete. Vocabulary loaded: ${vocabMap.size} words")
        } catch (e: Exception) {
            Log.e(TAG, "Error during initialization", e)
        }
    }

    private fun loadModel() {
        val modelFile = "next_word_model.tflite"
        try {
            // Create interpreter options
            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }

            // Load the model with options
            interpreter = Interpreter(loadModelFile(context, modelFile), options)
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}")
            throw e
        }
    }

    private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadVocabulary() {
        try {
            var count = 0
            context.assets.open("vocabulary.txt").bufferedReader().useLines { lines ->
                lines.forEachIndexed { index, word ->
                    val cleanWord = word.trim()
                    if (cleanWord.isNotEmpty()) {
                        vocabMap[cleanWord] = index
                        indexMap[index] = cleanWord
                        count++
                    }
                }
            }
            Log.d(TAG, "Loaded $count words from vocabulary")

            // Initialize frequency dictionary with common words
            fallbackWords.forEach { frequencyDict[it] = 1 }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading vocabulary", e)
            throw e
        }
    }

    fun getPredictions(text: String): List<String> {
        // First try to get predictions from the TF model
        val modelPredictions = getTFPredictions(text)

        // Check if model predictions are varied enough
        if (modelPredictions.size >= 2 && !arePredictionsSimilar(modelPredictions)) {
            Log.d(TAG, "Using model predictions: $modelPredictions")
            return modelPredictions
        }

        // If model predictions are not varied enough, use frequency-based predictions
        val frequencyPredictions = getFrequencyBasedPredictions(text)
        Log.d(TAG, "Using frequency-based predictions: $frequencyPredictions")
        return frequencyPredictions
    }

    private fun arePredictionsSimilar(predictions: List<String>): Boolean {
        // Consider predictions too similar if there are fewer than 2 distinct words
        return predictions.distinct().size < 2
    }

    private fun getTFPredictions(text: String): List<String> {
        try {
            // Extract last few words for prediction context
            val words = text.split(" ", "\n", "\t", ".", ",", "!", "?")
                .filter { it.isNotEmpty() }
                .takeLast(inputLength)
                .map { it.lowercase(Locale.getDefault()) }

            if (words.isEmpty()) return emptyList()

            Log.d(TAG, "Input words for model: $words")

            // Record this word in user history for later learning
            if (words.isNotEmpty()) {
                addToHistory(words.last())
            }

            // Clear and prepare input buffer
            inputBuffer.clear()

            // Add word indices to input buffer (in reverse order)
            for (i in 0 until inputLength) {
                val wordIndex = if (i < words.size) {
                    vocabMap[words[words.size - 1 - i]] ?: 0
                } else {
                    0 // Padding for sentences shorter than inputLength
                }
                inputBuffer.putInt(wordIndex)
            }
            inputBuffer.rewind()

            // Clear output buffer
            outputBuffer.clear()

            // Run inference
            interpreter?.run(inputBuffer, outputBuffer) ?: return emptyList()

            // Process results
            outputBuffer.rewind()
            for (i in 0 until vocabSize) {
                results[i] = outputBuffer.getFloat()
            }

            // Get top predictions
            val topPredictions = getTopK(results, maxSuggestions)
            Log.d(TAG, "TF model predictions: $topPredictions")
            return topPredictions
        } catch (e: Exception) {
            Log.e(TAG, "Error in TF predictions", e)
            return emptyList()
        }
    }

    private fun getFrequencyBasedPredictions(text: String): List<String> {
        // Extract the last word to predict what might follow it
        val lastWord = text.split(" ", "\n", "\t", ".", ",", "!", "?")
            .filter { it.isNotEmpty() }
            .lastOrNull()?.lowercase(Locale.getDefault()) ?: ""

        // Try to get words that commonly follow the last word
        if (lastWord.isNotEmpty() && commonFollowingWords.containsKey(lastWord)) {
            val followingWords = commonFollowingWords[lastWord]?.entries
                ?.sortedByDescending { it.value }
                ?.take(maxSuggestions)
                ?.map { it.key }

            if (!followingWords.isNullOrEmpty() && followingWords.size >= min(2, maxSuggestions)) {
                return followingWords
            }
        }

        // Fall back to most frequent overall words
        val frequentWords = frequencyDict.entries
            .sortedByDescending { it.value }
            .take(maxSuggestions * 2)
            .map { it.key }
            .shuffled()
            .take(maxSuggestions)

        if (frequentWords.isNotEmpty()) {
            return frequentWords
        }

        // Ultimate fallback - static common words
        return fallbackWords.shuffled().take(maxSuggestions)
    }

    private fun getTopK(predictions: FloatArray, k: Int): List<String> {
        // Find indices of top k predictions
        val topIndices = predictions.withIndex()
            .sortedByDescending { it.value }
            .take(k)
            .filter { it.value > 0.01f } // Filter predictions with very low confidence
            .map { it.index }

        // Map indices back to words
        val words = topIndices.mapNotNull {
            val word = indexMap[it]
            if (word != null) {
                Log.d(TAG, "Prediction: $word (${predictions[it]})")
            }
            word
        }

        return words
    }

    private fun addToHistory(word: String) {
        if (word.length < 2) return // Skip very short words

        // Add to user history
        userHistory.add(word)
        if (userHistory.size > historyLimit) {
            userHistory.removeAt(0)
        }

        // Update frequency dictionary
        frequencyDict[word] = (frequencyDict[word] ?: 0) + 1

        // Update following words map
        if (userHistory.size >= 2) {
            val previousWord = userHistory[userHistory.size - 2]
            val followingWordsForPrevious = commonFollowingWords.getOrPut(previousWord) { mutableMapOf() }
            followingWordsForPrevious[word] = (followingWordsForPrevious[word] ?: 0) + 1
        }
    }

    // Method to update the model with local user data
    fun updateLocalModel() {
        try {
            // Save frequency data
            Log.d(TAG, "Local model would be updated with ${userHistory.size} history items")
            Log.d(TAG, "Frequency dictionary contains ${frequencyDict.size} words")
            Log.d(TAG, "Contextual following words map contains ${commonFollowingWords.size} contexts")

            // In a real implementation, you would:
            // 1. Convert user history to training examples
            // 2. Perform fine-tuning on the existing model
            // 3. Save the updated model
        } catch (e: Exception) {
            Log.e(TAG, "Error updating local model", e)
        }
    }

    // Get user's most frequent words for debugging
    fun getTopUserWords(count: Int = 10): List<Pair<String, Int>> {
        return frequencyDict.entries
            .sortedByDescending { it.value }
            .take(count)
            .map { it.key to it.value }
    }

    fun close() {
        try {
            interpreter?.close()
            Log.d(TAG, "Interpreter closed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing interpreter", e)
        }
    }
}