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
    private var vocabSize = 10000  // Will be updated from actual model
    private val maxSuggestions = 3  // Number of suggestions to provide
    private val vocabMap: MutableMap<String, Int> = HashMap()  // Word to index mapping
    private val indexMap: MutableMap<Int, String> = HashMap()  // Index to word mapping
    private var inputLength = 5  // Will be updated from actual model

    // Model dimensions
    private var actualInputSize = 0
    private var actualOutputSize = 0

    // User's typing history for local learning
    private val userHistory = mutableListOf<String>()
    private val historyLimit = 1000  // Limit history size

    // Frequency dictionary to track user's most common words
    private val frequencyDict = mutableMapOf<String, Int>()
    private val commonFollowingWords = mutableMapOf<String, MutableMap<String, Int>>()

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

            // Get model input/output dimensions
            val inputShape = interpreter?.getInputTensor(0)?.shape() ?: intArrayOf()
            val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: intArrayOf()

            // Update class fields based on model dimensions
            if (inputShape.isNotEmpty()) {
                inputLength = inputShape[1]  // Assuming shape is [batch_size, sequence_length]
                actualInputSize = inputShape[1]
                Log.d(TAG, "Model input shape: ${inputShape.joinToString()}, using sequence length: $inputLength")
            }

            if (outputShape.isNotEmpty()) {
                vocabSize = outputShape[1]  // Assuming shape is [batch_size, vocab_size]
                actualOutputSize = outputShape[1]
                Log.d(TAG, "Model output shape: ${outputShape.joinToString()}, using vocab size: $vocabSize")
            }

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

                        // Log sample of vocabulary
                        if (index < 10 || index % 1000 == 0) {
                            Log.d(TAG, "Vocab entry $index: $cleanWord")
                        }
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
        // Preprocess input text
        val processedText = preprocessText(text)
        Log.d(TAG, "Processing text for predictions: '$processedText'")

        // First try to get predictions from the TF model
        val modelPredictions = getTFPredictions(processedText)

        // Check if model predictions are varied enough
        if (modelPredictions.size >= 2 && !arePredictionsSimilar(modelPredictions)) {
            Log.d(TAG, "Using model predictions: $modelPredictions")
            return modelPredictions
        }

        // If model predictions are not varied enough, use frequency-based predictions
        val frequencyPredictions = getFrequencyBasedPredictions(processedText)
        Log.d(TAG, "Using frequency-based predictions: $frequencyPredictions")
        return frequencyPredictions
    }

    private fun preprocessText(text: String): String {
        return text.lowercase(Locale.getDefault())
            .replace(Regex("[^\\w\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun arePredictionsSimilar(predictions: List<String>): Boolean {
        // Consider predictions too similar if there are fewer than 2 distinct words
        return predictions.distinct().size < 2
    }

    private fun getTFPredictions(text: String): List<String> {
        try {
            // Extract last few words
            val words = text.split(" ", "\n", "\t").filter { it.isNotEmpty() }

            // Create input
            val inputBuffer = ByteBuffer.allocateDirect(inputLength * 4).order(ByteOrder.nativeOrder())

            // Log your input for debugging
            val wordIndices = mutableListOf<Int>()

            // Fill with zeros first
            for (i in 0 until inputLength) {
                inputBuffer.putInt(0)
                wordIndices.add(0)
            }

            // Reset position and add actual words (if any)
            inputBuffer.rewind()
            for (i in 0 until min(words.size, inputLength)) {
                val wordIndex = vocabMap[words[words.size - 1 - i]] ?: 0
                inputBuffer.putInt(i * 4, wordIndex)  // Direct position write
                wordIndices[i] = wordIndex
            }
            inputBuffer.rewind()

            Log.d(TAG, "Input word indices: $wordIndices")

            // Create output - make sure it's big enough
            val outputSize = interpreter?.getOutputTensor(0)?.shape()?.get(1) ?: vocabSize
            val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4).order(ByteOrder.nativeOrder())

            // Run inference
            interpreter?.run(inputBuffer, outputBuffer)

            // Process results - use output size from model
            outputBuffer.rewind()
            val results = FloatArray(outputSize)
            for (i in 0 until outputSize) {
                results[i] = outputBuffer.getFloat()
            }

            // Get any non-zero results
            Log.d(TAG, "Max prediction value: ${results.maxOrNull()}")
            Log.d(TAG, "Non-zero count: ${results.count { it > 0 }}")

            // Get top predictions with lower threshold
            return results.withIndex()
                .sortedByDescending { it.value }
                .take(maxSuggestions)
                .filter { it.value > 0.000001f }  // Very low threshold for debugging
                .mapNotNull { indexMap[it.index] }
                .also { Log.d(TAG, "Model predictions: $it") }
        } catch (e: Exception) {
            Log.e(TAG, "Error in TF predictions", e)
            return emptyList()
        }
    }

    private fun getFrequencyBasedPredictions(text: String): List<String> {
        // Extract the last word to predict what might follow it
        val lastWord = text.split(" ", "\n", "\t")
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
            .take(k * 2)  // Take more than needed to filter
            .filter { it.value > 0.01f } // Filter predictions with very low confidence
            .take(k)
            .map { it.index }

        // Map indices back to words
        val words = topIndices.mapNotNull {
            val word = indexMap[it]
            if (word != null) {
                Log.d(TAG, "Prediction: $word (${predictions[it]})")
            } else {
                Log.d(TAG, "No word found for index $it")
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