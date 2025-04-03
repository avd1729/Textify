package com.example.keyboard.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

class NextWordPredictor(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val vocabSize = 10000  // Size of vocabulary - adjust based on your model
    private val maxSuggestions = 3  // Number of suggestions to provide
    private val vocabMap: MutableMap<String, Int> = HashMap()  // Word to index mapping
    private val indexMap: MutableMap<Int, String> = HashMap()  // Index to word mapping
    private val inputLength = 5  // Number of previous words to consider

    // User's typing history for local learning
    private val userHistory = mutableListOf<String>()
    private val historyLimit = 1000  // Limit history size

    init {
        try {
            loadModel()
            loadVocabulary()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadModel() {
        val modelFile = "next_word_model.tflite"
        interpreter = Interpreter(loadModelFile(context, modelFile))
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
        // Load vocabulary from a file in assets
        context.assets.open("vocabulary.txt").bufferedReader().useLines { lines ->
            lines.forEachIndexed { index, word ->
                vocabMap[word.trim()] = index
                indexMap[index] = word.trim()
            }
        }
    }

    fun getPredictions(text: String): List<String> {
        // Extract last few words for prediction context
        val words = text.split(" ", "\n", "\t", ".", ",", "!", "?")
            .filter { it.isNotEmpty() }
            .takeLast(inputLength)
            .map { it.lowercase(Locale.getDefault()) }

        if (words.isEmpty()) return emptyList()

        // Record this word in user history for later learning
        addToHistory(words.last())

        // Prepare input tensor
        val inputBuffer = ByteBuffer.allocateDirect(inputLength * 4)
            .order(ByteOrder.nativeOrder())

        // Add word indices to input buffer
        for (i in 0 until inputLength) {
            val wordIndex = if (i < words.size) {
                vocabMap[words[words.size - 1 - i]] ?: 0
            } else {
                0 // Padding for sentences shorter than inputLength
            }
            inputBuffer.putInt(wordIndex)
        }
        inputBuffer.rewind()

        // Prepare output buffer for predictions
        val outputBuffer = ByteBuffer.allocateDirect(vocabSize * 4)
            .order(ByteOrder.nativeOrder())

        // Run inference
        interpreter?.run(inputBuffer, outputBuffer)

        // Process results
        outputBuffer.rewind()
        val results = FloatArray(vocabSize)
        for (i in 0 until vocabSize) {
            results[i] = outputBuffer.getFloat()
        }

        // Get top predictions
        return getTopK(results, maxSuggestions)
    }

    private fun getTopK(predictions: FloatArray, k: Int): List<String> {
        // Find indices of top k predictions
        val topIndices = predictions.withIndex()
            .sortedByDescending { it.value }
            .take(k)
            .map { it.index }

        // Map indices back to words
        return topIndices.mapNotNull { indexMap[it] }
    }

    private fun addToHistory(word: String) {
        userHistory.add(word)
        if (userHistory.size > historyLimit) {
            userHistory.removeAt(0)
        }
    }

    // Method to update the model with local user data
    fun updateLocalModel() {
        // This would perform retraining or fine-tuning of the model
        // based on the collected user history

        // For now, we'll just log that it would happen
        println("Local model would be updated with ${userHistory.size} history items")

        // In a real implementation, you would:
        // 1. Convert user history to training examples
        // 2. Perform fine-tuning on the existing model
        // 3. Save the updated model
    }

    fun close() {
        interpreter?.close()
    }
}