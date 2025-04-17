package com.example.keyboard.ml

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.util.*

class NextWordPredictor(private val context: Context) {
    private val TAG = "NextWordPredictor"
    private var predictor: PyObject? = null
    private val maxSuggestions = 3  // Number of suggestions to provide

    // Fallback common words for emergency cases
    private val fallbackWords = listOf(
        "the", "and", "to", "a", "of", "is", "in", "that", "it", "was",
        "for", "on", "with", "he", "as", "you", "do", "at", "this", "but"
    )

    init {
        try {
            initializePython()
            Log.d(TAG, "Python predictor initialization complete")
        } catch (e: Exception) {
            Log.e(TAG, "Error during Python predictor initialization", e)
        }
    }

    private fun initializePython() {
        try {
            // Make sure Python is started
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }

            // Now get the Python instance
            val py = Python.getInstance()

            // Get application's files directory for model storage
            val filesDir = context.filesDir.absolutePath
            val modelPath = "$filesDir/keyboard_model.pkl"

            // Check if we have sample texts to initialize the model
            val assetManager = context.assets
            val sampleTexts = mutableListOf<String>()

            try {
                // Load sample texts from assets if available
                assetManager.list("sample_texts")?.forEach { fileName ->
                    val text = assetManager.open("sample_texts/$fileName").bufferedReader().use { it.readText() }
                    sampleTexts.add(text)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Could not load sample texts: ${e.message}")
            }

            // Create KeyboardPredictor instance
            val module = py.getModule("keyboard_predictor")
            predictor = module.callAttr(
                "KeyboardPredictor",
                modelPath,
                3,  // n-gram size
                if (sampleTexts.isNotEmpty()) sampleTexts.toTypedArray() else null
            )

            Log.d(TAG, "Python predictor created with model path: $modelPath")
        } catch (e: Exception) {
            Log.e(TAG, "Error in initializePython", e)
            throw e  // Re-throw to ensure initialization failure is properly handled
        }
    }

    fun getPredictions(text: String): List<String> {
        val processedText = preprocessText(text)
        Log.d(TAG, "Processing text for predictions: '$processedText'")

        try {
            // Call the Python predict method
            val predictions = predictor?.callAttr("predict", processedText, maxSuggestions)

            if (predictions != null) {
                // Convert Python list of (word, probability) tuples to Kotlin list of words
                val result = mutableListOf<String>()
                val size = predictions.asList().size

                for (i in 0 until size) {
                    val tuple = predictions.asList()[i]
                    val word = tuple.asList()[0].toString()
                    result.add(word)
                }

                Log.d(TAG, "Python model predictions: $result")
                return result
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting predictions from Python model", e)
        }

        // Fallback to common words if prediction fails
        return fallbackWords.shuffled().take(maxSuggestions)
    }

    private fun preprocessText(text: String): String {
        return text.lowercase(Locale.getDefault())
            .replace(Regex("[^\\w\\s']"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    fun addToHistory(text: String) {
        try {
            predictor?.callAttr("add_to_history", text)
            Log.d(TAG, "Added text to history: $text")
        } catch (e: Exception) {
            Log.e(TAG, "Error adding text to history", e)
        }
    }

    fun updateLocalModel() {
        try {
            predictor?.callAttr("train_on_history")
            Log.d(TAG, "Updated local model with user history")
        } catch (e: Exception) {
            Log.e(TAG, "Error updating local model", e)
        }
    }

    fun exportModel(exportPath: String? = null): String? {
        try {
            val path = predictor?.callAttr("export_for_aggregation", exportPath)?.toString()
            Log.d(TAG, "Exported model to: $path")
            return path
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting model", e)
            return null
        }
    }

    fun close() {
        try {
            // Save the model before closing
            predictor?.callAttr("save_model")
            Log.d(TAG, "Model saved before closing")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing Python predictor", e)
        }
    }
}