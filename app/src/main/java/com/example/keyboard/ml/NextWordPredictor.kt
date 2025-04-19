package com.example.keyboard.ml

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.util.*
import java.util.concurrent.TimeUnit

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

    /**
     * Initializes the embedded Python interpreter and sets up the KeyboardPredictor instance
     * with a saved local model.
     */
    private fun initializePython() {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }

            val py = Python.getInstance()

            val filesDir = context.filesDir.absolutePath
            val modelPath = "$filesDir/keyboard_model.pkl"

            val assetManager = context.assets
            val sampleTexts = mutableListOf<String>()

            try {

                assetManager.list("sample_texts")?.forEach { fileName ->
                    val text = assetManager.open("sample_texts/$fileName").bufferedReader().use { it.readText() }
                    sampleTexts.add(text)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Could not load sample texts: ${e.message}")
            }

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
            throw e
        }
    }

    /**
     * Uploads the model to the Flask backend server
     * @param serverUrl The base URL of the Flask server
     * @return Success status of the upload
     */
    suspend fun uploadModelToServer(serverUrl: String): Boolean {
        val exportPath = exportModel(context.filesDir.absolutePath + "/export_model.pkl")
        if (exportPath == null) {
            Log.e(TAG, "Failed to export model for upload")
            return false
        }

        return withContext(Dispatchers.IO) {
            try {
                val file = File(exportPath)
                if (!file.exists()) {
                    Log.e(TAG, "Export file does not exist: $exportPath")
                    return@withContext false
                }

                Log.d(TAG, "Uploading model file, size: ${file.length()} bytes")

                val client = OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .writeTimeout(60, TimeUnit.SECONDS)
                    .readTimeout(60, TimeUnit.SECONDS)
                    .build()

                val requestBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart(
                        "model_file",
                        file.name,
                        file.asRequestBody("application/octet-stream".toMediaTypeOrNull())
                    )

                    .addFormDataPart("device_id", getDeviceId())
                    .build()

                val request = Request.Builder()
                    .url("$serverUrl/upload_model")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val success = response.isSuccessful

                if (success) {
                    Log.d(TAG, "Model upload successful: ${response.body?.string()}")
                } else {
                    Log.e(TAG, "Model upload failed: ${response.code} - ${response.body?.string()}")
                }

                file.delete()

                return@withContext success
            } catch (e: Exception) {
                Log.e(TAG, "Error uploading model", e)
                return@withContext false
            }
        }
    }

    /**
     * Gets a unique device ID for the model upload
     * You might want to use a more persistent ID in a real app
     */
    private fun getDeviceId(): String {
        val prefs = context.getSharedPreferences("keyboard_prefs", Context.MODE_PRIVATE)
        var deviceId = prefs.getString("device_id", null)

        if (deviceId == null) {
            deviceId = UUID.randomUUID().toString()
            prefs.edit().putString("device_id", deviceId).apply()
        }

        return deviceId
    }

    /**
     * Provides next-word predictions based on the user’s current input.
     *
     * @param text The input text string from the keyboard.
     * @return A list of predicted words (maximum of [maxSuggestions]).
     *         Returns fallback common words if prediction fails.
     */
    fun getPredictions(text: String): List<String> {
        val processedText = preprocessText(text)
        Log.d(TAG, "Processing text for predictions: '$processedText'")

        try {

            val predictions = predictor?.callAttr("predict", processedText, maxSuggestions)

            if (predictions != null) {

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

        return fallbackWords.shuffled().take(maxSuggestions)
    }

    /**
     * Cleans up the input text by removing punctuation and lowercasing.
     *
     * @param text Raw user input.
     * @return Preprocessed text string.
     */
    private fun preprocessText(text: String): String {
        return text.lowercase(Locale.getDefault())
            .replace(Regex("[^\\w\\s']"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    /**
     * Adds the current input text to the user’s local typing history.
     * This history is used later for local model training.
     *
     * @param text The string to store in the history.
     */
    fun addToHistory(text: String) {
        try {
            predictor?.callAttr("add_to_history", text)
            Log.d(TAG, "Added text to history: $text")
        } catch (e: Exception) {
            Log.e(TAG, "Error adding text to history", e)
        }
    }

    /**
     * Triggers local training using the accumulated user history.
     * This updates the model with personalized data.
     */
    fun updateLocalModel() {
        try {
            predictor?.callAttr("train_on_history")
            Log.d(TAG, "Updated local model with user history")
        } catch (e: Exception) {
            Log.e(TAG, "Error updating local model", e)
        }
    }

    /**
     * Prepares the model for server upload and saves it to disk.
     *
     * @param exportPath Optional custom export path (defaults to internal storage).
     * @return The path where the exported model is saved, or null if failed.
     */
    fun exportModel(exportPath: String? = null): String? {
        try {
            predictor?.callAttr("prepare_for_export")

            val path = predictor?.callAttr("export_for_aggregation", exportPath)?.toString()
            Log.d(TAG, "Exported model to: $path")
            return path
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting model", e)
            return null
        }
    }

    /**
     * Saves the current model state before the object is destroyed.
     * Recommended to call this on app shutdown or user logout.
     */
    fun close() {
        try {
            predictor?.callAttr("save_model")
            Log.d(TAG, "Model saved before closing")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing Python predictor", e)
        }
    }

    /**
     * Downloads the aggregated model from the server and replaces the local model
     * @param serverUrl The base URL of the Flask server
     * @return Success status of the download
     */
    suspend fun downloadAggregatedModel(serverUrl: String): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val client = OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(60, TimeUnit.SECONDS)
                    .build()

                val request = Request.Builder()
                    .url("$serverUrl/download_aggregated_model")
                    .get()
                    .build()

                Log.d(TAG, "Requesting aggregated model from server")

                val response = client.newCall(request).execute()

                if (!response.isSuccessful) {
                    Log.e(TAG, "Failed to download aggregated model: ${response.code}")
                    return@withContext false
                }

                val modelBytes = response.body?.bytes() ?: run {
                    Log.e(TAG, "Empty response body")
                    return@withContext false
                }

                val modelPath = context.filesDir.absolutePath + "/keyboard_model.pkl"
                val file = File(modelPath)

                file.writeBytes(modelBytes)

                Log.d(TAG, "Downloaded and saved aggregated model (${modelBytes.size} bytes) to $modelPath")

                try {
                    // Reinitialize Python to load the new model
                    initializePython()

                    Log.d(TAG, "Successfully reinitialized predictor with new model")
                    return@withContext true
                } catch (e: Exception) {
                    Log.e(TAG, "Error reinitializing predictor: ${e.message}")
                    return@withContext false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error downloading aggregated model", e)
                return@withContext false
            }
        }
    }
}