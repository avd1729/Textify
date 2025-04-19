package com.example.keyboard.ml

import android.content.Context
import android.hardware.usb.UsbDevice.getDeviceId
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

    /**
     * Uploads the model to the Flask backend server
     * @param serverUrl The base URL of the Flask server
     * @return Success status of the upload
     */
    suspend fun uploadModelToServer(serverUrl: String): Boolean {
        // First export the model to the app's files directory
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

                // Create OkHttp client with extended timeouts for large files
                val client = OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .writeTimeout(60, TimeUnit.SECONDS)
                    .readTimeout(60, TimeUnit.SECONDS)
                    .build()

                // Create multipart request body with the file
                val requestBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart(
                        "model_file",
                        file.name,
                        file.asRequestBody("application/octet-stream".toMediaTypeOrNull())
                    )
                    // Add device ID or user info if needed
                    .addFormDataPart("device_id", getDeviceId())
                    .build()

                // Build the request
                val request = Request.Builder()
                    .url("$serverUrl/upload_model")
                    .post(requestBody)
                    .build()

                // Execute the request
                val response = client.newCall(request).execute()
                val success = response.isSuccessful

                if (success) {
                    Log.d(TAG, "Model upload successful: ${response.body?.string()}")
                } else {
                    Log.e(TAG, "Model upload failed: ${response.code} - ${response.body?.string()}")
                }

                // Clean up the exported file after upload
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
            // Ensure vocabulary is exported as a dictionary
            predictor?.callAttr("prepare_for_export")

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

                // Build the request
                val request = Request.Builder()
                    .url("$serverUrl/download_aggregated_model")
                    .get()
                    .build()

                Log.d(TAG, "Requesting aggregated model from server")

                // Execute the request
                val response = client.newCall(request).execute()

                if (!response.isSuccessful) {
                    Log.e(TAG, "Failed to download aggregated model: ${response.code}")
                    return@withContext false
                }

                // Get the model data
                val modelBytes = response.body?.bytes() ?: run {
                    Log.e(TAG, "Empty response body")
                    return@withContext false
                }

                // Get the model file path where we should save it
                val modelPath = context.filesDir.absolutePath + "/keyboard_model.pkl"
                val file = File(modelPath)

                // Save the downloaded model directly to the expected model path
                file.writeBytes(modelBytes)

                Log.d(TAG, "Downloaded and saved aggregated model (${modelBytes.size} bytes) to $modelPath")

                // Create a new predictor instance with the updated model
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