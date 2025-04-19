package com.example.keyboard.service

import android.content.Intent
import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.view.LayoutInflater
import android.widget.Button
import android.widget.GridLayout
import android.widget.LinearLayout
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.keyboard.KeyboardActivity
import com.example.keyboard.R
import com.example.keyboard.ml.NextWordPredictor
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.*

class SecureKeyboardService : InputMethodService() {
    private val TAG = "SecureKeyboardService"
    private var isShiftEnabled = false
    private var isSymbolsEnabled = false
    private var currentText = StringBuilder()
    private lateinit var predictor: NextWordPredictor
    private lateinit var suggestionButtons: List<Button>
    private lateinit var keyboardView: View
    private var modelUpdateJob: Job? = null

    private val handler = Handler(Looper.getMainLooper())

    private var keyboardInitialized = false

    private var lastWord = ""
    private var wordBoundaryDetected = false

    /**
     * Initializes the keyboard service and prediction model.
     * Sets up federated learning by downloading the latest aggregated model and uploading local model.
     */
    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate")
        try {
            if (!ensurePythonInitialized()) {
                launchInitActivity()
                return
            }

            predictor = NextWordPredictor(this)

            val serverUrl = "https://zyphor-fl.onrender.com"

            // Download the latest aggregated model first, then upload local model
            CoroutineScope(Dispatchers.Main).launch {
                // Try to download the aggregated model first
                val downloadSuccess = predictor.downloadAggregatedModel(serverUrl)
                Log.d(TAG, "Initial model download ${if (downloadSuccess) "successful" else "failed"}")

                // Then upload the local model (which might now include the downloaded data)
                val uploadSuccess = predictor.uploadModelToServer(serverUrl)
                Log.d(TAG, "Model upload ${if (uploadSuccess) "successful" else "failed"}")
            }

            // Schedule periodic model updates
            scheduleModelUpdates(serverUrl)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing predictor", e)
            launchInitActivity()
        }
    }

    /**
     * Ensures Python is initialized for prediction model usage.
     * @return Boolean indicating if Python was successfully initialized.
     */
    private fun ensurePythonInitialized(): Boolean {
        return try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(this))
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Python initialization failed", e)
            false
        }
    }

    /**
     * Launches the initialization activity in case Python is not initialized.
     * Displays an error if initialization fails.
     */
    private fun launchInitActivity() {
        try {
            val intent = Intent(this, KeyboardActivity::class.java)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch init activity", e)
        }
    }

    override fun onDestroy() {
        Log.d(TAG, "onDestroy")
        modelUpdateJob?.cancel()
        if (::predictor.isInitialized) {
            predictor.close()
        }
        super.onDestroy()
    }

    /**
     * Schedules periodic updates for the model from the server.
     * @param serverUrl The server URL where models are updated from.
     */
    private fun scheduleModelUpdates(serverUrl: String) {
        // Cancel any existing job
        modelUpdateJob?.cancel()

        // Create a new job for periodic model updates
        modelUpdateJob = CoroutineScope(Dispatchers.IO).launch {
            while (isActive) {
                try {
                    // Check for model updates every 24 hours
                    delay(TimeUnit.HOURS.toMillis(24))

                    Log.d(TAG, "Checking for model updates")
                    if (::predictor.isInitialized) {
                        val success = predictor.downloadAggregatedModel(serverUrl)
                        Log.d(TAG, "Model update ${if (success) "successful" else "failed"}")

                        // If successful, update suggestions with new model
                        if (success) {
                            withContext(Dispatchers.Main) {
                                updateSuggestions()
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in scheduled model update", e)
                }
            }
        }
    }

    /**
     * Creates and returns the keyboard view when the input method is displayed.
     * @return The keyboard view to be displayed.
     */
    override fun onCreateInputView(): View {
        Log.d(TAG, "onCreateInputView")
        if (!::keyboardView.isInitialized) {
            keyboardView = LayoutInflater.from(this).inflate(R.layout.keyboard_layout, null)
            initializeKeyboard()
        }
        return keyboardView
    }

    /**
     * Initializes the keyboard layout and suggestion bar if not already initialized.
     * Sets up the keyboard UI components.
     */
    private fun initializeKeyboard() {
        Log.d(TAG, "initializeKeyboard")
        if (keyboardInitialized) return

        try {
            setupKeyboard(keyboardView)
            setupSuggestionBar(keyboardView)
            keyboardInitialized = true
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing keyboard", e)
        }
    }

    /**
     * Called when the input editing session starts.
     * Resets text buffers and suggestions when not restarting.
     * @param attribute The attributes of the edit field being edited
     * @param restarting Whether we are restarting input on the same text field
     */
    override fun onStartInput(attribute: android.view.inputmethod.EditorInfo?, restarting: Boolean) {
        super.onStartInput(attribute, restarting)
        Log.d(TAG, "onStartInput: restarting=$restarting")

        if (!restarting) {
            currentText.clear()
            lastWord = ""
            wordBoundaryDetected = false
            handler.post { clearSuggestions() }
        }
    }

    /**
     * Called when the input editing session is completed.
     * Adds the last word to history and updates the local model.
     */
    override fun onFinishInput() {
        Log.d(TAG, "onFinishInput")
        if (::predictor.isInitialized && lastWord.isNotEmpty()) {
            predictor.addToHistory(lastWord)
            lastWord = ""
        }

        if (::predictor.isInitialized) {
            handler.post { updateLocalModel() }
        }

        super.onFinishInput()
    }

    /**
     * Called when the text selection changes in the editing field.
     * Updates current text state and refreshes suggestions.
     */
    override fun onUpdateSelection(oldSelStart: Int, oldSelEnd: Int, newSelStart: Int,
                                   newSelEnd: Int, candidatesStart: Int, candidatesEnd: Int) {
        super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd)
        Log.d(TAG, "onUpdateSelection: newSelStart=$newSelStart newSelEnd=$newSelEnd")

        try {
            val ic = currentInputConnection
            if (ic != null) {
                val textBeforeCursor = ic.getTextBeforeCursor(100, 0)?.toString() ?: ""
                currentText = StringBuilder(textBeforeCursor)
                if (::predictor.isInitialized) {
                    handler.post { updateSuggestions() }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in onUpdateSelection", e)
        }
    }

    /**
     * Sets up the suggestion bar with word prediction buttons.
     * Configures click listeners for each suggestion button.
     * @param view The root keyboard view
     */
    private fun setupSuggestionBar(view: View) {
        Log.d(TAG, "setupSuggestionBar")
        try {
            val suggestionBar = view.findViewById<LinearLayout>(R.id.suggestion_bar)
            suggestionButtons = listOf(
                view.findViewById(R.id.suggestion_1),
                view.findViewById(R.id.suggestion_2),
                view.findViewById(R.id.suggestion_3)
            )

            suggestionButtons.forEach { button ->
                button.setOnClickListener {
                    val word = button.text.toString()
                    if (word.isNotEmpty()) {
                        if (currentText.isNotEmpty() && currentText.last() != ' ') {
                            currentInputConnection?.commitText(" ", 1)
                            currentText.append(" ")
                        }

                        currentInputConnection?.commitText(word, 1)
                        currentText.append(word)

                        if (::predictor.isInitialized) {
                            predictor.addToHistory(word)
                        }

                        if (::predictor.isInitialized) {
                            handler.post { updateSuggestions() }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in setupSuggestionBar", e)
        }
    }

    /**
     * Sets up the main keyboard layout with appropriate key set.
     * Configures special keys (space, shift, symbols, backspace, enter).
     * @param view The root keyboard view
     */
    private fun setupKeyboard(view: View) {
        Log.d(TAG, "setupKeyboard")
        try {
            val keyboardContainer: GridLayout = view.findViewById(R.id.keyboard_container)

            keyboardContainer.removeAllViews()

            if (isSymbolsEnabled) {
                setupSymbolKeyboard(keyboardContainer)
            } else {
                setupAlphaKeyboard(keyboardContainer)
            }

            val spaceButton: Button = view.findViewById(R.id.button_space)
            spaceButton.setOnClickListener {
                handleKeyPress(" ")
            }

            val shiftButton: Button = view.findViewById(R.id.button_shift)
            shiftButton.setOnClickListener {
                isShiftEnabled = !isShiftEnabled
                updateKeyLabels(keyboardContainer)
            }

            val symbolsButton: Button = view.findViewById(R.id.button_symbols)
            symbolsButton.setOnClickListener {
                isSymbolsEnabled = !isSymbolsEnabled
                setupKeyboard(view)

                symbolsButton.text = if (isSymbolsEnabled) "ABC" else "\\?123"
            }

            val backspaceButton: Button = view.findViewById(R.id.button_backspace)
            backspaceButton.setOnClickListener {
                currentInputConnection?.deleteSurroundingText(1, 0)

                if (currentText.isNotEmpty()) {
                    currentText.deleteCharAt(currentText.length - 1)
                    if (::predictor.isInitialized) {
                        handler.post { updateSuggestions() }
                    }
                }
            }

            val enterButton: Button = view.findViewById(R.id.button_enter)
            enterButton.setOnClickListener {
                handleKeyPress("\n")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in setupKeyboard", e)
        }
    }

    /**
     * Creates and configures the alphabetic keyboard layout.
     * @param keyboardContainer The container to add key buttons to
     */
    private fun setupAlphaKeyboard(keyboardContainer: GridLayout) {
        val keys = listOf(
            "q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
            "a", "s", "d", "f", "g", "h", "j", "k", "l",
            "z", "x", "c", "v", "b", "n", "m"
        )

        for (key in keys) {
            val button = createKeyButton(keyboardContainer.context, key)
            keyboardContainer.addView(button)
        }
    }

    /**
     * Creates and configures the symbols keyboard layout.
     * @param keyboardContainer The container to add symbol buttons to
     */
    private fun setupSymbolKeyboard(keyboardContainer: GridLayout) {
        val symbols = listOf(
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            "@", "#", "$", "%", "&", "-", "+", "(", ")",
            "!", "\"", "'", ":", ";", "/", "?", "*"
        )

        for (symbol in symbols) {
            val button = createKeyButton(keyboardContainer.context, symbol)
            keyboardContainer.addView(button)
        }
    }

    /**
     * Creates a keyboard button with the specified text and styling.
     * @param context The context used to create the button
     * @param text The text to display on the button
     * @return A styled button ready to be added to the keyboard
     */
    private fun createKeyButton(context: android.content.Context, text: String): Button {
        return Button(context).apply {
            this.text = if (isShiftEnabled && !isSymbolsEnabled) text.uppercase() else text
            textSize = 18f
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundResource(R.drawable.key_background)

            val params = GridLayout.LayoutParams()
            params.width = 0
            params.height = android.view.ViewGroup.LayoutParams.WRAP_CONTENT
            params.setMargins(4, 4, 4, 4)
            params.columnSpec = GridLayout.spec(GridLayout.UNDEFINED, 1f)
            layoutParams = params

            minimumHeight = resources.getDimensionPixelSize(android.R.dimen.app_icon_size)

            setOnClickListener {
                val keyText = if (isShiftEnabled && !isSymbolsEnabled) text.uppercase() else text
                handleKeyPress(keyText)
                if (isShiftEnabled) {
                    isShiftEnabled = false
                    updateKeyLabels(parent as GridLayout)
                }
            }
        }
    }

    /**
     * Updates the letter case on keyboard keys based on shift state.
     * Only affects alphabet keys when in alphabet mode.
     * @param keyboardContainer The container holding the key buttons
     */
    private fun updateKeyLabels(keyboardContainer: GridLayout) {
        try {
            // Only update if we're in alphabet mode
            if (isSymbolsEnabled) return

            for (i in 0 until keyboardContainer.childCount) {
                val child = keyboardContainer.getChildAt(i)
                if (child is Button) {
                    val buttonText = child.text.toString()
                    if (buttonText.length == 1 && buttonText[0].isLetter()) {
                        child.text = if (isShiftEnabled) buttonText.uppercase() else buttonText.lowercase()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in updateKeyLabels", e)
        }
    }

    /**
     * Handles key press events for text input.
     * Updates current text state, tracks word boundaries, and refreshes suggestions.
     * @param text The text to be inserted
     */
    private fun handleKeyPress(text: String) {
        try {
            val inputConnection = currentInputConnection
            inputConnection?.commitText(text, 1)

            // Update our tracking of the current text
            currentText.append(text)

            // If a word boundary is detected (space, newline, punctuation)
            if (text == " " || text == "\n" || text.matches(Regex("[.,!?;:]"))) {
                // Add the last word to history if there is one
                if (lastWord.isNotEmpty() && ::predictor.isInitialized) {
                    predictor.addToHistory(lastWord)
                    lastWord = ""
                }
                wordBoundaryDetected = true
            } else if (wordBoundaryDetected) {
                // Start a new word
                lastWord = text
                wordBoundaryDetected = false
            } else {
                // Continue the current word
                lastWord += text
            }

            // After each keypress, update suggestions on a background thread
            if (text != "\n" && ::predictor.isInitialized) {
                handler.post { updateSuggestions() }
            } else {
                // If enter was pressed, clear current text tracking
                currentText.clear()
                handler.post { clearSuggestions() }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in handleKeyPress", e)
        }
    }

    /**
     * Updates the suggestion buttons with predicted words.
     * Gets predictions from the model based on current text context.
     */
    private fun updateSuggestions() {
        try {
            if (!::predictor.isInitialized) return

            val predictions = predictor.getPredictions(currentText.toString())

            if (::suggestionButtons.isInitialized) {
                for (i in suggestionButtons.indices) {
                    val button = suggestionButtons[i]
                    if (i < predictions.size) {
                        button.text = predictions[i]
                        button.visibility = View.VISIBLE
                    } else {
                        button.text = ""
                        button.visibility = View.INVISIBLE
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in updateSuggestions", e)
        }
    }

    /**
     * Clears all word suggestions from the suggestion bar.
     * Hides suggestion buttons when no predictions are available.
     */
    private fun clearSuggestions() {
        try {
            if (::suggestionButtons.isInitialized) {
                suggestionButtons.forEach { button ->
                    button.text = ""
                    button.visibility = View.INVISIBLE
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in clearSuggestions", e)
        }
    }

    /**
     * Updates the local prediction model with new learning data.
     * Should be called after input sessions to improve predictions.
     */
    private fun updateLocalModel() {
        if (::predictor.isInitialized) {
            predictor.updateLocalModel()
        }
    }
}