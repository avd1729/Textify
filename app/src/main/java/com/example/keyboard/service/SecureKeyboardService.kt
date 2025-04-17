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

class SecureKeyboardService : InputMethodService() {
    private val TAG = "SecureKeyboardService"
    private var isShiftEnabled = false
    private var isSymbolsEnabled = false
    private var currentText = StringBuilder()
    private lateinit var predictor: NextWordPredictor
    private lateinit var suggestionButtons: List<Button>
    private lateinit var keyboardView: View

    // Handler for background operations
    private val handler = Handler(Looper.getMainLooper())

    // Flag to prevent recreating the keyboard on every key press
    private var keyboardInitialized = false

    // Track the last word for adding to history
    private var lastWord = ""
    private var wordBoundaryDetected = false

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate")
        try {
            // Ensure Python is initialized before creating the predictor
            if (!ensurePythonInitialized()) {
                // Try to launch the activity to initialize Python
                launchInitActivity()
                return
            }

            // Initialize the Python predictor
            predictor = NextWordPredictor(this)
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing predictor", e)
            // Try to launch the activity to initialize Python
            launchInitActivity()
        }

        val serverUrl = "https://zyphor-fl.onrender.com"   // Replace with your actual server URL
        CoroutineScope(Dispatchers.Main).launch {
            val success = predictor.uploadModelToServer(serverUrl)
            if (success) {
                Log.d(TAG, "Model upload successful")
            } else {
                Log.e(TAG, "Model upload failed")
            }
        }
    }

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
        if (::predictor.isInitialized) {
            predictor.close()
        }
        super.onDestroy()
    }

    override fun onCreateInputView(): View {
        Log.d(TAG, "onCreateInputView")
        // Only inflate the layout if we haven't already
        if (!::keyboardView.isInitialized) {
            keyboardView = LayoutInflater.from(this).inflate(R.layout.keyboard_layout, null)
            initializeKeyboard()
        }
        return keyboardView
    }

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

    override fun onStartInput(attribute: android.view.inputmethod.EditorInfo?, restarting: Boolean) {
        super.onStartInput(attribute, restarting)
        Log.d(TAG, "onStartInput: restarting=$restarting")

        if (!restarting) {
            currentText.clear()
            lastWord = ""
            wordBoundaryDetected = false
            // Update suggestions in the next frame to avoid UI freezes
            handler.post { clearSuggestions() }
        }
    }

    override fun onFinishInput() {
        Log.d(TAG, "onFinishInput")
        // Save any pending word to history
        if (::predictor.isInitialized && lastWord.isNotEmpty()) {
            predictor.addToHistory(lastWord)
            lastWord = ""
        }

        // Update model with accumulated history
        if (::predictor.isInitialized) {
            handler.post { updateLocalModel() }
        }

        super.onFinishInput()
    }

    // Called when text selection changes
    override fun onUpdateSelection(oldSelStart: Int, oldSelEnd: Int, newSelStart: Int,
                                   newSelEnd: Int, candidatesStart: Int, candidatesEnd: Int) {
        super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd)
        Log.d(TAG, "onUpdateSelection: newSelStart=$newSelStart newSelEnd=$newSelEnd")

        // Extract current text to keep prediction context accurate
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

    private fun setupSuggestionBar(view: View) {
        Log.d(TAG, "setupSuggestionBar")
        try {
            val suggestionBar = view.findViewById<LinearLayout>(R.id.suggestion_bar)
            suggestionButtons = listOf(
                view.findViewById(R.id.suggestion_1),
                view.findViewById(R.id.suggestion_2),
                view.findViewById(R.id.suggestion_3)
            )

            // Set up click listeners for suggestion buttons
            suggestionButtons.forEach { button ->
                button.setOnClickListener {
                    val word = button.text.toString()
                    if (word.isNotEmpty()) {
                        // Add space if needed
                        if (currentText.isNotEmpty() && currentText.last() != ' ') {
                            currentInputConnection?.commitText(" ", 1)
                            currentText.append(" ")
                        }

                        // Insert the suggested word
                        currentInputConnection?.commitText(word, 1)
                        currentText.append(word)

                        // Save this word to history
                        if (::predictor.isInitialized) {
                            predictor.addToHistory(word)
                        }

                        // Update suggestions after selection
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

    private fun setupKeyboard(view: View) {
        Log.d(TAG, "setupKeyboard")
        try {
            val keyboardContainer: GridLayout = view.findViewById(R.id.keyboard_container)

            // Clear any existing views
            keyboardContainer.removeAllViews()

            if (isSymbolsEnabled) {
                setupSymbolKeyboard(keyboardContainer)
            } else {
                setupAlphaKeyboard(keyboardContainer)
            }

            // Space button logic
            val spaceButton: Button = view.findViewById(R.id.button_space)
            spaceButton.setOnClickListener {
                handleKeyPress(" ")
            }

            // Shift button logic
            val shiftButton: Button = view.findViewById(R.id.button_shift)
            shiftButton.setOnClickListener {
                isShiftEnabled = !isShiftEnabled
                updateKeyLabels(keyboardContainer)
            }

            // Symbols button logic
            val symbolsButton: Button = view.findViewById(R.id.button_symbols)
            symbolsButton.setOnClickListener {
                isSymbolsEnabled = !isSymbolsEnabled
                setupKeyboard(view)

                // Update the symbol button text
                symbolsButton.text = if (isSymbolsEnabled) "ABC" else "\\?123"
            }

            // Backspace button logic
            val backspaceButton: Button = view.findViewById(R.id.button_backspace)
            backspaceButton.setOnClickListener {
                currentInputConnection?.deleteSurroundingText(1, 0)

                // Update current text
                if (currentText.isNotEmpty()) {
                    currentText.deleteCharAt(currentText.length - 1)
                    if (::predictor.isInitialized) {
                        handler.post { updateSuggestions() }
                    }
                }
            }

            // Enter button logic
            val enterButton: Button = view.findViewById(R.id.button_enter)
            enterButton.setOnClickListener {
                handleKeyPress("\n")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in setupKeyboard", e)
        }
    }

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

    private fun createKeyButton(context: android.content.Context, text: String): Button {
        return Button(context).apply {
            this.text = if (isShiftEnabled && !isSymbolsEnabled) text.uppercase() else text
            textSize = 18f
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundResource(R.drawable.key_background)

            // Set layout params for consistent key size
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

    private fun updateSuggestions() {
        try {
            if (!::predictor.isInitialized) return

            // Get predictions based on current text
            val predictions = predictor.getPredictions(currentText.toString())

            // Update suggestion buttons if they're initialized
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

    // Call this method to update the local model with user data
    private fun updateLocalModel() {
        if (::predictor.isInitialized) {
            predictor.updateLocalModel()
        }
    }
}