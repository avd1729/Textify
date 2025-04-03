package com.example.keyboard.service

import android.inputmethodservice.InputMethodService
import android.view.View
import android.view.LayoutInflater
import android.view.inputmethod.InputConnection
import android.widget.Button
import android.widget.GridLayout
import android.widget.LinearLayout
import android.widget.TextView
import com.example.keyboard.R
import com.example.keyboard.ml.NextWordPredictor

class SecureKeyboardService : InputMethodService() {
    private var isShiftEnabled = false
    private var isSymbolsEnabled = false
    private var currentText = StringBuilder()
    private lateinit var predictor: NextWordPredictor
    private lateinit var suggestionButtons: List<Button>

    override fun onCreate() {
        super.onCreate()
        predictor = NextWordPredictor(this)
    }

    override fun onDestroy() {
        predictor.close()
        super.onDestroy()
    }

    override fun onCreateInputView(): View {
        // Inflate custom keyboard layout
        val view = LayoutInflater.from(this).inflate(R.layout.keyboard_layout, null)
        setupKeyboard(view)
        setupSuggestionBar(view)
        return view
    }

    private fun setupSuggestionBar(view: View) {
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

                    // Update suggestions after selection
                    updateSuggestions()
                }
            }
        }
    }

    private fun setupKeyboard(view: View) {
        val inputConnection = currentInputConnection
        val keyboardContainer: GridLayout = view.findViewById(R.id.keyboard_container)

        // Clear any existing views
        keyboardContainer.removeAllViews()

        if (isSymbolsEnabled) {
            setupSymbolKeyboard(keyboardContainer, inputConnection)
        } else {
            setupAlphaKeyboard(keyboardContainer, inputConnection)
        }

        // Space button logic
        val spaceButton: Button = view.findViewById(R.id.button_space)
        spaceButton.setOnClickListener {
            handleKeyPress(inputConnection, " ")
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
            inputConnection?.deleteSurroundingText(1, 0)

            // Update current text
            if (currentText.isNotEmpty()) {
                currentText.deleteCharAt(currentText.length - 1)
                updateSuggestions()
            }
        }

        // Enter button logic
        val enterButton: Button = view.findViewById(R.id.button_enter)
        enterButton.setOnClickListener {
            handleKeyPress(inputConnection, "\n")
        }
    }

    private fun setupAlphaKeyboard(keyboardContainer: GridLayout, inputConnection: InputConnection?) {
        val keys = listOf(
            "q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
            "a", "s", "d", "f", "g", "h", "j", "k", "l",
            "z", "x", "c", "v", "b", "n", "m"
        )

        for (key in keys) {
            val button = createKeyButton(keyboardContainer.context, key, inputConnection)
            keyboardContainer.addView(button)
        }
    }

    private fun setupSymbolKeyboard(keyboardContainer: GridLayout, inputConnection: InputConnection?) {
        val symbols = listOf(
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            "@", "#", "$", "%", "&", "-", "+", "(", ")",
            "!", "\"", "'", ":", ";", "/", "?", "*"
        )

        for (symbol in symbols) {
            val button = createKeyButton(keyboardContainer.context, symbol, inputConnection)
            keyboardContainer.addView(button)
        }
    }

    private fun createKeyButton(context: android.content.Context, text: String, inputConnection: InputConnection?): Button {
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
                handleKeyPress(inputConnection, keyText)
                if (isShiftEnabled) {
                    isShiftEnabled = false
                    updateKeyLabels(parent as GridLayout)
                }
            }
        }
    }

    private fun updateKeyLabels(keyboardContainer: GridLayout) {
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
    }

    private fun handleKeyPress(inputConnection: InputConnection?, text: String) {
        inputConnection?.commitText(text, 1)

        // Update our tracking of the current text
        currentText.append(text)

        // After each keypress, update suggestions
        if (text != "\n") {
            updateSuggestions()
        } else {
            // If enter was pressed, clear current text tracking
            currentText.clear()
            clearSuggestions()
        }
    }

    private fun updateSuggestions() {
        // Get predictions based on current text
        val predictions = predictor.getPredictions(currentText.toString())

        // Update suggestion buttons
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

    private fun clearSuggestions() {
        suggestionButtons.forEach { button ->
            button.text = ""
            button.visibility = View.INVISIBLE
        }
    }

    // Call this method periodically (maybe when keyboard is hidden)
    // to update the local model with user data
    private fun updateLocalModel() {
        predictor.updateLocalModel()
    }
}