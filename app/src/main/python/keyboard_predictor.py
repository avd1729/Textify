import os
import re
import pickle
import random
import traceback
from collections import defaultdict, Counter
from ngram_model import NGramModel

class KeyboardPredictor:
    """Main class for keyboard prediction functionality"""

    def __init__(self, model_path=None, n=3, sample_texts=None):
        """
        Initialize the keyboard predictor

        Args:
            model_path: Path to load existing model from (or None to create new)
            n: N-gram size for new models
            sample_texts: List of sample texts to pre-train the model if creating new
        """
        print(f"KeyboardPredictor initializing with model_path: {model_path}")

        # Log directory information for debugging
        if model_path:
            model_dir = os.path.dirname(model_path)
            print(f"Model directory: {model_dir}")
            print(f"Directory exists: {os.path.exists(model_dir)}")
            print(f"Directory writable: {os.access(model_dir, os.W_OK) if os.path.exists(model_dir) else 'N/A'}")

        # Create new model or load existing
        if model_path and os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            try:
                self.model = NGramModel.load(model_path)
                print(f"Model successfully loaded from {model_path}")
                print(f"Model stats: vocab size={len(self.model.vocabulary)}, total words={self.model.total_words}")
            except Exception as e:
                print(f"ERROR loading model: {str(e)}")
                print(traceback.format_exc())
                self.model = NGramModel(n=n)
                print(f"Created new model as fallback after load error")
        else:
            print(f"Creating new NGramModel with n={n}")
            self.model = NGramModel(n=n)

            # If sample texts provided and we're creating a new model, pre-train it
            if sample_texts:
                print(f"Pre-training model with {len(sample_texts)} sample texts")
                for i, text in enumerate(sample_texts):
                    print(f"Training on sample text {i+1}/{len(sample_texts)}, length: {len(text)}")
                    self.model.train(text)
                print(f"Pre-training complete. Model stats: vocab size={len(self.model.vocabulary)}, total words={self.model.total_words}")

        self.model_path = model_path or "keyboard_model.pkl"
        self.user_history = []

        # Save the initial model
        print("Saving initial model...")
        self.save_model()

    def add_to_history(self, text):
        """Add user input to history for later training"""
        print(f"Adding to history: '{text}'")
        self.user_history.append(text)

        # Log current history status
        print(f"Current history size: {len(self.user_history)}/100")

        # If history gets too long, train on it and clear
        if len(self.user_history) >= 100:
            print(f"History reached threshold (100 items). Training model...")
            self.train_on_history()

    def train_on_history(self):
        """Train the model on collected user history"""
        if not self.user_history:
            print("No history to train on. Skipping training.")
            return

        # Combine all history entries and train
        all_text = " ".join(self.user_history)
        print(f"Training on combined history text, length: {len(all_text)}")
        self.model.train(all_text)
        print(f"Training complete. Updated model stats: vocab size={len(self.model.vocabulary)}, total words={self.model.total_words}")

        # Save updated model
        print("Saving model after training...")
        self.save_model()

        # Clear history after training
        history_length = len(self.user_history)
        self.user_history = []
        print(f"Cleared {history_length} history items after training")

    def predict(self, context, num_predictions=3):
        """Predict next words based on context"""
        print(f"Predicting next words for context: '{context}'")
        predictions = self.model.predict_next_word(context, num_predictions)
        print(f"Predictions: {predictions}")
        return predictions

    def save_model(self):
        """Save the model to the specified path"""
        try:
            print(f"Attempting to save model to {self.model_path}")

            # Ensure directory exists
            model_dir = os.path.dirname(self.model_path)
            if model_dir and not os.path.exists(model_dir):
                print(f"Creating directory: {model_dir}")
                os.makedirs(model_dir, exist_ok=True)

            # Save the model
            self.model.save(self.model_path)

            # Verify file was created
            if os.path.exists(self.model_path):
                file_size = os.path.getsize(self.model_path)
                print(f"Model successfully saved to {self.model_path} (size: {file_size} bytes)")
            else:
                print(f"WARNING: Model file not found at {self.model_path} after save attempt")
        except Exception as e:
            print(f"ERROR saving model: {str(e)}")
            print(traceback.format_exc())

    def prepare_for_export(self):
        """
        Ensure vocabulary is in a format compatible with the server's aggregation process.
        """
        # If vocabulary is a set, convert it to a dict with counts
        if isinstance(self.model.vocabulary, set):
            self.model.vocabulary = {word: 1 for word in self.model.vocabulary}
        return True

    def export_for_aggregation(self, export_path=None):
        """Export the model in a format suitable for aggregation"""
        if export_path is None:
            export_path = self.model_path.replace('.pkl', '_export.pkl')

        # Prepare the model data for export
        export_data = {
            'n': self.model.n,
            'smoothing': self.model.smoothing,
            'models': {k: dict(v) for k, v in self.model.models.items()},  # Convert defaultdict to dict
            'word_count': self.model.word_count,
            'total_words': self.model.total_words,
            'vocabulary': dict(self.model.vocabulary) if hasattr(self.model.vocabulary, 'items') else {word: 1 for word in self.model.vocabulary}
        }

        with open(export_path, 'wb') as f:
            pickle.dump(export_data, f)

        return export_path

    def load_model(self, model_path):
        """
        Load a model from the given path
        Returns True if successful, False otherwise
        """
        try:
            print(f"Attempting to load model from {model_path}")
            self.model = NGramModel.load(model_path)
            print(f"Model successfully loaded from {model_path}")
            print(f"Model stats: vocab size={len(self.model.vocabulary)}, total words={self.model.total_words}")

            # Save to the original model path to ensure it persists
            self.save_model()
            return True
        except Exception as e:
            print(f"ERROR loading model: {str(e)}")
            print(traceback.format_exc())
            return False