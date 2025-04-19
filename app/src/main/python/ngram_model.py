import re
import pickle
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n=3, smoothing=0.1):
        """
        Initialize the N-gram model

        Args:
            n: The maximum n-gram size
            smoothing: Smoothing parameter for unseen n-grams
        """
        self.n = n
        self.smoothing = smoothing
        self.models = {}  # Contains different n-gram models (unigram, bigram, trigram)
        self.word_count = Counter()  # Total word count for unigram probabilities
        self.total_words = 0
        self.vocabulary = set()

    def preprocess_text(self, text):
        """Clean and tokenize the text"""
        # Convert to lowercase and replace certain punctuation with spaces
        text = text.lower()
        text = re.sub(r'[^\w\s\']', ' ', text)

        # Split into words and remove empty strings
        words = [word for word in text.split() if word]
        return words

    def build_ngrams(self, words, n):
        """Generate n-grams from a list of words"""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def train(self, text):
        """Train the model on the given text"""
        words = self.preprocess_text(text)
        if not words:
            return

        # Update vocabulary and word count
        self.vocabulary.update(words)
        self.word_count.update(words)
        self.total_words += len(words)

        # Train models for different n-gram sizes (1 to n)
        for i in range(1, self.n + 1):
            if i == 1:
                # Unigram model is just word frequencies
                continue

            ngrams = self.build_ngrams(words, i)

            # Create or update the model for this n-gram size
            if i not in self.models:
                self.models[i] = defaultdict(Counter)

            # Count n-gram occurrences
            for ngram in ngrams:
                context = ngram[:-1]  # All but the last word
                word = ngram[-1]      # The last word
                self.models[i][context][word] += 1

    def predict_next_word(self, context, num_predictions=3):
        """
        Predict the next word given a context

        Args:
            context: A string of words
            num_predictions: Number of predictions to return

        Returns:
            List of (word, probability) tuples
        """
        words = self.preprocess_text(context)

        # If no valid words in context, return most common words
        if not words:
            most_common = self.word_count.most_common(num_predictions)
            total = self.total_words or 1
            return [(word, count / total) for word, count in most_common]

        candidates = Counter()

        # Try using the largest n-gram model that fits our context
        for n in range(min(self.n, len(words) + 1), 1, -1):
            context_tuple = tuple(words[-(n-1):]) if n > 1 else tuple()

            # If we have this context in our model
            if n in self.models and context_tuple in self.models[n]:
                counter = self.models[n][context_tuple]
                total = sum(counter.values())

                # Calculate probabilities with smoothing
                vocab_size = len(self.vocabulary)
                for word, count in counter.items():
                    prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
                    candidates[word] += prob

                # If we found enough predictions, return them
                if len(candidates) >= num_predictions:
                    break

        # If we don't have enough predictions, fall back to unigrams
        if len(candidates) < num_predictions:
            remaining = num_predictions - len(candidates)
            total = self.total_words or 1
            vocab_size = len(self.vocabulary) or 1

            # Add most common words that aren't already candidates
            for word, count in self.word_count.most_common():
                if word not in candidates and remaining > 0:
                    prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)
                    candidates[word] = prob
                    remaining -= 1
                if remaining <= 0:
                    break

        # Return top predictions
        return candidates.most_common(num_predictions)

    def save(self, filepath):
        """Save the model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'smoothing': self.smoothing,
                'models': dict(self.models),  # Convert defaultdict to dict for serialization
                'word_count': self.word_count,
                'total_words': self.total_words,
                'vocabulary': self.vocabulary
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load the model from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(n=data['n'], smoothing=data['smoothing'])

        # Convert dict back to defaultdict
        model.models = {k: defaultdict(Counter, v) for k, v in data['models'].items()}
        model.word_count = data['word_count']
        model.total_words = data['total_words']
        model.vocabulary = data['vocabulary']

        return model

    def load_model(self, model_path):
        """
        Load a model from the given path
        Returns True if successful, False otherwise
        """
        try:
            # Use the class method you provided to load the model
            new_model = self.__class__.load(model_path)

            # Copy the loaded model's attributes to this instance
            self.models = new_model.models
            self.word_count = new_model.word_count
            self.total_words = new_model.total_words
            self.vocabulary = new_model.vocabulary
            self.n = new_model.n
            self.smoothing = new_model.smoothing

            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False