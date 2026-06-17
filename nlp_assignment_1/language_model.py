import sys
import math
import re
from collections import defaultdict, Counter

# ==========================================
# 1. Tokenizer (Integrated for standalone use)
# ==========================================
def tokenize(text):
    """
    A robust tokenizer handling words, punctuation, and edge cases.
    """
    token_pattern = r"""
        (?:https?://\S+)|(?:\b\d+(?:-year-old|\syears?\sold)\b)|
        (?:(?:[0-1]?[0-9]|2[0-3]):[0-5][0-9](?:\s?[aApP][mM])?)|
        (?:\b\d{4}s|\d{1,2}th\scentury\b)|(?:\b\d+(?:\.\d+)?%)|
        (?:[@#]\w+)|(?:\b\w+\b)|(?:[^\w\s])
    """
    sentences = re.split(r'(?<=[.!?]) +', text.strip().lower())
    tokenized_sentences = []
    for sent in sentences:
        if sent:
            tokens = re.findall(token_pattern, sent, re.VERBOSE)
            tokenized_sentences.append(tokens)
    return tokenized_sentences

# ==========================================
# 2. Language Model Core
# ==========================================
class LanguageModel:
    def __init__(self, lm_type, n=3):
        self.lm_type = lm_type.lower()
        self.n = n
        
        # Frequency maps for N-grams and (N-1)-grams (context)
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        
        # For Interpolation (needs unigrams, bigrams, and trigrams)
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        self.total_words = 0
        
        self.vocabulary = set()
        
        # Good-Turing specific variables
        self.freq_of_freqs = Counter()
        self.total_ngrams = 0
        self.gt_r_star = {} # Precomputed adjusted counts
        self.gt_p_zero = 0  # Probability for unseen events

    def train(self, tokenized_corpus):
        """Builds frequency distributions from the training corpus."""
        for sentence in tokenized_corpus:
            self.total_words += len(sentence)
            for word in sentence:
                self.vocabulary.add(word)
                self.unigram_counts[(word,)] += 1

            # Pad sentences for N-gram context
            padded_sentence = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            
            # Count N-grams and Contexts
            for i in range(len(padded_sentence) - self.n + 1):
                ngram = tuple(padded_sentence[i : i + self.n])
                context = tuple(padded_sentence[i : i + self.n - 1])
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.total_ngrams += 1
                
                # Build bigrams and trigrams specifically for Interpolation
                if self.n >= 3:
                    self.trigram_counts[ngram] += 1
                    self.bigram_counts[tuple(padded_sentence[i+1 : i+3])] += 1

        if self.lm_type == 'g':
            self._prepare_good_turing()

    def _prepare_good_turing(self):
        """Calculates the Good-Turing regression line and r* values."""
        # 1. Calculate frequency of frequencies (N_r)
        for count in self.ngram_counts.values():
            self.freq_of_freqs[count] += 1
            
        # 2. Calculate Unseen Probability (N_1 / N)
        n1 = self.freq_of_freqs.get(1, 0)
        
        # Total possible ngrams = V^N. Unseen = V^N - len(seen)
        vocab_size = len(self.vocabulary)
        total_possible_ngrams = vocab_size ** self.n
        unseen_count = total_possible_ngrams - len(self.ngram_counts)
        
        if unseen_count > 0 and self.total_ngrams > 0:
            # (N1 / N) divided by the number of unseen events
            self.gt_p_zero = (n1 / self.total_ngrams) / unseen_count 
        else:
            self.gt_p_zero = 0

        # 3. Simple Linear Regression for log(N_r) = a + b * log(r)
        # We only fit the regression for r values that actually exist
        r_values = sorted(list(self.freq_of_freqs.keys()))
        log_r = [math.log(r) for r in r_values]
        log_nr = [math.log(self.freq_of_freqs[r]) for r in r_values]
        
        n_items = len(r_values)
        if n_items > 1:
            sum_x = sum(log_r)
            sum_y = sum(log_nr)
            sum_xy = sum(x*y for x, y in zip(log_r, log_nr))
            sum_xx = sum(x*x for x in log_r)
            
            # Slope (b) and Intercept (a)
            denominator = (n_items * sum_xx - sum_x**2)
            b = (n_items * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0
            a = (sum_y - b * sum_x) / n_items
        else:
            a, b = 0, 0

        # Helper to get Smoothed S(r)
        def S(r):
            if r <= 5: # Use exact N_r for small values
                return self.freq_of_freqs.get(r, 0)
            return math.exp(a + b * math.log(r)) # Use regression for large values

        # 4. Precompute r* for all observed r
        for r in r_values:
            s_r = S(r)
            s_r_plus_1 = S(r + 1)
            
            if s_r == 0: # Failsafe
                self.gt_r_star[r] = r 
            else:
                r_star = (r + 1) * s_r_plus_1 / s_r
                self.gt_r_star[r] = r_star

    # ==========================================
    # 3. Scoring Functions
    # ==========================================
    def get_probability(self, ngram, context):
        if self.lm_type == 'l':
            return self._laplace(ngram, context)
        elif self.lm_type == 'g':
            return self._good_turing(ngram)
        elif self.lm_type == 'i':
            return self._interpolation(ngram)
        else:
            return 0.0

    def _laplace(self, ngram, context):
        """Add-One Smoothing: (Count(ngram) + 1) / (Count(context) + V)"""
        count_ngram = self.ngram_counts.get(ngram, 0)
        count_context = self.context_counts.get(context, 0)
        vocab_size = len(self.vocabulary)
        
        return (count_ngram + 1) / (count_context + vocab_size)

    def _good_turing(self, ngram):
        """P_GT = r* / N for seen, N_1 / N for unseen."""
        r = self.ngram_counts.get(ngram, 0)
        
        if r == 0:
            return self.gt_p_zero
        else:
            r_star = self.gt_r_star.get(r, r)
            return r_star / self.total_ngrams

    def _interpolation(self, ngram):
        """Linear combination of Trigram, Bigram, and Unigram probabilities."""
        # Weights (must sum to 1.0)
        lambda1, lambda2, lambda3 = 0.1, 0.3, 0.6 
        
        w3 = ngram[-1]
        w2 = ngram[-2] if len(ngram) > 1 else None
        w1 = ngram[-3] if len(ngram) > 2 else None

        # FIX: Provide a baseline fallback for completely unseen words (OOV)
        if (w3,) in self.unigram_counts:
            p_unigram = self.unigram_counts[(w3,)] / self.total_words
        else:
            # Fallback: 1 / Vocabulary Size
            vocab_size = len(self.vocabulary)
            p_unigram = 1.0 / vocab_size if vocab_size > 0 else 0
        
        # P(w3 | w2)
        bigram_context_count = self.unigram_counts.get((w2,), 0)
        p_bigram = self.bigram_counts.get((w2, w3), 0) / bigram_context_count if bigram_context_count > 0 else 0
        
        # P(w3 | w1, w2)
        trigram_context_count = self.bigram_counts.get((w1, w2), 0)
        p_trigram = self.trigram_counts.get(ngram, 0) / trigram_context_count if trigram_context_count > 0 else 0

        return (lambda1 * p_unigram) + (lambda2 * p_bigram) + (lambda3 * p_trigram)
    
    def score_sentence(self, sentence_tokens):
        """Calculates the overall probability of a sentence."""
        padded_sentence = ['<s>'] * (self.n - 1) + sentence_tokens + ['</s>']
        total_prob = 1.0
        
        for i in range(len(padded_sentence) - self.n + 1):
            ngram = tuple(padded_sentence[i : i + self.n])
            context = tuple(padded_sentence[i : i + self.n - 1])
            
            prob = self.get_probability(ngram, context)
            total_prob *= prob
            
        return total_prob
    
    def calculate_perplexity(self, sentence_tokens):
        """Calculates the perplexity of a sentence using log probabilities to prevent underflow."""
        if not sentence_tokens:
            return float('inf')

        padded_sentence = ['<s>'] * (self.n - 1) + sentence_tokens + ['</s>']
        log_prob_sum = 0.0
        
        # N is the number of words in the sentence (plus the stop token)
        N = len(sentence_tokens) + 1 
        
        for i in range(len(padded_sentence) - self.n + 1):
            ngram = tuple(padded_sentence[i : i + self.n])
            context = tuple(padded_sentence[i : i + self.n - 1])
            
            prob = self.get_probability(ngram, context)
            
            # Failsafe: If probability is somehow 0, perplexity goes to infinity
            if prob <= 0:
                return float('inf')
                
            # Log addition instead of standard multiplication
            log_prob_sum += math.log(prob)
            
        # The math formula for perplexity using logs: e^(-(1/N) * sum(log(P)))
        return math.exp(-log_prob_sum / N)

# ==========================================
# 4. Main Execution and Interactive Loop
# ==========================================
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 language_model.py <lm_type> <corpus_path>")
        print("lm_type: l (Laplace), g (Good-Turing), i (Interpolation)")
        sys.exit(1)

    lm_type = sys.argv[1].lower()
    corpus_path = sys.argv[2]

    if lm_type not in ['l', 'g', 'i']:
        print("Error: LM type must be 'l', 'g', or 'i'.")
        sys.exit(1)

    print(f"Loading corpus from {corpus_path}...")
    try:
        with open(corpus_path, 'r', encoding='utf-8') as file:
            corpus_text = file.read()
    except FileNotFoundError:
        print(f"Error: Could not find file {corpus_path}")
        sys.exit(1)

    print("Tokenizing corpus...")
    tokenized_corpus = tokenize(corpus_text)

    print("Training Language Model (N=3)...")
    lm = LanguageModel(lm_type=lm_type, n=3)
    lm.train(tokenized_corpus)
    print("Training complete.\n")

    # Interactive Loop
    while True:
        try:
            user_input = input("input sentence (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
                
            # Tokenize the single input sentence
            tokenized_input = tokenize(user_input)
            
            if not tokenized_input:
                continue
                
            # Score the first sentence found in the input
            tokens = tokenized_input[0]
            score = lm.score_sentence(tokens)
            
            print(f"score: {score:.8e}\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()