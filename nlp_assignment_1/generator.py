import sys
# Import our previously built tokenizer and LanguageModel class
from language_model import tokenize, LanguageModel

def generate_next_words(lm, input_sentence, k):
    """
    Given an input string, predicts the top k next possible words.
    """
    # 1. Tokenize the input to get our context
    tokenized_input = tokenize(input_sentence)
    
    if not tokenized_input:
        return []
        
    # We only care about the very last sentence typed for next word prediction
    tokens = tokenized_input[-1]
    
    # 2. Pad the input just like we did in training to handle short prompts
    n = lm.n
    padded_tokens = ['<s>'] * (n - 1) + tokens
    
    # Extract the context (the last N-1 words)
    context = tuple(padded_tokens[-(n-1):])
    
    candidates = []
    
    # 3. Test every word in our vocabulary
    for word in lm.vocabulary:
        # Construct the hypothetical N-gram
        ngram = context + (word,)
        
        # Ask our model for the probability
        prob = lm.get_probability(ngram, context)
        
        # Only keep words with a probability > 0 to save sorting time
        if prob > 0:
            candidates.append((word, prob))
            
    # 4. Sort the candidates by probability (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 5. Return the top k
    return candidates[:k]

def main():
    # Enforce the command-line arguments specified in the assignment
    if len(sys.argv) != 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        print("lm_type: l (Laplace), g (Good-Turing), i (Interpolation)")
        sys.exit(1)

    lm_type = sys.argv[1].lower()
    corpus_path = sys.argv[2]
    
    try:
        k = int(sys.argv[3])
    except ValueError:
        print("Error: k must be an integer.")
        sys.exit(1)

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

    print(f"Training Language Model (N=3)...")
    lm = LanguageModel(lm_type=lm_type, n=3)
    lm.train(tokenized_corpus)
    print("Training complete.\n")

    # Interactive Generation Loop
    while True:
        try:
            user_input = input("input sentence: ")
            if user_input.lower() == 'exit':
                break
                
            print("output:")
            top_k = generate_next_words(lm, user_input, k)
            
            for word, prob in top_k:
                # Format exactly as requested in the assignment (word <space> probability)
                print(f"{word} {prob:.8e}")
            print() # Blank line for readability
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()