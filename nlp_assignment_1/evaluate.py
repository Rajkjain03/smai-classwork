import os
import random
import math
from language_model import tokenize, LanguageModel

# --- Configuration ---
# Ensure your corpus text files are named exactly like this, or update the paths
CORPORA = {
    "pride": {"path": "./pride_and_prejudice.txt", "lm_base": 0}, # LMs 1, 2, 3
    "ulysses": {"path": "./ulysses.txt", "lm_base": 3}            # LMs 4, 5, 6
}

LM_TYPES = {
    'l': 1, # Laplace adds 1 to base
    'g': 2, # Good-Turing adds 2 to base
    'i': 3  # Interpolation adds 3 to base
}

N_VALUES = [1, 3, 5]
TEST_SET_SIZE = 1000
ROLL_NUMBER = "2025201036"

def load_and_split_corpus(filepath):
    """Loads the corpus, tokenizes it, and splits it into Train and Test sets."""
    print(f"Loading and tokenizing {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}. Please check the filename.")
        return None, None

    sentences = tokenize(text)
    
    # Remove any empty sentences that might have slipped through
    sentences = [s for s in sentences if len(s) > 0]
    
    # We use a fixed seed so the "random" split is the same every time you run it
    random.seed(42)
    random.shuffle(sentences)
    
    test_set = sentences[:TEST_SET_SIZE]
    train_set = sentences[TEST_SET_SIZE:]
    
    return train_set, test_set

def write_perplexity_file(filename, dataset, model):
    """Calculates perplexity for a dataset and writes it to a file."""
    perplexities = []
    lines_to_write = []
    
    for sentence in dataset:
        perp = model.calculate_perplexity(sentence)
        perplexities.append(perp)
        
        # Reconstruct the sentence string for the file output
        sentence_str = " ".join(sentence)
        lines_to_write.append(f"{sentence_str}\t{perp:.4f}\n")
    
    # Calculate Average Perplexity (ignoring infinity values from absolute unseen OOV)
    valid_perps = [p for p in perplexities if p != float('inf')]
    if valid_perps:
        avg_perplexity = sum(valid_perps) / len(valid_perps)
    else:
        avg_perplexity = float('inf')
        
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{avg_perplexity:.4f}\n")
        f.writelines(lines_to_write)

def main():
    print("Starting automated evaluation. This may take several minutes...\n")
    
    for corpus_name, data in CORPORA.items():
        train_set, test_set = load_and_split_corpus(data["path"])
        
        if not train_set:
            continue
            
        print(f"[{corpus_name.upper()}] Train size: {len(train_set)}, Test size: {len(test_set)}")
        
        for n in N_VALUES:
            for lm_key, lm_offset in LM_TYPES.items():
                lm_number = data["lm_base"] + lm_offset
                
                print(f"  Training LM{lm_number} (Type: {lm_key}, N: {n}) on {corpus_name}...")
                model = LanguageModel(lm_type=lm_key, n=n)
                model.train(train_set)
                
                # --- Generate Training Perplexity File ---
                train_filename = f"{ROLL_NUMBER}_LM{lm_number}_{n}_train-perplexity.txt"
                write_perplexity_file(train_filename, train_set, model)
                
                # --- Generate Testing Perplexity File ---
                test_filename = f"{ROLL_NUMBER}_LM{lm_number}_{n}_test-perplexity.txt"
                write_perplexity_file(test_filename, test_set, model)
                
        print(f"Finished processing {corpus_name}.\n")
        
    print("All 36 perplexity files have been successfully generated!")

if __name__ == "__main__":
    main()