import re

def tokenize_sentence(sentence):
    # This regex pattern is where the magic happens. 
    # It tells the computer exactly what a "token" looks like.
    # We use the OR operator (|) to separate our patterns.
    
    token_pattern = r"""
        (?:https?://\S+)                                        # Matches URLs (e.g., http://example.com)
      | (?:[@#]\w+)                                             # Matches Mentions or Hashtags (e.g., @user, #nlp)
      | (?:\d+(?:-year-old|\syears?\sold))                      # Matches Age values (e.g., 25-year-old, 18 years old)
      | (?:(?:[0-1]?[0-9]|2[0-3]):[0-5][0-9](?:\s?[aApP][mM])?) # Matches Time expressions (e.g., 10:30 AM, 14:00)
      | (?:\d{4}s|\d{1,2}th\scentury)                           # Matches Time periods (e.g., 1990s, 20th century)
      | (?:\d+(?:\.\d+)?%)                                      # Matches Percentages (e.g., 99%, 4.5%)
      | (?:\b\w+\b)                                             # Matches standard words or standalone numbers
      | (?:[^\w\s])                                             # Matches remaining single punctuation marks
    """
    
    # re.findall extracts all non-overlapping matches of the pattern
    # re.VERBOSE allows us to write the regex across multiple lines with comments!
    tokens = re.findall(token_pattern, sentence, re.VERBOSE | re.IGNORECASE)
    return tokens

def main():
    # 1. prompt for the user 
    user_input = input("your text: ")
    
    # 2. Split the input into sentences. 
    # This is a basic split that keeps the punctuation attached for now.
    # A more robust version would handle abbreviations like "Mr." or "e.g."
    raw_sentences = re.split(r'(?<=[.!?]) +', user_input.strip())
    
    tokenized_text = []
    
    # 3. Tokenize each sentence and add it to our final list
    for sent in raw_sentences:
        if sent: # ensure it's not empty
            tokens = tokenize_sentence(sent)
            tokenized_text.append(tokens)
            
    # 4. Output the result in the exact format requested 
    print(f"tokenized text: {tokenized_text}")

if __name__ == "__main__":
    main()