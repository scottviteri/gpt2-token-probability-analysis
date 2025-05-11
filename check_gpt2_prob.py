import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wikipedia
import random

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Task 1: Set up environment
print("Loading GPT-2 small model and tokenizer...")
model_name = "gpt2"  # This is the small version
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

# GPT-2 maximum context length
MAX_LENGTH = 1024

# Task 2: Fetch random Wikipedia article
print("Fetching random Wikipedia article...")
random_title = wikipedia.random(1)
print(f"Selected article: {random_title}")
try:
    wiki_content = wikipedia.page(random_title).content
    # Take a shorter excerpt to ensure we stay under token limit
    article_start = ' '.join(wiki_content.split()[:300])
    print(f"Article length: {len(article_start)} characters")
except:
    # Fallback in case of disambiguation or other issues
    print("Had trouble with that article, trying another one...")
    random_titles = wikipedia.random(3)
    for title in random_titles:
        try:
            wiki_content = wikipedia.page(title).content
            article_start = ' '.join(wiki_content.split()[:300])
            print(f"Selected article: {title}")
            print(f"Article length: {len(article_start)} characters")
            break
        except:
            continue

# Task 3 & 4: Calculate average log probability for the tokens
print("Calculating token log probabilities...")

# Tokenize the text
inputs = tokenizer(article_start, return_tensors="pt").to(device)
input_ids = inputs.input_ids

# Ensure we don't exceed the maximum context length
if input_ids.size(1) > MAX_LENGTH:
    print(f"Truncating input from {input_ids.size(1)} to {MAX_LENGTH} tokens")
    input_ids = input_ids[:, :MAX_LENGTH]

# Get the output logits
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)

    # Get log probabilities
    log_probs = outputs.loss.item()  # Cross-entropy loss

    # Calculate more detailed token-level probabilities
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Calculate log probabilities for each token
    log_probs_per_token = []
    for i in range(shift_logits.size(1)):
        token_logits = shift_logits[0, i]
        token_label = shift_labels[0, i].item()
        token_log_prob = torch.log_softmax(token_logits, dim=0)[token_label].item()
        log_probs_per_token.append(token_log_prob)

    # Convert to numpy for easier analysis
    log_probs_per_token = np.array(log_probs_per_token)

    # Calculate average log probability per token
    avg_log_prob = np.mean(log_probs_per_token)
    median_log_prob = np.median(log_probs_per_token)
    min_log_prob = np.min(log_probs_per_token)
    max_log_prob = np.max(log_probs_per_token)

# Task 5: Display results
print("\nRESULTS:")
print(f"GPT-2 small model average log probability: {avg_log_prob:.4f}")
print(f"Median log probability: {median_log_prob:.4f}")
print(f"Min log probability: {min_log_prob:.4f}")
print(f"Max log probability: {max_log_prob:.4f}")
print(f"Model cross-entropy loss: {log_probs:.4f}")
print(f"Perplexity: {np.exp(log_probs):.4f}")

# Calculate distribution statistics
def get_percentile(arr, p):
    return np.percentile(arr, p)

percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print("\nDISTRIBUTION OF TOKEN LOG PROBABILITIES:")
print("Percentile | Log Probability")
print("-----------+----------------")
for p in percentiles:
    value = get_percentile(log_probs_per_token, p)
    print(f"{p:9.0f}% | {value:9.4f}")

# Count tokens with extreme log probabilities
very_low_prob = sum(log_probs_per_token < -10)
low_prob = sum((log_probs_per_token >= -10) & (log_probs_per_token < -5))
medium_prob = sum((log_probs_per_token >= -5) & (log_probs_per_token < -2))
high_prob = sum(log_probs_per_token >= -2)

total_tokens = len(log_probs_per_token)
print("\nLOG PROBABILITY RANGES:")
print(f"Very Low  (< -10): {very_low_prob:4d} tokens ({very_low_prob/total_tokens*100:.1f}%)")
print(f"Low   (-10 to -5): {low_prob:4d} tokens ({low_prob/total_tokens*100:.1f}%)")
print(f"Medium (-5 to -2): {medium_prob:4d} tokens ({medium_prob/total_tokens*100:.1f}%)")
print(f"High      (> -2): {high_prob:4d} tokens ({high_prob/total_tokens*100:.1f}%)")

# Print a small sample of text with its probabilities
print("\nSAMPLE TEXT WITH TOKEN PROBABILITIES:")
sample_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:20])
for token, log_prob in zip(sample_tokens[1:], log_probs_per_token[:19]):
    print(f"Token: '{token}', Log Prob: {log_prob:.4f}")