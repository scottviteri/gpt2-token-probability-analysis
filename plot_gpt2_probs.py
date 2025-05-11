import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wikipedia
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading GPT-2 small model and tokenizer...")
model_name = "gpt2"  # This is the small version
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

# GPT-2 maximum context length
MAX_LENGTH = 1024

# Fetch random Wikipedia article
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

# Calculate token log probabilities
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
    tokens = []
    for i in range(shift_logits.size(1)):
        token_logits = shift_logits[0, i]
        token_label = shift_labels[0, i].item()
        token_log_prob = torch.log_softmax(token_logits, dim=0)[token_label].item()
        log_probs_per_token.append(token_log_prob)
        tokens.append(tokenizer.convert_ids_to_tokens(token_label))
    
    # Convert to numpy for easier analysis
    log_probs_per_token = np.array(log_probs_per_token)
    
    # Calculate statistics
    avg_log_prob = np.mean(log_probs_per_token)
    median_log_prob = np.median(log_probs_per_token)
    min_log_prob = np.min(log_probs_per_token)
    max_log_prob = np.max(log_probs_per_token)

# Create a figure with multiple plots
plt.figure(figsize=(15, 12))

# Plot 1: Histogram of log probabilities
plt.subplot(2, 2, 1)
plt.hist(log_probs_per_token, bins=25, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(avg_log_prob, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_log_prob:.2f}')
plt.axvline(median_log_prob, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_log_prob:.2f}')
plt.xlabel('Log Probability')
plt.ylabel('Number of Tokens')
plt.title('Distribution of Token Log Probabilities')
plt.legend()

# Plot 2: Density plot (smoothed histogram)
plt.subplot(2, 2, 2)
sns.kdeplot(log_probs_per_token, fill=True)
plt.axvline(avg_log_prob, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_log_prob:.2f}')
plt.axvline(median_log_prob, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_log_prob:.2f}')
plt.xlabel('Log Probability')
plt.ylabel('Density')
plt.title('Density Plot of Log Probabilities')
plt.legend()

# Plot 3: Log probabilities over token position (first 100 tokens)
plt.subplot(2, 2, 3)
token_positions = np.arange(min(100, len(log_probs_per_token)))
probs_to_plot = log_probs_per_token[:min(100, len(log_probs_per_token))]
plt.plot(token_positions, probs_to_plot, marker='o', linestyle='-', alpha=0.7)
plt.axhline(avg_log_prob, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_log_prob:.2f}')
plt.axhline(median_log_prob, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_log_prob:.2f}')
plt.xlabel('Token Position')
plt.ylabel('Log Probability')
plt.title('Log Probabilities by Token Position (first 100 tokens)')
plt.legend()

# Plot 4: Boxplot
plt.subplot(2, 2, 4)
plt.boxplot(log_probs_per_token, vert=False, widths=0.7, patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
plt.axvline(avg_log_prob, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_log_prob:.2f}')
plt.xlabel('Log Probability')
plt.title('Boxplot of Token Log Probabilities')
plt.grid(axis='x')

# Find the 10 lowest probability tokens
lowest_prob_indices = np.argsort(log_probs_per_token)[:10]
lowest_prob_tokens = [tokens[i] for i in lowest_prob_indices]
lowest_probs = log_probs_per_token[lowest_prob_indices]

# Add a text box with the 10 lowest probability tokens
lowest_tokens_text = "10 Lowest Probability Tokens:\n"
for token, prob in zip(lowest_prob_tokens, lowest_probs):
    lowest_tokens_text += f"'{token}': {prob:.4f}\n"
plt.figtext(0.02, 0.02, lowest_tokens_text, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8))

# Add a title for the entire figure
plt.suptitle(f'GPT-2 Small Token Log Probability Analysis\nArticle: {random_title}', 
             fontsize=16, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/scottviteri/Projects/gpt2check/gpt2_prob_plots.png', dpi=300)
print("Plots saved to /home/scottviteri/Projects/gpt2check/gpt2_prob_plots.png")

# Display summary statistics
print("\nSUMMARY STATISTICS:")
print(f"Mean log probability: {avg_log_prob:.4f}")
print(f"Median log probability: {median_log_prob:.4f}")
print(f"Standard deviation: {np.std(log_probs_per_token):.4f}")
print(f"Min log probability: {min_log_prob:.4f}")
print(f"Max log probability: {max_log_prob:.4f}")
print(f"Perplexity: {np.exp(-avg_log_prob):.4f}")