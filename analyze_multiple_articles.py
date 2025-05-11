import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wikipedia
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm

# Set up plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Number of articles to analyze - using 20 for faster runtime
# Change to 100 for full analysis
NUM_ARTICLES = 20

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

# Initialize storage for results
article_stats = []

# Function to analyze a single article
def analyze_article(title):
    try:
        # Fetch article content
        wiki_content = wikipedia.page(title).content
        # Take a portion to keep processing manageable
        article_text = ' '.join(wiki_content.split()[:300])
        
        # Tokenize the text
        inputs = tokenizer(article_text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        # Ensure we don't exceed the maximum context length
        if input_ids.size(1) > MAX_LENGTH:
            input_ids = input_ids[:, :MAX_LENGTH]
            
        # Get the output logits and probabilities
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            
            # Get cross-entropy loss
            loss = outputs.loss.item()
            
            # Calculate token-level probabilities
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
                
            # Convert to numpy array
            log_probs_per_token = np.array(log_probs_per_token)
            
            # Calculate statistics
            stats = {
                'title': title,
                'num_tokens': len(log_probs_per_token),
                'mean_log_prob': np.mean(log_probs_per_token),
                'median_log_prob': np.median(log_probs_per_token),
                'std_log_prob': np.std(log_probs_per_token),
                'min_log_prob': np.min(log_probs_per_token),
                'max_log_prob': np.max(log_probs_per_token),
                'perplexity': np.exp(-np.mean(log_probs_per_token)),
                'loss': loss
            }
            
            return stats, log_probs_per_token
            
    except Exception as e:
        print(f"Error processing article '{title}': {str(e)}")
        return None, None

# Process multiple articles
print(f"Analyzing {NUM_ARTICLES} random Wikipedia articles...")
pbar = tqdm(total=NUM_ARTICLES)

all_log_probs = []
successful_articles = 0

while successful_articles < NUM_ARTICLES:
    # Get a random article title
    try:
        title = wikipedia.random(1)
        stats, log_probs = analyze_article(title)
        
        if stats is not None:
            article_stats.append(stats)
            all_log_probs.extend(log_probs)
            successful_articles += 1
            pbar.update(1)
            
            # Print progress update every 10 articles
            if successful_articles % 10 == 0:
                print(f"Processed {successful_articles}/{NUM_ARTICLES} articles")
                
        # Small delay to avoid hammering the Wikipedia API
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Error with article: {str(e)}")
        time.sleep(1)  # Longer delay if there's an error

pbar.close()

# Convert to pandas DataFrame for analysis
df = pd.DataFrame(article_stats)

# Save raw data
df.to_csv('/home/scottviteri/Projects/gpt2check/gpt2_article_stats.csv', index=False)
print(f"Saved statistics for {len(df)} articles to gpt2_article_stats.csv")

# Create visualization plots
plt.figure(figsize=(16, 14))

# Plot 1: Distribution of mean log probabilities across articles
plt.subplot(2, 2, 1)
sns.histplot(df['mean_log_prob'], kde=True, color='blue', alpha=0.7)
plt.axvline(df['mean_log_prob'].mean(), color='red', linestyle='dashed', 
           linewidth=2, label=f'Mean of means: {df["mean_log_prob"].mean():.4f}')
plt.axvline(df['mean_log_prob'].median(), color='green', linestyle='dashed',
           linewidth=2, label=f'Median of means: {df["mean_log_prob"].median():.4f}')
plt.xlabel('Mean Log Probability')
plt.ylabel('Number of Articles')
plt.title('Distribution of Mean Log Probabilities Across Articles')
plt.legend()

# Plot 2: Mean vs Median log probability for each article
plt.subplot(2, 2, 2)
plt.scatter(df['mean_log_prob'], df['median_log_prob'], alpha=0.7)
# Add diagonal line y=x
min_val = min(df['mean_log_prob'].min(), df['median_log_prob'].min())
max_val = max(df['mean_log_prob'].max(), df['median_log_prob'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
plt.xlabel('Mean Log Probability')
plt.ylabel('Median Log Probability')
plt.title('Mean vs Median Log Probability by Article')
plt.grid(True)

# Add text showing average difference between median and mean
mean_diff = (df['median_log_prob'] - df['mean_log_prob']).mean()
plt.annotate(f'Avg. Difference (Median-Mean): {mean_diff:.4f}',
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Plot 3: Distribution of all token log probabilities
plt.subplot(2, 2, 3)
sns.histplot(all_log_probs, bins=50, kde=True, color='purple', alpha=0.7)
all_log_probs_mean = np.mean(all_log_probs)
all_log_probs_median = np.median(all_log_probs)
plt.axvline(all_log_probs_mean, color='red', linestyle='dashed',
           linewidth=2, label=f'Mean: {all_log_probs_mean:.4f}')
plt.axvline(all_log_probs_median, color='green', linestyle='dashed',
           linewidth=2, label=f'Median: {all_log_probs_median:.4f}')
plt.xlabel('Log Probability')
plt.ylabel('Count')
plt.title('Distribution of All Token Log Probabilities')
plt.legend()

# Plot 4: Article perplexity histogram
plt.subplot(2, 2, 4)
sns.histplot(df['perplexity'], bins=20, kde=True, color='orange', alpha=0.7)
plt.axvline(df['perplexity'].mean(), color='red', linestyle='dashed',
           linewidth=2, label=f'Mean: {df["perplexity"].mean():.2f}')
plt.axvline(df['perplexity'].median(), color='green', linestyle='dashed',
           linewidth=2, label=f'Median: {df["perplexity"].median():.2f}')
plt.xlabel('Perplexity')
plt.ylabel('Number of Articles')
plt.title('Distribution of Perplexity Across Articles')
plt.legend()

# Add summary statistics as text
plt.figtext(0.5, 0.01, 
            f"Overall Statistics (across all tokens, n={len(all_log_probs)}):\n"
            f"Mean Log Probability: {all_log_probs_mean:.4f}\n"
            f"Median Log Probability: {all_log_probs_median:.4f}\n"
            f"Overall Perplexity: {np.exp(-all_log_probs_mean):.2f}\n"
            f"Mean-Median Gap: {all_log_probs_mean - all_log_probs_median:.4f}",
            ha="center", fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

plt.suptitle('GPT-2 Small Log Probability Analysis Across Multiple Wikipedia Articles', 
            fontsize=16, y=0.99)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('/home/scottviteri/Projects/gpt2check/gpt2_multi_article_analysis.png', dpi=300)
print("Saved multi-article analysis plot to gpt2_multi_article_analysis.png")

# Print summary statistics
print("\nSUMMARY STATISTICS ACROSS ALL ARTICLES:")
print(f"Number of articles successfully analyzed: {len(df)}")
print(f"Total tokens analyzed: {len(all_log_probs)}")
print(f"Overall mean log probability: {all_log_probs_mean:.4f}")
print(f"Overall median log probability: {all_log_probs_median:.4f}")
print(f"Overall perplexity: {np.exp(-all_log_probs_mean):.2f}")
print(f"\nAverage article-level statistics:")
print(f"Mean log probability: {df['mean_log_prob'].mean():.4f} (std: {df['mean_log_prob'].std():.4f})")
print(f"Median log probability: {df['median_log_prob'].mean():.4f} (std: {df['median_log_prob'].std():.4f})")
print(f"Average perplexity: {df['perplexity'].mean():.2f} (std: {df['perplexity'].std():.2f})")

# If we have a lot of articles, also show the most and least predictable ones
if len(df) >= 10:
    print("\nMost predictable articles (lowest perplexity):")
    most_predictable = df.sort_values('perplexity').head(5)
    for _, row in most_predictable.iterrows():
        print(f"- {row['title']}: Perplexity {row['perplexity']:.2f}, Mean Log Prob {row['mean_log_prob']:.4f}")
        
    print("\nLeast predictable articles (highest perplexity):")
    least_predictable = df.sort_values('perplexity', ascending=False).head(5)
    for _, row in least_predictable.iterrows():
        print(f"- {row['title']}: Perplexity {row['perplexity']:.2f}, Mean Log Prob {row['mean_log_prob']:.4f}")