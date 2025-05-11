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

# Configuration
NUM_ARTICLES = 5          # Number of articles to analyze (reduced for speed)
NUM_SAMPLES_PER_ARTICLE = 2  # Number of starting positions per article
SEQUENCE_LENGTH = 256     # Length of context window to analyze (reduced)
POSITION_BINS = 16        # Number of bins for averaging positions

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading GPT-2 small model and tokenizer...")
model_name = "gpt2"  # This is the small version
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

# Initialize storage for results
all_position_probs = {}
for i in range(SEQUENCE_LENGTH):
    all_position_probs[i] = []

article_stats = []

def analyze_article_segment(content, start_idx=0):
    """Analyze a segment of an article starting at start_idx"""
    # Extract article segment
    words = content.split()
    
    if start_idx >= len(words) - 10:  # Ensure we have at least 10 words
        start_idx = max(0, len(words) - 100)
    
    # Take enough words for the sequence length
    segment_words = words[start_idx:start_idx + 1000]  # Take extra to ensure enough tokens
    segment = ' '.join(segment_words)
    
    # Tokenize the text
    inputs = tokenizer(segment, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Limit to sequence length
    if input_ids.size(1) > SEQUENCE_LENGTH:
        input_ids = input_ids[:, :SEQUENCE_LENGTH]
        
    # Get the output logits and probabilities
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        
        # Get cross-entropy loss
        loss = outputs.loss.item()
        
        # Calculate token-level probabilities
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Calculate log probabilities for each token by position
        log_probs_per_token = []
        
        for i in range(shift_logits.size(1)):
            token_logits = shift_logits[0, i]
            token_label = shift_labels[0, i].item()
            token_log_prob = torch.log_softmax(token_logits, dim=0)[token_label].item()
            log_probs_per_token.append(token_log_prob)
            
            # Store by position
            if i < SEQUENCE_LENGTH:
                all_position_probs[i].append(token_log_prob)
            
        # Convert to numpy array
        log_probs_per_token = np.array(log_probs_per_token)
        
        # Calculate statistics
        stats = {
            'start_idx': start_idx,
            'num_tokens': len(log_probs_per_token),
            'mean_log_prob': np.mean(log_probs_per_token),
            'median_log_prob': np.median(log_probs_per_token),
            'loss': loss
        }
        
        return stats, log_probs_per_token
    
    return None, None

# Process multiple articles
print(f"Analyzing {NUM_ARTICLES} articles with {NUM_SAMPLES_PER_ARTICLE} samples each...")
pbar = tqdm(total=NUM_ARTICLES * NUM_SAMPLES_PER_ARTICLE)

successful_samples = 0
all_articles = []

# Predefined list of Wikipedia articles that are likely to be long enough
preselected_articles = [
    "United States", "World War II", "Albert Einstein",
    "New York City", "Barack Obama", "Cat", "Computer science",
    "COVID-19 pandemic", "Climate change", "Artificial intelligence"
]

for title in preselected_articles:
    if len(all_articles) >= NUM_ARTICLES:
        break

    try:
        article_content = wikipedia.page(title).content

        # Only use articles with sufficient content
        if len(article_content.split()) > SEQUENCE_LENGTH * 2:
            all_articles.append((title, article_content))
            print(f"Added article: {title} ({len(article_content.split())} words)")
    except Exception as e:
        print(f"Error with article '{title}': {str(e)}")

    # Small delay to avoid hammering the Wikipedia API
    time.sleep(0.5)

# Now process each article with multiple starting positions
for title, content in all_articles:
    words = content.split()
    max_start = max(0, len(words) - SEQUENCE_LENGTH * 2)  # Ensure enough content
    
    # Generate different starting positions
    if max_start > 0:
        start_positions = sorted(random.sample(range(max_start), 
                                   min(NUM_SAMPLES_PER_ARTICLE, max_start)))
    else:
        start_positions = [0] * NUM_SAMPLES_PER_ARTICLE
    
    # Process each starting position
    for start_idx in start_positions:
        try:
            stats, _ = analyze_article_segment(content, start_idx)
            
            if stats is not None:
                stats['title'] = title
                article_stats.append(stats)
                successful_samples += 1
                pbar.update(1)
        except Exception as e:
            print(f"Error processing segment of '{title}': {str(e)}")

pbar.close()

# Convert position data to DataFrame for easier analysis
position_df = pd.DataFrame({
    'position': list(all_position_probs.keys()),
    'mean_log_prob': [np.mean(probs) if probs else np.nan for probs in all_position_probs.values()],
    'median_log_prob': [np.median(probs) if probs else np.nan for probs in all_position_probs.values()],
    'std_log_prob': [np.std(probs) if len(probs) > 1 else np.nan for probs in all_position_probs.values()],
    'count': [len(probs) for probs in all_position_probs.values()]
})

# Calculate binned position data for smoother plots
bin_size = SEQUENCE_LENGTH // POSITION_BINS
position_binned = []

for bin_idx in range(POSITION_BINS):
    start_pos = bin_idx * bin_size
    end_pos = start_pos + bin_size
    
    # Collect all probabilities in this position range
    bin_probs = []
    for pos in range(start_pos, min(end_pos, SEQUENCE_LENGTH)):
        bin_probs.extend(all_position_probs[pos])
    
    if bin_probs:
        position_binned.append({
            'bin_start': start_pos,
            'bin_end': end_pos,
            'bin_mid': (start_pos + end_pos) / 2,
            'mean_log_prob': np.mean(bin_probs),
            'median_log_prob': np.median(bin_probs),
            'std_log_prob': np.std(bin_probs),
            'count': len(bin_probs)
        })

binned_df = pd.DataFrame(position_binned)

# Save the data
position_df.to_csv('/home/scottviteri/Projects/gpt2check/position_effects.csv', index=False)
binned_df.to_csv('/home/scottviteri/Projects/gpt2check/position_effects_binned.csv', index=False)

# Create visualization plots
plt.figure(figsize=(15, 12))

# Plot 1: Mean log probability by position
plt.subplot(2, 2, 1)
plt.errorbar(position_df['position'], position_df['mean_log_prob'], 
             yerr=position_df['std_log_prob']/np.sqrt(position_df['count']),
             alpha=0.3, ecolor='gray')
plt.plot(position_df['position'], position_df['mean_log_prob'], 'o-', alpha=0.5, label='Mean')
plt.plot(position_df['position'], position_df['median_log_prob'], 's-', alpha=0.5, label='Median')
plt.xlabel('Token Position in Sequence')
plt.ylabel('Log Probability')
plt.title('Log Probability by Token Position (Raw)')
plt.legend()
plt.grid(True)

# Plot 2: Binned version for clarity
plt.subplot(2, 2, 2)
plt.errorbar(binned_df['bin_mid'], binned_df['mean_log_prob'], 
             yerr=binned_df['std_log_prob']/np.sqrt(binned_df['count']),
             ecolor='gray', capsize=3, alpha=0.7)
plt.plot(binned_df['bin_mid'], binned_df['mean_log_prob'], 'o-', linewidth=2, label='Mean')
plt.plot(binned_df['bin_mid'], binned_df['median_log_prob'], 's-', linewidth=2, label='Median')
plt.xlabel('Token Position in Sequence (Binned)')
plt.ylabel('Log Probability')
plt.title(f'Log Probability by Token Position (Binned, {POSITION_BINS} bins)')
plt.legend()
plt.grid(True)

# Plot 3: Sample count by position
plt.subplot(2, 2, 3)
plt.bar(position_df['position'], position_df['count'], alpha=0.6)
plt.xlabel('Token Position in Sequence')
plt.ylabel('Number of Samples')
plt.title('Sample Count by Position')
plt.grid(True)

# Plot 4: Mean-Median Gap by Position (Binned)
plt.subplot(2, 2, 4)
gap = binned_df['median_log_prob'] - binned_df['mean_log_prob']
plt.plot(binned_df['bin_mid'], gap, 'o-', linewidth=2)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Token Position in Sequence (Binned)')
plt.ylabel('Median - Mean Log Probability')
plt.title('Mean-Median Gap by Position')
plt.grid(True)

# Add a text box with summary statistics
sample_stats = f"""
Articles: {NUM_ARTICLES}
Samples per article: {NUM_SAMPLES_PER_ARTICLE}
Total samples: {successful_samples}
Sequence length: {SEQUENCE_LENGTH}
Overall mean log prob: {np.mean([s['mean_log_prob'] for s in article_stats]):.4f}
Overall median log prob: {np.median([s['median_log_prob'] for s in article_stats]):.4f}
"""

plt.figtext(0.02, 0.02, sample_stats, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Add title
plt.suptitle('GPT-2 Small Token Log Probability by Position in Context Window', fontsize=16, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('/home/scottviteri/Projects/gpt2check/position_effects.png', dpi=300)
print("Saved position effects plot to position_effects.png")

# Print summary
print(f"\nSUMMARY STATISTICS:")
print(f"Analyzed {successful_samples} samples across {NUM_ARTICLES} articles")
print(f"Average token log probability: {np.mean([s['mean_log_prob'] for s in article_stats]):.4f}")
print(f"Median token log probability: {np.median([s['median_log_prob'] for s in article_stats]):.4f}")

print("\nPosition effect statistics:")
# Group by position range for easier interpretation
groups = [(0, 50), (50, 100), (100, 200), (200, 400), (400, SEQUENCE_LENGTH)]
for start, end in groups:
    positions = position_df[(position_df['position'] >= start) & (position_df['position'] < end)]
    if not positions.empty:
        avg = np.nanmean(positions['mean_log_prob'])
        print(f"Position {start}-{end}: Avg log probability = {avg:.4f}")