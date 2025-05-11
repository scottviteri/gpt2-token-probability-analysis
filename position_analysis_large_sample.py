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
import os
import pickle

# Set up plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Configuration - optimized for large sample
NUM_ARTICLES = 30                  # Target number of articles
SEQUENCE_LENGTH = 256              # Length of context window
POSITION_BINS = 32                 # Number of bins for visualization
CACHE_FILE = '/home/scottviteri/Projects/gpt2check/position_data_cache.pkl'

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading GPT-2 small model and tokenizer...")
model_name = "gpt2"  # This is the small version
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

# Function to analyze an article for positional effects
def analyze_article_positions(article_text):
    """Analyze a single article for positional effects in log probabilities"""
    # Tokenize the full text
    tokens = tokenizer.encode(article_text)
    
    # We'll use multiple windows from this article, with some overlap
    # This gives us more samples while being efficient
    position_probs = {i: [] for i in range(SEQUENCE_LENGTH)}
    
    # Determine how many windows we can extract
    # We want at least 25% new content in each window
    stride = SEQUENCE_LENGTH // 4
    max_start = max(0, len(tokens) - SEQUENCE_LENGTH)
    
    # Process multiple windows from this article
    window_positions = list(range(0, max_start + 1, stride))
    if not window_positions:  # Ensure we at least process one window
        window_positions = [0]
    
    for start_pos in window_positions[:8]:  # Limit to 8 windows per article to prevent very long articles from dominating
        # Extract window
        end_pos = min(start_pos + SEQUENCE_LENGTH, len(tokens))
        window_tokens = tokens[start_pos:end_pos]
        
        # Skip windows that are too short
        if len(window_tokens) < SEQUENCE_LENGTH // 2:
            continue
            
        # Convert to tensor and process
        input_ids = torch.tensor([window_tokens]).to(device)
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits
            
            # Calculate token probabilities
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            for i in range(min(shift_logits.size(1), SEQUENCE_LENGTH)):
                token_logits = shift_logits[0, i]
                token_label = shift_labels[0, i].item()
                token_log_prob = torch.log_softmax(token_logits, dim=0)[token_label].item()
                
                # Store by position in sequence
                position_probs[i].append(token_log_prob)
    
    return position_probs

# Check if we have cached data to use
if os.path.exists(CACHE_FILE):
    print(f"Loading cached position data from {CACHE_FILE}")
    with open(CACHE_FILE, 'rb') as f:
        all_position_probs, article_count = pickle.load(f)
    print(f"Loaded data from {article_count} articles")
else:
    # Initialize data storage
    all_position_probs = {i: [] for i in range(SEQUENCE_LENGTH)}
    article_count = 0
    
    # Predefined popular Wikipedia topics to ensure we get substantial content
    popular_topics = [
        "United States", "World War II", "Albert Einstein", "Barack Obama", 
        "China", "India", "Russia", "United Kingdom", "France", "Germany",
        "Ancient Rome", "Ancient Greece", "Middle Ages", "Renaissance",
        "Industrial Revolution", "American Civil War", "Cold War",
        "Artificial intelligence", "Internet", "Climate change",
        "Solar System", "Mars", "Jupiter", "Black hole", "Quantum mechanics",
        "Theory of relativity", "Evolution", "DNA", "Atom", "Chemistry",
        "Mathematics", "Physics", "Biology", "Economics", "Psychology",
        "Literature", "Music", "Art", "Film", "Television", "Sports",
        "Football", "Basketball", "Tennis", "Cricket", "Olympics",
        "New York City", "Tokyo", "London", "Paris", "Moscow", "Beijing",
        "Africa", "Europe", "Asia", "North America", "South America",
        "Pacific Ocean", "Atlantic Ocean", "Amazon Rainforest", "Sahara Desert",
        "Mount Everest", "Great Barrier Reef", "Grand Canyon", "Niagara Falls",
        "Leonardo da Vinci", "Mozart", "Shakespeare", "Beethoven",
        "Mahatma Gandhi", "Nelson Mandela", "Martin Luther King Jr.",
        "Microsoft", "Apple Inc.", "Google", "Amazon (company)", "Facebook"
    ]
    
    # Process articles
    print(f"Analyzing up to {NUM_ARTICLES} articles...")
    random.shuffle(popular_topics)
    pbar = tqdm(total=NUM_ARTICLES)
    
    for topic in popular_topics:
        if article_count >= NUM_ARTICLES:
            break
            
        try:
            # Fetch article
            article = wikipedia.page(topic)
            article_text = article.content
            
            # Process if article is long enough
            if len(article_text.split()) > SEQUENCE_LENGTH:
                print(f"Processing: {topic}")
                
                # Analyze article
                article_position_probs = analyze_article_positions(article_text)
                
                # Add to overall data
                for pos, probs in article_position_probs.items():
                    all_position_probs[pos].extend(probs)
                
                article_count += 1
                pbar.update(1)
                
                # Save cache periodically
                if article_count % 5 == 0:
                    with open(CACHE_FILE, 'wb') as f:
                        pickle.dump((all_position_probs, article_count), f)
                    print(f"Saved cache with {article_count} articles")
        
        except Exception as e:
            print(f"Error processing '{topic}': {str(e)}")
            
        # Small delay to avoid hammering Wikipedia API
        time.sleep(0.5)
    
    pbar.close()
    
    # Save final cache
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump((all_position_probs, article_count), f)
    print(f"Saved cache with {article_count} articles")

# Process the data for analysis
position_stats = []
total_tokens = sum(len(probs) for probs in all_position_probs.values())

for pos in range(SEQUENCE_LENGTH):
    probs = all_position_probs[pos]
    if probs:
        position_stats.append({
            'position': pos,
            'mean_log_prob': np.mean(probs),
            'median_log_prob': np.median(probs),
            'std_log_prob': np.std(probs),
            'count': len(probs),
            'min_log_prob': np.min(probs),
            'max_log_prob': np.max(probs)
        })

# Convert to DataFrame
position_df = pd.DataFrame(position_stats)

# Create binned data for clearer trends
bin_size = SEQUENCE_LENGTH // POSITION_BINS
binned_stats = []

for bin_idx in range(POSITION_BINS):
    bin_start = bin_idx * bin_size
    bin_end = min(bin_start + bin_size, SEQUENCE_LENGTH)
    
    bin_positions = position_df[(position_df['position'] >= bin_start) & 
                               (position_df['position'] < bin_end)]
    
    if not bin_positions.empty:
        bin_probs = []
        for _, row in bin_positions.iterrows():
            pos = row['position']
            bin_probs.extend(all_position_probs[pos])
            
        if bin_probs:
            binned_stats.append({
                'bin_start': bin_start,
                'bin_end': bin_end,
                'bin_mid': (bin_start + bin_end) / 2,
                'mean_log_prob': np.mean(bin_probs),
                'median_log_prob': np.median(bin_probs),
                'std_log_prob': np.std(bin_probs),
                'count': len(bin_probs)
            })

binned_df = pd.DataFrame(binned_stats)

# Save the data
position_df.to_csv('/home/scottviteri/Projects/gpt2check/large_sample_position_effects.csv', index=False)
binned_df.to_csv('/home/scottviteri/Projects/gpt2check/large_sample_position_effects_binned.csv', index=False)

# Create visualization plots
plt.figure(figsize=(15, 12))

# Plot 1: Mean and median log probability by position
plt.subplot(2, 2, 1)
# Plot with error bands (shaded regions) for better visualization
plt.fill_between(position_df['position'], 
                 position_df['mean_log_prob'] - position_df['std_log_prob']/np.sqrt(position_df['count']),
                 position_df['mean_log_prob'] + position_df['std_log_prob']/np.sqrt(position_df['count']),
                 alpha=0.2, color='blue')
plt.plot(position_df['position'], position_df['mean_log_prob'], 
         color='blue', linewidth=1, alpha=0.6, label='Mean')
plt.plot(position_df['position'], position_df['median_log_prob'], 
         color='green', linewidth=1, alpha=0.6, label='Median')

# Add smoothed trendlines
window = 10
position_df['mean_smooth'] = position_df['mean_log_prob'].rolling(window=window, center=True).mean()
position_df['median_smooth'] = position_df['median_log_prob'].rolling(window=window, center=True).mean()
plt.plot(position_df['position'], position_df['mean_smooth'], 
         color='blue', linewidth=2, label='Mean (smoothed)')
plt.plot(position_df['position'], position_df['median_smooth'], 
         color='green', linewidth=2, label='Median (smoothed)')

plt.xlabel('Token Position in Sequence')
plt.ylabel('Log Probability')
plt.title('Log Probability by Token Position')
plt.legend()
plt.grid(True)

# Plot 2: Binned version for clearer trend
plt.subplot(2, 2, 2)
plt.errorbar(binned_df['bin_mid'], binned_df['mean_log_prob'], 
             yerr=binned_df['std_log_prob']/np.sqrt(binned_df['count']),
             fmt='o-', linewidth=2, label='Mean', capsize=4)
plt.errorbar(binned_df['bin_mid'], binned_df['median_log_prob'], 
             yerr=binned_df['std_log_prob']/np.sqrt(binned_df['count']),
             fmt='s-', linewidth=2, label='Median', capsize=4)
plt.xlabel('Token Position in Sequence (Binned)')
plt.ylabel('Log Probability')
plt.title(f'Log Probability by Token Position (Binned, {POSITION_BINS} bins)')
plt.legend()
plt.grid(True)

# Plot 3: Median-Mean gap by position (binned)
plt.subplot(2, 2, 3)
gap = binned_df['median_log_prob'] - binned_df['mean_log_prob']
plt.bar(binned_df['bin_mid'], gap, width=bin_size*0.8, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Token Position in Sequence (Binned)')
plt.ylabel('Median - Mean Log Probability')
plt.title('Median-Mean Gap by Position')
plt.grid(True)

# Plot 4: Sample count distribution
plt.subplot(2, 2, 4)
plt.bar(binned_df['bin_mid'], binned_df['count'], width=bin_size*0.8, alpha=0.7)
plt.xlabel('Token Position in Sequence (Binned)')
plt.ylabel('Number of Samples')
plt.title('Sample Count by Position')
plt.grid(True)

# Add a text box with summary statistics
stats_text = f"""
Sample Statistics:
Articles analyzed: {article_count}
Total token samples: {total_tokens}
Overall mean log prob: {position_df['mean_log_prob'].mean():.4f}
Overall median log prob: {position_df['median_log_prob'].mean():.4f}
Position effect: {position_df['mean_log_prob'].iloc[-50:].mean() - position_df['mean_log_prob'].iloc[:50].mean():.4f}
"""

plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Add overall title
plt.suptitle(f'GPT-2 Small Token Log Probability by Position (Large Sample)\n{article_count} articles, {total_tokens} token samples', 
             fontsize=16, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('/home/scottviteri/Projects/gpt2check/position_effects_large_sample.png', dpi=300)
print("Saved position effects plot to position_effects_large_sample.png")

# Print detailed statistics
print("\nPosition effect statistics (using binned data):")
first_bin = binned_df.iloc[0]['mean_log_prob']
last_bin = binned_df.iloc[-1]['mean_log_prob']
improvement = last_bin - first_bin

print(f"First bin mean log prob: {first_bin:.4f}")
print(f"Last bin mean log prob: {last_bin:.4f}")
print(f"Improvement: {improvement:.4f} ({improvement/abs(first_bin)*100:.1f}%)")

# Group by position ranges for easier interpretation
print("\nMean log probability by position range:")
ranges = [(0, 32), (32, 64), (64, 128), (128, 192), (192, 256)]
for start, end in ranges:
    bins_in_range = binned_df[(binned_df['bin_mid'] >= start) & (binned_df['bin_mid'] < end)]
    if not bins_in_range.empty:
        mean_prob = bins_in_range['mean_log_prob'].mean()
        median_prob = bins_in_range['median_log_prob'].mean()
        gap = median_prob - mean_prob
        print(f"Position {start}-{end}: Mean={mean_prob:.4f}, Median={median_prob:.4f}, Gap={gap:.4f}")