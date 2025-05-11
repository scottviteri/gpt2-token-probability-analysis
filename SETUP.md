# Setting Up GitHub Repository

To push this local repository to GitHub:

1. Create a new GitHub repository at https://github.com/new
   - Name: `gpt2-probability-analysis` (or your preferred name)
   - Set as public
   - Do not initialize with README (since we already have one)

2. Push your local repository to GitHub:
   ```bash
   cd /home/scottviteri/Projects/gpt2check
   git remote add origin https://github.com/YOUR-USERNAME/gpt2-probability-analysis.git
   git branch -M main
   git push -u origin main
   ```

3. After pushing, your analysis will be available at:
   `https://github.com/YOUR-USERNAME/gpt2-probability-analysis`

## Repository Structure

```
.
├── README.md                             # Project overview with key findings
├── check_gpt2_prob.py                    # Basic GPT-2 probability check
├── plot_gpt2_probs.py                    # Visualizations for token distributions
├── analyze_multiple_articles.py          # Multi-article comparison
├── analyze_position_effects.py           # Position effect analysis (small)
├── position_analysis_large_sample.py     # Position effect analysis (large)
├── gpt2_prob_plots.png                   # Distribution visualization
├── gpt2_multi_article_analysis.png       # Multi-article comparison plot
├── position_effects.png                  # Position effects (small sample)
├── position_effects_large_sample.png     # Position effects (large sample)
└── *.csv                                 # Results data files
```