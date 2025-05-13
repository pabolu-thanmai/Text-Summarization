# 1. Setup and Imports
!pip install transformers torch --quiet

import torch
from transformers import pipeline

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# 2. Model Selection
# Using BART for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# 3. User Input
input_text = input("Enter the text to summarize:\n")

# Ensure input is not empty
if input_text.strip():
    # Generate Summary
    summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)
    print("\nSummary:\n", summary[0]['summary_text'])
else:
    print("Please enter some text to summarize.")
