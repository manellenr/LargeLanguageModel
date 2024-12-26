# Model Loading and Text Generation Time Analysis

## Overview

This script evaluates and compares the loading and text generation times of several pre-trained language models. It measures the performance of models like GPT-2, GPT-Neo, and OPT on a simple text generation task. The results provide insights into the trade-offs between model size, performance, and generation speed.

## Code Walkthrough

1. **Imports and Setup:**
   - We import necessary libraries like `time` for time measurement, `pandas` for handling results, `plotly.express` for visualization, and `transformers` for loading pre-trained models.
   
2. **List of Models:**
   - The script tests multiple language models from the Hugging Face model hub:
     - `gpt2`: A smaller GPT-2 model.
     - `EleutherAI/gpt-neo-1.3B`: A larger GPT-Neo model with 1.3 billion parameters.
     - `facebook/opt-125m`: A lightweight model from Facebook's OPT family.
   
3. **Text Generation:**
   - For each model, the script:
     1. Loads the model and tokenizer.
     2. Measures the loading time.
     3. Tokenizes the input prompt and generates a response.
     4. Measures the time taken to generate the text.
     5. Decodes generated text into readable format.

## Example Output

Here is a sample output after running the script:

![Screenshot from 2024-12-26 12-20-56](https://github.com/user-attachments/assets/00f58751-27fe-4319-9d76-d583022c33cd)
