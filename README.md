# Model Loading and Text Generation Time Analysis

## Overview

This script evaluates and compares the loading and text generation times of several pre-trained language models. It also analyzes the diversity of the generated text and estimates the memory usage of each model. The results are saved in a CSV file and visualized using interactive plots.

## Features

- **Model Loading Time:** Measures the time it takes to load each model from the Hugging Face model hub.
- **Text Generation Performance:** Evaluates the time required to generate text for a set of prompts.
- **Diversity Analysis:** Computes a diversity score based on the number of unique words in the generated text.
- **Memory Estimation:** Estimates the memory usage of each model in megabytes (MB).
- **Results Export:** Saves all results to a CSV file for further analysis.
- **Interactive Visualization:** Uses Plotly to create bar charts showing performance metrics.

## Code Walkthrough

### 1. **Imports and Setup**
The script uses the following libraries:
- `time`: For measuring execution time.
- `pandas`: To store and manipulate the results.
- `plotly.express`: For creating interactive visualizations.
- `transformers`: To load and interact with pre-trained language models.

### 2. **List of Models**
The script tests multiple language models from the Hugging Face model hub:
- **`gpt2`:** A small and efficient GPT-2 model suitable for lightweight tasks.
- **`EleutherAI/gpt-neo-1.3B`:** A larger GPT-Neo model with 1.3 billion parameters, offering more complex responses.
- **`facebook/opt-125m`:** A lightweight model from Facebook's OPT family.

### 3. **Prompts for Text Generation**
Three prompts are used to evaluate the models:
1. "In the near future, robots and humans"
2. "Artificial intelligence will change the world in"
3. "Data centers are critical to modern infrastructure because"

### 4. **Metrics and Functions**
- **Diversity Score:** Computes the ratio of unique words to total words in the generated text.
- **Model Memory Estimation:** Estimates the memory usage based on the number of model parameters (assuming 4 bytes per parameter).

### 5. **Process**
For each model:
1. The model and tokenizer are loaded, and the loading time is measured.
2. The model processes each prompt to generate text.
3. The generation time is recorded.
4. The diversity of the generated text is calculated.
5. The estimated memory usage of the model is logged.
6. Results are stored in a DataFrame.

### 6. **Saving Results**
The results are exported to a CSV file named `results.csv`, containing:
- Model name
- Input prompt
- Generated text
- Diversity score
- Loading time (in seconds)
- Generation time (in seconds)
- Estimated memory usage (in MB)

### 7. **Visualization**
The script generates three bar charts using Plotly:
1. **Loading and Generation Time:** Stacked bar chart showing the time taken for loading and text generation for each model.

![Screenshot from 2024-12-26 15-19-56](https://github.com/user-attachments/assets/116f71d1-2e1f-47f7-9412-4e6b4f080d76)

2. **Diversity Scores:** Bar chart comparing the diversity scores of generated text.

![Screenshot from 2024-12-26 15-21-05](https://github.com/user-attachments/assets/bb278cf1-7861-4868-947a-87035862fe8a)

3. **Memory Usage:** Bar chart displaying the estimated memory usage of each model.

![Screenshot from 2024-12-26 15-21-36](https://github.com/user-attachments/assets/cb1c365c-02a5-43eb-a8e3-7f846c2d7551)

### CSV File
The script generates a CSV file (`results.csv`) with the following structure:
| Model                  | Prompt                                               | Generated Text                        | Diversity Score | Loading Time (s) | Generation Time (s) | Memory Usage (MB) |
|------------------------|-----------------------------------------------------|---------------------------------------|-----------------|------------------|---------------------|-------------------|
| gpt2                  | In the near future, robots and humans               | In the near future, robots...         | 0.75            | 1.008958         | 2.330392           | 474.700195        |
| gpt2                  | Artificial intelligence will change the world in    | Artificial intelligence will...       | 0.892857        | 1.008958         | 0.723652           | 474.700195        |
| gpt2                  | Data centers are critical to modern infrastructure  | Data centers are critical...          | 0.707865        | 1.008958         | 2.328390           | 474.700195        |
| EleutherAI/gpt-neo-1.3B| In the near future, robots and humans               | In the near future, robots...         | 0.685393        | 0.910809         | 18.094589          | 5018.523438       |
| EleutherAI/gpt-neo-1.3B| Artificial intelligence will change the world in    | Artificial intelligence will...       | 0.765432        | 0.910809         | 18.155935          | 5018.523438       |
| EleutherAI/gpt-neo-1.3B| Data centers are critical to modern infrastructure  | Data centers are critical...          | 0.766667        | 0.910809         | 21.922727          | 5018.523438       |
| facebook/opt-125m      | In the near future, robots and humans               | In the near future, robots...         | 0.625           | 0.973019         | 2.887027           | 477.750000        |
| facebook/opt-125m      | Artificial intelligence will change the world in    | Artificial intelligence will...       | 0.698795        | 0.973019         | 2.899922           | 477.750000        |
| facebook/opt-125m      | Data centers are critical to modern infrastructure  | Data centers are critical...          | 0.593407        | 0.973019         | 2.409411           | 477.750000        |


