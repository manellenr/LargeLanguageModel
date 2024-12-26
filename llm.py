import time
import pandas as pd
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer

# List of models to test
model_names = [
    "gpt2",  # GPT-2 (small base model)
    "EleutherAI/gpt-neo-1.3B",  # GPT-Neo (1.3B parameters, larger)
    "facebook/opt-125m",  # OPT (125M parameters, lightweight model)
]

# List of prompts to test
prompts = [
    "In the near future, robots and humans",
    "Artificial intelligence will change the world in",
    "Data centers are critical to modern infrastructure because"
]

# Function to calculate diversity score
def diversity_score(text):
    words = text.split()
    unique_words = len(set(words))
    return unique_words / len(words) if words else 0

# Function to estimate model memory size
def estimate_model_size(model):
    param_size = sum(p.numel() for p in model.parameters()) * 4
    return param_size / (1024 ** 2)

# Create an empty list to store results
results = []

# Loop through each model
for model_name in model_names:
    print(f"\nLoading and generating with model: {model_name}")

    # Measure loading time
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    load_time = time.time() - start_time
    print(f"Loading time for model '{model_name}': {load_time:.2f} seconds")

    # Estimate memory size
    memory_usage = estimate_model_size(model)

    for prompt in prompts:
        # Tokenize the input text
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Measure generation time
        start_time = time.time()
        output = model.generate(
            input_ids,
            max_length=100,  # Maximum number of tokens to generate
            num_return_sequences=1,  # Number of sequences to generate
            no_repeat_ngram_size=2,  # Prevent repetition of patterns
            top_k=50,  # Filter improbable words
            top_p=0.95,  # Nucleus sampling
            temperature=0.7,  # Control creativity
        )
        generation_time = time.time() - start_time

        # Decode the result
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        diversity = diversity_score(generated_text)

        print(f"Generation time for model '{model_name}' and prompt '{prompt}': {generation_time:.2f} seconds")
        print("Generated text:")
        print(generated_text)

        # Append the results to the list
        results.append({
            "Model": model_name,
            "Prompt": prompt,
            "Generated Text": generated_text,
            "Diversity Score": diversity,
            "Loading Time (s)": load_time,
            "Generation Time (s)": generation_time,
            "Memory Usage (MB)": memory_usage
        })

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv("results.csv", index=False)
print("\nResults have been saved to 'results.csv'.")

# Display the DataFrame
print("\nSummary of Results:")
print(df)

# Plotting with Plotly
fig1 = px.bar(
    df,
    x='Model',
    y=['Loading Time (s)', 'Generation Time (s)'],
    title="Loading and Generation Time for Each Model",
    labels={'value': 'Time (seconds)', 'variable': 'Type of Time'},
    barmode='stack', 
    color='variable',
    color_discrete_map={'Loading Time (s)': 'skyblue', 'Generation Time (s)': 'orange'}
)
fig1.show()

fig2 = px.bar(
    df,
    x='Model',
    y='Diversity Score',
    title="Diversity Score of Generated Texts by Model",
    labels={'Diversity Score': 'Diversity Score'},
    color='Model'
)
fig2.show()

fig3 = px.bar(
    df,
    x='Model',
    y='Memory Usage (MB)',
    title="Estimated Memory Usage by Model",
    labels={'Memory Usage (MB)': 'Memory Usage (MB)'},
    color='Model'
)
fig3.show()
