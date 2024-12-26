import time
import pandas as pd
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer

# List of models to test
model_names = [
    "gpt2",  # GPT-2 (small base model)
    #"EleutherAI/gpt-neo-125M",  # GPT-Neo (125M parameters)
    "EleutherAI/gpt-neo-1.3B",  # GPT-Neo (1.3B parameters, larger)
    #"EleutherAI/gpt-neo-2.7B",  # GPT-Neo (2.7B parameters, more powerful)
    "facebook/opt-125m",  # OPT (125M parameters, lightweight model)
    #"facebook/opt-350m",  # OPT (350M parameters, more performant)
    #"facebook/opt-1.3b",  # OPT (1.3B parameters)
    #"facebook/opt-2.7b",  # OPT (2.7B parameters)
    #"bigscience/bloom-560m",  # BLOOM (560M parameters, multilingual)
    #"bigscience/bloom-1b1",  # BLOOM (1.1B parameters)
    #"bigscience/bloom-3b",  # BLOOM (3B parameters)
]

# Input text
prompt = "In the near future, robots and humans"

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
    print(f"Generation time for model '{model_name}': {generation_time:.2f} seconds")
    print("Generated text:")
    print(generated_text)
    
    # Append the results to the list
    results.append({
        "Model": model_name,
        "Generated Text": generated_text,
        "Loading Time (s)": load_time,
        "Generation Time (s)": generation_time
    })

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Display the DataFrame
print("\nSummary of Results:")
print(df)

# Plotting with Plotly
fig = px.bar(
    df,
    x='Model',
    y=['Loading Time (s)', 'Generation Time (s)'],
    title="Loading and Generation Time for Each Model",
    labels={'value': 'Time (seconds)', 'variable': 'Type of Time'},
    barmode='stack',  # Stack the bars to show both loading and generation times
    color='variable',
    color_discrete_map={'Loading Time (s)': 'skyblue', 'Generation Time (s)': 'orange'}
)

# Show the plot
fig.show()