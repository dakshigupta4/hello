import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import torch
import os

# Free up GPU memory (optional, if using GPU)
torch.cuda.empty_cache()

# Set environment variable to avoid fragmentation (optional, if using GPU)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login with Hugging Face token
access_token_read = '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'
login(token=access_token_read)

# Set the device to CPU (Streamlit Cloud might not support GPU)
device = -1  # Using CPU for Streamlit Cloud
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device=device)

st.title("Text GenAI Model")

# Input prompt from the user
text = st.text_input("Enter Your Prompt")

if text:
    # Generate text using the pipeline
    gentext = pipe(text, max_new_tokens=100)
    
    # Extract and display the generated text
    st.write(gentext[0]["generated_text"])
