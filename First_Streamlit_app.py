import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import torch
import os

# Free up any unnecessary cache memory
torch.cuda.empty_cache()

# Set environment variable for PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Use Streamlit secrets for the access token
access_token_read = st.secrets["HUGGINGFACE_TOKEN"]  # Ensure this is configured in Streamlit Cloud
login(token=access_token_read)

# Use CPU as Streamlit Cloud does not have GPU support
device = -1

# Load the text generation pipeline
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device=device)

st.title("Text GenAI Model")
text = st.text_input("Enter Your Prompt")

if text:
    messages = [{"role": "user", "content": text}]
    gentext = pipe(messages, max_new_tokens=100)
    st.write(gentext[0]["generated_text"])  # Adjust based on the output format of your model
