import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import torch
import os

# Set page configuration
st.set_page_config(page_title="Text GenAI Model", page_icon="🤖")
st.title("Text GenAI Model")
st.subheader("Generate Text Using Hugging Face Models")

# Fetch Hugging Face token from Streamlit Secrets
access_token_read = st.secrets["HUGGINGFACE_TOKEN"]  # Ensure this is set in your Streamlit Cloud Secrets

# Check if token is available
if access_token_read:
    st.write("Hugging Face token loaded successfully.")
else:
    st.write("Hugging Face token is missing. Please ensure it's in the Streamlit Secrets section.")

# Free up GPU memory (if using GPU)
torch.cuda.empty_cache()

# Set environment variable to avoid fragmentation (for GPU usage, optional)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login to Hugging Face Hub using the access token
try:
    login(token=access_token_read)
except Exception as e:
    st.write(f"Error while logging in to Hugging Face: {e}")

# Initialize the text generation pipeline with a smaller model (for better compatibility on Streamlit Cloud)
pipe = pipeline("text-generation", model="distilgpt2", device=-1)  # Using CPU, adjust based on availability

# Input from the user
text = st.text_input("Enter Your Prompt")

if text:
    # Generate text using the pipeline
    gentext = pipe(text, max_new_tokens=100)
    
    # Extract and display the generated text
    st.write(gentext[0]["generated_text"])
