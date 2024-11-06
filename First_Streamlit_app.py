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

# Free up GPU memory (if using GPU)
torch.cuda.empty_cache()

# Set environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login to Hugging Face Hub using the access token
login(token=access_token_read)

# Initialize the text generation pipeline with GPT-2 (or a similar model)
pipe = pipeline("text-generation", model="gpt2", device=-1)  # Using CPU for Streamlit Cloud

# Input from the user
text = st.text_input("Ask a Random Question")

if text:
    # Generate a response using the text generation pipeline
    try:
        gentext = pipe(text, max_new_tokens=100)  # Generate text based on the input prompt
        # Extract and display the generated text
        st.write(gentext[0]["generated_text"])
    except Exception as e:
        st.error(f"Error generating text: {e}")
