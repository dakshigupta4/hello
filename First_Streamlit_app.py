import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import torch
import os

# Set page configuration
st.set_page_config(page_title="Text GenAI Model", page_icon="🤖")
st.title("Text GenAI Model")
st.subheader("Answer Random Questions Using Hugging Face Models")

# Fetch Hugging Face token from Streamlit Secrets
access_token_read = st.secrets["HUGGINGFACE_TOKEN"]  # Ensure this is set in your Streamlit Cloud Secrets

# Free up GPU memory (if using GPU)
torch.cuda.empty_cache()

# Set environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login to Hugging Face Hub using the access token
login(token=access_token_read)

# Initialize the text generation pipeline with GPT-2 model
pipe = pipeline("text-generation", model="gpt2", device=-1)  # Using CPU

# Input from the user
text = st.text_input("Ask a Random Question")

if text:
    # Generate text based on the random question
    response = pipe(f"Answer the question: {text}", max_length=150, num_return_sequences=1)
    
    # Display the generated response
    st.write(f"Answer: {response[0]['generated_text']}")
