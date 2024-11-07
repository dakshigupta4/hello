import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import os

# Set page configuration
st.set_page_config(page_title="Text GenAI Model", page_icon="🤖")
st.title("Text GenAI Model")
st.subheader("Answer Random Questions Using Hugging Face Models")

# Fetch Hugging Face token from Streamlit Secrets
access_token_read = st.secrets["HUGGINGFACE_TOKEN"]

# Set environment variable to avoid fragmentation (optional)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login to Hugging Face Hub using the access token
login(token=access_token_read)

try:
    # Load the larger model for text generation
    with st.spinner("Loading model... This may take a few moments."):
        pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device=-1)  # Set to CPU
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Input from the user
text = st.text_input("Ask a Random Question")

if text:
    # Generate text based on the random question
    try:
        response = pipe(f"Answer the question: {text}", max_length=150, num_return_sequences=1)
        st.write(f"Answer: {response[0]['generated_text']}")
    except Exception as e:
        st.error(f"An error occurred while generating text: {e}")
