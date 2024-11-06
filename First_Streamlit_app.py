import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import torch
import os

# Set page configuration
st.set_page_config(page_title="Text GenAI Model", page_icon="🤖")
st.title("Text GenAI Model")
st.subheader("Answer Questions Using Hugging Face Models")

# Fetch Hugging Face token from Streamlit Secrets
access_token_read = st.secrets["HUGGINGFACE_TOKEN"]  # Ensure this is set in your Streamlit Cloud Secrets

# Free up GPU memory (if using GPU)
torch.cuda.empty_cache()

# Set environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login to Hugging Face Hub using the access token
login(token=access_token_read)

# Initialize the question-answering pipeline with a QA model
pipe = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)  # Using CPU

# Context for answering questions
context = """
Delhi is the capital city of India. It is located in the northern part of the country and is known for its rich history, culture, and modern development. Delhi is a major hub for politics, business, and culture in India. It consists of two parts: Old Delhi, which has historical monuments like the Red Fort, and New Delhi, which serves as the seat of the Indian government.
"""

# Input from the user
text = st.text_input("Ask a Random Question")

if text:
    # Use the question-answering pipeline to find the answer based on the context
    result = pipe(question=text, context=context)
    
    # Display the answer
    st.write(f"Answer: {result['answer']}")
