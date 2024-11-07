import streamlit as st
from transformers import pipeline
from huggingface_hub import login

# Set up the page
st.set_page_config(page_title="Text GenAI Model", page_icon="🤖")
st.title("Text GenAI Model")
st.subheader("Answer Random Questions Using Hugging Face Models")

# Login to Hugging Face
access_token_read = st.secrets["HUGGINGFACE_TOKEN"]
login(token=access_token_read)

# Initialize the BlenderBot model
try:
    with st.spinner("Loading model..."):
        pipe = pipeline("conversational", model="distilgpt2 ", device=-1)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# User input
text = st.text_input("Ask a Random Question")

if text:
    # Generate response using conversational pipeline
    try:
        response = pipe(f"Answer the question: {text}")
        st.write(f"Answer: {response[0]['generated_text']}")
    except Exception as e:
        st.error(f"Error generating response: {e}")
