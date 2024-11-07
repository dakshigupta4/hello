import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from huggingface_hub import login

# Set up the page
st.set_page_config(page_title="Text GenAI Model", page_icon="🤖")
st.title("Text GenAI Model")
st.subheader("Answer Random Questions Using Hugging Face Models")

# Fetch Hugging Face token from Streamlit Secrets
access_token_read = st.secrets["HUGGINGFACE_TOKEN"]

# Login to Hugging Face
login(token=access_token_read)

# Initialize the BlenderBot tokenizer and model
try:
    with st.spinner("Loading model..."):
        tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Input from the user
text = st.text_input("Ask a Random Question")

if text:
    try:
        # Encode the input question
        inputs = tokenizer([text], return_tensors="pt")
        
        # Generate the response
        reply_ids = model.generate(inputs["input_ids"], max_length=150)
        
        # Decode the generated response
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        
        # Display the generated response
        st.write(f"Answer: {response}")
    except Exception as e:
        st.error(f"Error generating response: {e}")
