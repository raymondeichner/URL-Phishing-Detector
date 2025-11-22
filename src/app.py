import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

id_to_type = {0: "legitimate", 1: "phishing"}

def classify(url):
    """Classify a URL as legitimate or phishing."""
    inputs = tokenizer(
        url,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=1).item()
    return id_to_type[predicted_id]

st.title("URL Phishing Detector")

url_input = st.text_input("Enter a URL to analyze:")

if st.button("Analyze"):
    if url_input.strip():
        result = classify(url_input)
        if result == "phishing":
            st.error("The URL is classified as PHISHING")
        else:
            st.success("The URL is classified as LEGITIMATE")
    else:
        st.warning("Please enter a valid URL.")
