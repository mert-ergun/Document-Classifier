import os
import json
import requests
import streamlit as st
from document_classifier.utils.translations import t
from document_classifier.utils.config import API_URL

def get_installed_models():
    response = requests.get(f"{API_URL}/tags")
    if response.status_code == 200:
        models = response.json()["models"]
        return [model["name"].split(":")[0] for model in models]
    else:
        st.error(t("failed_to_fetch", "en"))
        return []
    
def classify_document(text, model):
    lang = st.session_state.lang_code
    try:
        response = requests.post(f"{API_URL}/classify", json={"text": text, "model": model, "lang": lang})
        response.raise_for_status()
        return response.json()["classification"]
    except requests.RequestException as e:
        st.error(t("error_classifying", "en"))
        return "Classification Error"
    
def explain_classification(text, model, classification):
    lang = st.session_state.lang_code
    try:
        response = requests.post(f"{API_URL}/explain", json={"text": text, "model": model, "classification": classification, "lang": lang})
        response.raise_for_status()
        return response.json()["explanation"]
    except requests.RequestException as e:
        st.error(f"Error explaining classification: {str(e)}")
        return "Explanation Error"
    

    
