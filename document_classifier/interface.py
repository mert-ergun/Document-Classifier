import streamlit as st
import pandas as pd
import plotly.express as px
import PyPDF2
import docx
import io
import requests
import time
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_installed_models():
    response = requests.get(f"{API_URL}/tags")
    if response.status_code == 200:
        models = response.json()["models"]
        return [model["name"].split(":")[0] for model in models]
    else:
        st.error("Failed to fetch installed models.")
        return []
    
def classify_document(text, model):
    try:
        response = requests.post(f"{API_URL}/classify", json={"text": text, "model": model})
        response.raise_for_status()
        return response.json()["classification"]
    except requests.RequestException as e:
        st.error(f"An error occurred while classifying the document: {str(e)}")
        return "Classification Error"

def extract_text_from_document(file):
    # If file is a string, it's a file path
    if isinstance(file, str):
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension == '.txt':
            with open(file, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_extension == '.pdf':
            try:
                with open(file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return ""
        elif file_extension in ['.docx', '.doc']:
            try:
                doc = docx.Document(file)
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(f"Error reading DOCX: {str(e)}")
                return ""
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
        
    else:
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'txt':
            return file.getvalue().decode("utf-8")
        
        elif file_extension == 'pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return ""
        
        elif file_extension in ['docx', 'doc']:
            try:
                doc = docx.Document(io.BytesIO(file.getvalue()))
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(f"Error reading DOCX: {str(e)}")
                return ""
        
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
    
def scan_directory(directory, model):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.txt', '.pdf', '.docx', '.doc')):
                file_path = os.path.join(root, file)
                text = extract_text_from_document(file_path)
                if text:
                    classification = classify_document(text, model)
                    results.append({"Filename": file, "Classification": classification})
    return results

st.set_page_config(page_title="Document Classification System", layout="wide")

st.title("Document Classification System")

installed_models = get_installed_models()
local_models = ["gemma2", "phi3", "llama3", "mistral", "llama3.1", "mistral-nemo", "mertergun/phi3_finetuned", "gemma2", "qwen2"]
all_models = installed_models + [m for m in local_models if m not in installed_models] + ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-turbo", "gpt-4o", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gpt-4o-mini"]

model = st.selectbox("Select Classification Model", all_models)
if model not in installed_models and model in local_models:
            if st.button("Pull Selected Model"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(completed, total):
                    progress = (completed / total) * 100 if total > 0 else 0
                    progress = min(progress, 100)  # Ensure progress doesn't exceed 100%
                    progress_bar.progress(int(progress))
                    status_text.text(f"Pulling model... {progress:.2f}% ({completed}/{total} bytes)")

                response = requests.post(f"{API_URL}/pull_model", json={"model": model}, stream=True)
                if response.status_code == 200:
                    total_size = None
                    completed_size = 0
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if 'total' in data:
                                    total_size = int(data['total'])
                                if 'completed' in data:
                                    completed_size = int(data['completed'])
                                    if total_size:
                                        update_progress(completed_size, total_size)
                            except json.JSONDecodeError:
                                status_text.text(line.decode('utf-8'))
                    
                    if total_size and completed_size == total_size:
                        st.success(f"Model '{model}' has been successfully pulled.")
                    elif total_size:
                        st.warning(f"Model pull may not have completed. Please check the model status.")
                    else:
                        st.info(f"Model '{model}' pull process completed, but size information was not available.")
                    
                    time.sleep(1)  
                    st.rerun()
                else:
                    st.error("Failed to pull the selected model.")

# Set a radio button to select the select upload files or scan a directory
option = st.radio("Choose an option", ["Upload Files", "Scan Directory"])

if option == "Scan Directory":
    st.write("Please select a directory to scan for document classification.")
    directory = st.text_input("Directory path", value=".")
    if st.button("Scan Directory"):
        with st.spinner(f'Scanning directory {directory}...'):
            results = scan_directory(directory, model)
        
        if results:
            results_df = pd.DataFrame(results)
            st.subheader("Classification Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            fig = px.pie(results_df, names='Classification', title='Distribution of Document Classifications')
            st.plotly_chart(fig)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("No valid documents were processed. Please check your files and try again.")

else:
    uploaded_files = st.file_uploader("Upload document(s) for classification", accept_multiple_files=True, type=['txt', 'pdf', 'docx'])

    if uploaded_files:
        results = []
        for file in uploaded_files:
            with st.spinner(f'Processing {file.name}...'):
                text = extract_text_from_document(file)
                if text:
                    classification = classify_document(text, model)
                    results.append({"Filename": file.name, "Classification": classification})
        
        if results:
            results_df = pd.DataFrame(results)
            
            if len(results) == 1:
                st.success(f"File Classification Complete!")
                st.markdown(f"<h2 style='text-align: center; color: #1E90FF;'>Classification Result</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>File: {results[0]['Filename']}</h3>", unsafe_allow_html=True)
                
                classification = results[0]['Classification']
                color = "#000000" 
                if classification.lower() == "top secret":
                    color = "#FF0000"  
                elif classification.lower() == "secret":
                    color = "#FFA500"  
                elif classification.lower() == "confidential":
                    color = "#800080" 
                elif classification.lower() == "restricted":
                    color = "#0000FF" 
                elif classification.lower() == "unclassified":
                    color = "#008000"  
                
                st.markdown(f"<h1 style='text-align: center; color: {color};'>{classification}</h1>", unsafe_allow_html=True)

            else:
                st.subheader("Classification Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization
                fig = px.pie(results_df, names='Classification', title='Distribution of Document Classifications')
                st.plotly_chart(fig)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("No valid documents were processed. Please check your files and try again.")

st.sidebar.title("About")
st.sidebar.info(
    "This document classification system uses advanced Language Model technology "
    "to categorize your documents into five security levels: Top Secret, Secret, "
    "Confidential, Restricted, and Unclassified. Simply upload your documents and "
    "receive instant classification results."
)

st.sidebar.title("Instructions")
st.sidebar.write(
    "1. Select a classification model from the dropdown menu.\n"
    "2. Upload one or more documents using the file uploader.\n"
    "3. Supported file types: TXT, PDF, DOCX\n"
    "4. The system will automatically process and classify each document.\n"
    "5. View the results in the table and pie chart.\n"
    "6. Download the results as a CSV file if needed."
)