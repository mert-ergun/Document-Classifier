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
from openpyxl import load_workbook
from pptx import Presentation
import csv

API_URL = os.getenv("API_URL", "http://localhost:8000")

translations = {
    'en': {
        'title': "Document Classification System",
        'model_select': "Select Classification Model",
        'pull_model': "Pull Selected Model",
        'pulling_model': "Pulling model...",
        'option_radio': "Choose an option",
        'upload_files': "Upload Files",
        'scan_directory': "Scan Directory",
        'please_select': "Please select a directory to scan for document classification.",
        'directory_path': "Directory path",
        'scanning_directory': "Scanning directory",
        'failed_to_fetch': "Failed to fetch installed models.",
        'error_classifying': "An error occurred while classifying the document: {str(e)}",
        'error_reading_pdf': "Error reading PDF: {str(e)}",
        'error_reading_docx': "Error reading DOCX: {str(e)}",
        'unsupported_file': "Unsupported file type: {file_extension}",
        'classification_results': "Classification Results",
        'classification': "Classification",
        'distribution': "Distribution of Document Classifications",
        'download_results': "Download results as CSV",
        'no_valid_documents': "No valid documents were processed. Please check your files and try again.",
        'upload_document': "Upload document(s) for classification",
        'completed': "File Classification Complete!",
        'file': "File:",

        'about': "About",
        'about1': 'This document classification system uses advanced Language Model technology',
        'about2': 'to categorize your documents into five security levels: Top Secret, Secret,',
        'about3': 'Confidential, Restricted, and Unclassified. Simply upload your documents and',
        'about4': 'receive instant classification results.',

        'instructions': "Instructions",
        'instructions1': "1. Select a classification model from the dropdown menu.",
        'instructions2': "2. Upload one or more documents using the file uploader.",
        'instructions3': "3. Supported file types: TXT, PDF, DOCX",
        'instructions4': "4. The system will automatically process and classify each document.",
        'instructions5': "5. View the results in the table and pie chart.",
        'instructions6': "6. Download the results as a CSV file if needed."
    },
    'tr': {
        'title': "Belge Sınıflandırma Sistemi",
        'model_select': "Sınıflandırma Modelini Seçin",
        'pull_model': "Seçilen Modeli Çek",
        'pulling_model': "Model çekiliyor... ",
        'option_radio': "Bir seçenek belirleyin",
        'upload_files': "Dosya Yükle",
        'scan_directory': "Dizini Tara",
        'please_select': "Belge sınıflandırması için taranacak bir dizin seçin.",
        'directory_path': "Dizin yolu",
        'scanning_directory': "Dizin taranıyor",
        'failed_to_fetch': "Yüklü modeller alınamadı.",
        'error_classifying': "Belge sınıflandırılırken bir hata oluştu: {str(e)}",
        'error_reading_pdf': "PDF okunurken hata oluştu: {str(e)}",
        'error_reading_docx': "DOCX okunurken hata oluştu: {str(e)}",
        'unsupported_file': "Desteklenmeyen dosya türü: {file_extension}",
        'classification_results': "Sınıflandırma Sonuçları",
        'classification': "Sınıflandırma",
        'distribution': "Belge Sınıflandırmalarının Dağılımı",
        'download_results': "Sonuçları CSV olarak indir",
        'no_valid_documents': "Geçerli belge işlenmedi. Lütfen dosyalarınızı kontrol edin ve tekrar deneyin.",
        'upload_document': "Sınıflandırma için belge(ler) yükleyin",
        'completed': "Dosya Sınıflandırma Tamamlandı!",
        'file': "Dosya:",

        'about': "Hakkında",
        'about1': 'Bu belge sınıflandırma sistemi, belgelerinizi beş güvenlik seviyesine',
        'about2': 'sınıflandırmak için gelişmiş Dil Modeli teknolojisini kullanır: Çok Gizli, Gizli,',
        'about3': 'Hizmete Özel, Sınırlı ve Sınıflandırılmamış. Belgelerinizi yükleyin ve',
        'about4': 'anında sınıflandırma sonuçları alın.',

        'instructions': "Kullanım Talimatları",
        'instructions1': "1. Açılır menüden bir sınıflandırma modeli seçin.",
        'instructions2': "2. Dosya yükleyin veya dizin taraması yapın.",
        'instructions3': "3. Desteklenen dosya türleri: TXT, PDF, DOCX",
        'instructions4': "4. Sistem otomatik olarak her belgeyi işleyecek ve sınıflandıracak.",
        'instructions5': "5. Sonuçları tablo ve pasta grafiği ile görüntüleyin.",
        'instructions6': "6. Sonuçları CSV dosyası olarak indirin."
    }
}

def t(key):
    return translations[st.session_state.lang_code].get(key, key)


def get_installed_models():
    response = requests.get(f"{API_URL}/tags")
    if response.status_code == 200:
        models = response.json()["models"]
        return [model["name"].split(":")[0] for model in models]
    else:
        st.error(t("failed_to_fetch"))
        return []
    
def classify_document(text, model):
    lang = st.session_state.lang_code
    try:
        response = requests.post(f"{API_URL}/classify", json={"text": text, "model": model, "lang": lang})
        response.raise_for_status()
        return response.json()["classification"]
    except requests.RequestException as e:
        st.error(t("error_classifying"))
        return "Classification Error"

def extract_text_from_document(file):
    if isinstance(file, str):
        file_extension = os.path.splitext(file)[1].lower()
        with open(file, 'rb') as f:
            file_content = f.read()
    else:
        file_extension = os.path.splitext(file.name)[1].lower()
        file_content = file.getvalue()

    if file_extension == '.txt':
        return file_content.decode('utf-8')
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_content)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_content)
    elif file_extension == '.pptx':
        return extract_text_from_pptx(file_content)
    elif file_extension in ['.xlsx', '.xls']:
        return extract_text_from_excel(file_content)
    elif file_extension in ['.csv', '.psv']:
        return extract_text_from_csv(file_content, delimiter=',' if file_extension == '.csv' else '|')
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return ""


def extract_text_from_pdf(file_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        return "\n".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""
    

def extract_text_from_docx(file_content):
    try:
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""
    

def extract_text_from_pptx(file_content):
    try:
        prs = Presentation(io.BytesIO(file_content))
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, 'text'):
                    text.append(shape.text)
        return "\n".join(text)
    except Exception as e:
        st.error(f"Error reading PPTX: {str(e)}")
        return ""


def extract_text_from_excel(file_content):
    try:
        wb = load_workbook(io.BytesIO(file_content), read_only=True)
        text = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            for row in ws.iter_rows(values_only=True):
                text.append("\t".join(str(cell) for cell in row if cell is not None))
        return "\n".join(text)
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return ""


def extract_text_from_csv(file_content, delimiter=','):
    try:
        text = []
        csv_data = csv.reader(io.StringIO(file_content.decode('utf-8')), delimiter=delimiter)
        for row in csv_data:
            text.append("\t".join(row))
        return "\n".join(text)
    except Exception as e:
        st.error(f"Error reading CSV/PSV file: {str(e)}")
        return ""

    
def scan_directory(directory, model):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.txt', '.pdf', '.docx', '.doc', 'pptx', 'xlsx', 'xls', 'csv', 'psv')):
                file_path = os.path.join(root, file)
                text = extract_text_from_document(file_path)
                if text:
                    classification = classify_document(text, model)
                    results.append({"Filename": file, "Classification": classification})
    return results

st.set_page_config(page_title="Document Classification System", layout="wide")
lang = st.selectbox("Select Language / Dil Seçin", ["English", "Türkçe"], index=0, key="lang")
st.session_state.lang_code = 'en' if lang == "English" else 'tr'

st.title(t("title"))

installed_models = get_installed_models()
local_models = ["gemma2", "phi3", "llama3", "mistral", "llama3.1", "mistral-nemo", "mertergun/phi3_finetuned", "gemma2", "qwen2"]
all_models = installed_models + [m for m in local_models if m not in installed_models] + ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-turbo", "gpt-4o", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gpt-4o-mini"]

model = st.selectbox(t("model_select"), all_models)
if model not in installed_models and model in local_models:
            if st.button(t("pull_model")):
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
option = st.radio(t("option_radio"), [t("upload_files"), t("scan_directory")])

if option == t("scan_directory"):
    st.write(t("please_select"))
    directory = st.text_input(t("directory_path"), value=".")
    if st.button(t("scan_directory")):
        with st.spinner(f"{t('scanning_directory')} {directory}..."):
            results = scan_directory(directory, model)
        
        if results:
            results_df = pd.DataFrame(results)
            st.subheader(t("classification_results"))
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            fig = px.pie(results_df, names="Classification", title=t('distribution'))
            st.plotly_chart(fig)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label=t("download_results"),
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
            )
        else:
            st.warning(t("no_valid_documents"))

else:
    uploaded_files = st.file_uploader(t("upload_document"), accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'doc', 'pptx', 'xlsx', 'xls', 'csv', 'psv'])

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
                st.success(t(f"completed"))
                st.markdown(f"<h2 style='text-align: center; color: #1E90FF;'>Classification Result</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>File: {results[0]['Filename']}</h3>", unsafe_allow_html=True)
                
                classification = results[0]['Classification']
                color = "#000000" 
                if classification.lower() == "top secret" or classification.lower() == "çok gizli":
                    color = "#FF0000"  
                elif classification.lower() == "secret" or classification.lower() == "gizli":
                    color = "#FFA500"  
                elif classification.lower() == "confidential" or classification.lower() == "hizmete özel":
                    color = "#800080" 
                elif classification.lower() == "restricted" or classification.lower() == "kısıtlı":
                    color = "#0000FF" 
                elif classification.lower() == "unclassified" or classification.lower() == "sınıflandırılmamış":
                    color = "#008000"  
                else: # Means error
                    color = "#000000"
                
                st.markdown(f"<h1 style='text-align: center; color: {color};'>{classification}</h1>", unsafe_allow_html=True)

            else:
                st.subheader(t("classification_results"))
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization
                fig = px.pie(results_df, names="Classification", title=t('distribution'))
                st.plotly_chart(fig)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label=t("download_results"),
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
            )
        else:
            st.warning(t("no_valid_documents"))

st.sidebar.title(t("about"))
st.sidebar.info(
    f"{t('about1')}\n"
    f"{t('about2')}\n"
    f"{t('about3')}\n"
    f"{t('about4')}"
)

st.sidebar.title(t("instructions"))
st.sidebar.write(
    f"{t('instructions1')}\n"
    f"{t('instructions2')}\n"
    f"{t('instructions3')}\n"
    f"{t('instructions4')}\n"
    f"{t('instructions5')}\n"
    f"{t('instructions6')}"
)