import streamlit as st
import os
import pandas as pd
import plotly.express as px
import requests
import json
import time
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from document_classifier.utils.file_utils import (
    create_classified_zip,
    extract_text_from_document,
    scan_directory,
    generate_training_dataset
)

from document_classifier.utils.translations import get_translations
from document_classifier.utils.config import API_URL

from document_classifier.utils.interface_utils import (
    classify_document,
    explain_classification,
    get_installed_models,
)

st.set_page_config(page_title="Document Classification System", layout="wide")  

with open("./credentials.yaml", "r") as file:
    credentials = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials['credentials'],
    credentials['cookie']['name'],
    credentials['cookie']['key'],
    credentials['cookie']['expiry_days'],
    credentials['preauthorized']
)

def check_first_time_login(username):
    if not os.path.exists("./credentials.yaml"):
        return True
    with open("./credentials.yaml", "r") as file:
        credentials = yaml.load(file, Loader=SafeLoader)
        first_time = credentials['credentials']['usernames'][username]['first_time_login']

    return first_time

translations = get_translations()

def t(key):
    return translations[st.session_state.lang_code].get(key, key)

def main(): 
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "changePassword" not in st.session_state:
        st.session_state.changePassword = False

    if not st.session_state.authenticated or not st.session_state["authentication_status"]:
        name, authentication_status, username = authenticator.login(key='Login / Giriş Yap', location='main', fields={'Form name':'Login / Giriş Yap', 
                                                                                                                      'Username':'Username / Kullanıcı Adı', 
                                                                                                                      'Password':'Password / Şifre', 
                                                                                                                      'Login':'Login / Giriş Yap'})
        st.session_state.authenticated = False

    if st.session_state["authentication_status"]:
        if check_first_time_login(st.session_state['username']):
            try:
                st.session_state.changePassword = True
                st.write('Please reset your password / Lütfen şifrenizi sıfırlayın')
                st.write("Your password must be at least 8 characters long, contain at least one uppercase letter, one lowercase letter, one number, and one special character")
                st.write("Şifreniz en az 8 karakter uzunluğunda olmalı, en az bir büyük harf, bir küçük harf, bir rakam ve bir özel karakter içermelidir")
                if authenticator.reset_password(st.session_state['username'], fields={'Form name':'Reset password / Şifre sıfırla', 
                                                                                    'Current password':'Current password / Mevcut şifre', 
                                                                                    'New password':'New password / Yeni şifre', 
                                                                                    'Repeat password': 'Repeat password / Şifreyi tekrar girin', 
                                                                                    'Reset':'Reset / Sıfırla'}):
                    st.success('Password modified successfully / Şifre başarıyla değiştirildi')
                    credentials['credentials']['usernames'][st.session_state['username']]['first_time_login'] = False
                    st.session_state.changePassword = False
                    with open('./credentials.yaml', 'w') as file:
                        yaml.dump(credentials, file, default_flow_style=False)

            except Exception as e:
                st.error(e)

        if not st.session_state.changePassword:
            if not st.session_state.authenticated:
                st.session_state.authenticated = True
                st.rerun()
            authenticator.logout('Logout / Çıkış Yap', 'main')
            lang = st.selectbox("Select Language / Dil Seçin", ["English", "Türkçe"], index=0, key="lang")
            st.session_state.lang_code = 'en' if lang == "English" else 'tr'

            if "showresults" not in st.session_state:
                st.session_state.showresults = False

            if "results" not in st.session_state:
                st.session_state.results = []

            if "showscan" not in st.session_state:
                st.session_state.showscan = False

            if "documents" not in st.session_state:
                st.session_state.documents = []

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
                if st.button(t("scan_directory"), key="scan_directory") or st.session_state.showscan:
                    if not st.session_state.showscan or st.session_state.scan_directory:
                        st.session_state.showscan = True
                        with st.spinner(f"{t('scanning_directory')} {directory}..."):
                            st.session_state.results, st.session_state.documents = scan_directory(directory, model)
                    
                    if st.session_state.results:
                        results_df = pd.DataFrame(st.session_state.results)
                        st.subheader(t("classification_results"))
                        # Make the dataframe editable, so user change the classification if needed using selectbox in the Classification column
                        edited_df = st.data_editor(results_df, use_container_width=True, column_config={
                            "Classification": st.column_config.SelectboxColumn(
                                "Classification",
                                width="medium",
                                options=[t("ts"), t("s"), t("c"), t("r"), t("u")],
                                required=True,
                            )
                        }, num_rows="fixed")
                        
                        # Visualization
                        fig = px.pie(edited_df, names="Classification", title=t('distribution'))
                        st.plotly_chart(fig)
                    
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label=t("download_results"),
                            data=csv,
                            file_name="classification_results.csv",
                            mime="text/csv",
                        )

                        if st.button(t("generate_dataset")):
                            with st.spinner(t("generating_dataset")):
                                training_data = generate_training_dataset(edited_df, st.session_state.documents, scan=True)
                                csv = training_data.to_csv(index=False)
                                st.download_button(
                                    label="Download Train Dataset",
                                    data=csv,
                                    file_name="train_dataset.csv",
                                    mime="text/csv",
                                )

                        if st.button(t("download_classified_zip")):
                            zip_file = create_classified_zip(st.session_state.results, st.session_state.documents)
                            st.download_button(
                                label=t("download_classified_documents"),
                                data=zip_file,
                                file_name="classifications.zip",
                                mime="application/zip",
                            )
                    else:
                        st.warning(t("no_valid_documents"))

            else:
                uploaded_files = st.file_uploader(t("upload_document"), accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'doc', 'pptx', 'xlsx', 'xls', 'csv', 'psv'])

                if st.button(t("upload_files"), key="upload") or st.session_state.showresults:
                    if not st.session_state.showresults or st.session_state.upload:
                        st.session_state.showresults = True
                        st.session_state.results = []
                        for file in uploaded_files:
                            with st.spinner(f'Processing {file.name}...'):
                                text = extract_text_from_document(file)
                                if text:
                                    classification = classify_document(text, model)
                                    st.session_state.results.append({"Filename": file.name, "Classification": classification})
                    
                    if st.session_state.results:
                        results_df = pd.DataFrame(st.session_state.results)
                        
                        if len(st.session_state.results) == 1:
                            st.success(t(f"completed"))
                            st.markdown(f"<h2 style='text-align: center; color: #1E90FF;'>{t('classification_results')}</h2>", unsafe_allow_html=True)
                            st.markdown(f"<h3 style='text-align: center;'>{t('file')}: {st.session_state.results[0]['Filename']}</h3>", unsafe_allow_html=True)
                            
                            classification = st.session_state.results[0]['Classification']
                            color = "#000000" 
                            if classification.lower() == t("ts"):
                                color = "#FF0000"  
                            elif classification.lower() == t("s"):
                                color = "#FFA500"  
                            elif classification.lower() == t("c"):
                                color = "#800080" 
                            elif classification.lower() == t("r"):
                                color = "#0000FF" 
                            elif classification.lower() == t("u"):
                                color = "#008000"  
                            else: # Means error
                                color = "#000000"
                            
                            st.markdown(f"<h1 style='text-align: center; color: {color};'>{classification.upper()}</h1>", unsafe_allow_html=True)

                            if uploaded_files:
                                text = extract_text_from_document(uploaded_files[0])

                                if st.button(t("explain")):
                                    explanation = explain_classification(text, model, classification)
                                    st.write(explanation)

                        else:
                            st.subheader(t("classification_results"))
                            # Make the dataframe editable, so user change the classification if needed using selectbox in the Classification column
                            edited_df = st.data_editor(results_df, use_container_width=True, column_config={
                                "Classification": st.column_config.SelectboxColumn(
                                    "Classification",
                                    width="medium",
                                    options=[t("ts"), t("s"), t("c"), t("r"), t("u")],
                                    required=True,
                                )
                            }, num_rows="fixed")
                            
                            # Visualization
                            fig = px.pie(edited_df, names="Classification", title=t('distribution'))
                            st.plotly_chart(fig)
                        
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label=t("download_results"),
                                data=csv,
                                file_name="classification_results.csv",
                                mime="text/csv",
                            )

                            if st.button(t("generate_dataset")):
                                with st.spinner(t("generating_dataset")):
                                    training_data = generate_training_dataset(edited_df, uploaded_files)
                                    csv = training_data.to_csv(index=False)
                                    st.download_button(
                                        label="Download Train Dataset",
                                        data=csv,
                                        file_name="train_dataset.csv",
                                        mime="text/csv",
                                    )

                            if st.button(t("download_classified_zip")):
                                zip_file = create_classified_zip(st.session_state.results, uploaded_files)
                                st.download_button(
                                    label=t("download_classified_documents"),
                                    data=zip_file,
                                    file_name="classifications.zip",
                                    mime="application/zip",
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

    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')

if __name__ == "__main__":
    main()