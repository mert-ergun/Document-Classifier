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
import datetime
import glob
import shutil

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
        last_time = time.time() - credentials['credentials']['usernames'][username]['last_change'] > 90 * 24 * 60 * 60

    return first_time or last_time

def getUserRole(username):
    with open("./credentials.yaml", "r") as file:
        credentials = yaml.load(file, Loader=SafeLoader)
        return credentials['credentials']['usernames'][username]['role']

translations = get_translations()

def t(key):
    return translations[st.session_state.lang_code].get(key, key)

def main(): 
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "changePassword" not in st.session_state:
        st.session_state.changePassword = False

    if not st.session_state.authenticated or not st.session_state.get("authentication_status"):
        name, authentication_status, username = authenticator.login(key='Login / Giriş Yap', location='main', fields={'Form name':'Login / Giriş Yap', 
                                                                                                                      'Username':'Username / Kullanıcı Adı', 
                                                                                                                      'Password':'Password / Şifre', 
                                                                                                                      'Login':'Login / Giriş Yap'})
        st.session_state.authenticated = False

    if st.session_state.get("authentication_status"):
        if check_first_time_login(st.session_state['username']):
            try:
                st.session_state.changePassword = True
                st.write('Please reset your password / Lütfen şifrenizi sıfırlayın')
                st.write("Your password must be at least 8 characters long, contain at least one uppercase letter, one lowercase letter, one number, and one special character")
                st.write("ifreniz en az 8 karakter uzunluğunda olmalı, en az bir büyük harf, bir küçük harf, bir rakam ve bir özel karakter içermelidir")
                if authenticator.reset_password(st.session_state['username'], fields={'Form name':'Reset password / Şifre sıfırla', 
                                                                                    'Current password':'Current password / Mevcut şifre', 
                                                                                    'New password':'New password / Yeni şifre', 
                                                                                    'Repeat password': 'Repeat password / Şifreyi tekrar girin', 
                                                                                    'Reset':'Reset / Sıfırla'}):
                    st.success('Password modified successfully / Şifre başarıyla değiştirildi')
                    credentials['credentials']['usernames'][st.session_state['username']]['first_time_login'] = False
                    credentials['credentials']['usernames'][st.session_state['username']]['last_change'] = time.time()
                    st.session_state.changePassword = False
                    with open('./credentials.yaml', 'w') as file:
                        yaml.dump(credentials, file, default_flow_style=False)

            except Exception as e:
                st.error(e)

        if not st.session_state.changePassword:
            role = getUserRole(st.session_state['username'])
            if not st.session_state.authenticated:
                st.session_state.authenticated = True
                st.rerun()
            authenticator.logout('Logout / Çıkış Yap', 'main')
            lang = st.selectbox("Select Language / Dil Seçin", ["English", "Türkçe"], index=0, key="lang")
            st.session_state.lang_code = 'en' if lang == "English" else 'tr'

            if role == 'user':
                # Content for user
                if "showresults" not in st.session_state:
                    st.session_state.showresults = False

                if "results" not in st.session_state:
                    st.session_state.results = []

                if "showscan" not in st.session_state:
                    st.session_state.showscan = False

                if "documents" not in st.session_state:
                    st.session_state.documents = []

                if "showsubmissions" not in st.session_state:
                    st.session_state.showsubmissions = False

                st.title(t("title"))

                # Notify user about submission statuses
                def check_submission_status(username):
                    approved_dir = 'approved'
                    disapproved_dir = 'disapproved'
                    user_submissions = []

                    # Check approved submissions
                    if os.path.exists(approved_dir):
                        for eval_dir in os.listdir(approved_dir):
                            if username:
                                if eval_dir.startswith(username):
                                    eval_path = os.path.join(approved_dir, eval_dir)
                                    with open(os.path.join(eval_path, 'metadata.yaml'), 'r') as meta_file:
                                        metadata = yaml.load(meta_file, Loader=SafeLoader)
                                        if metadata.get('notified', False) == False:
                                            user_submissions.append({'status': 'approved', 'path': eval_path, 'metadata': metadata})

                    # Check disapproved submissions
                    if os.path.exists(disapproved_dir):
                        for eval_dir in os.listdir(disapproved_dir):
                            if username:
                                if eval_dir.startswith(username):
                                    eval_path = os.path.join(disapproved_dir, eval_dir)
                                    with open(os.path.join(eval_path, 'metadata.yaml'), 'r') as meta_file:
                                        metadata = yaml.load(meta_file, Loader=SafeLoader)
                                        if metadata.get('notified', False) == False:
                                            user_submissions.append({'status': 'rejected', 'path': eval_path, 'metadata': metadata})

                    return user_submissions

                def notify_user(submissions):
                    for submission in submissions:
                        status = submission['status']
                        eval_path = submission['path']
                        metadata = submission['metadata']

                        if status == 'approved':
                            st.success(f"Your submission '{os.path.basename(eval_path)}' has been approved.")
                            # Mark as notified
                            metadata['notified'] = True
                            with open(os.path.join(eval_path, 'metadata.yaml'), 'w') as meta_file:
                                yaml.dump(metadata, meta_file)
                        elif status == 'rejected':
                            st.error(f"Your submission '{os.path.basename(eval_path)}' has been rejected.")
                            # Display manager's notes if available
                            notes = metadata.get('manager_notes', '')
                            if notes:
                                st.info(f"Manager's Notes: {notes}")

                            # Provide option to reclassify
                            if st.button(f"Reclassify submission '{os.path.basename(eval_path)}'") or st.session_state.showsubmissions:
                                st.session_state.showsubmissions = True
                                # Load the submission files and allow reclassification
                                st.session_state.uploaded_files = []
                                class_file = os.path.join(eval_path, 'classification.csv')
                                if os.path.exists(class_file):
                                    eval_df = pd.read_csv(class_file)
                                    st.subheader("Previous Classification Results")
                                    edited_df = st.data_editor(eval_df, use_container_width=True, column_config={
                                        "Classification": st.column_config.SelectboxColumn(
                                            "Classification",
                                            width="medium",
                                            options=[t("ts"), t("s"), t("c"), t("r"), t("u")],
                                            required=True,
                                        )
                                    }, num_rows="fixed")
                                    # Save reclassified data
                                    if st.button("Submit Reclassified Evaluation"):
                                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                        new_eval_id = f"{st.session_state['username']}_{timestamp}"
                                        new_eval_dir = f"evaluations/{new_eval_id}"
                                        os.makedirs(new_eval_dir, exist_ok=True)
                                        edited_df.to_csv(os.path.join(new_eval_dir, 'classification.csv'), index=False)
                                        # Copy files over
                                        files_in_eval = [f for f in os.listdir(eval_path) if f != 'classification.csv' and f != 'metadata.yaml']
                                        for file in files_in_eval:
                                            shutil.copy(os.path.join(eval_path, file), new_eval_dir)
                                        # Create new metadata
                                        with open(os.path.join(new_eval_dir, 'metadata.yaml'), 'w') as meta_file:
                                            yaml.dump({'status': 'pending', 'user': st.session_state['username']}, meta_file)
                                        st.success('Your reclassified evaluation has been sent to the manager for review.')
                                        st.session_state.showsubmissions = False

                                        # Mark as notified
                                        metadata['notified'] = True
                                        with open(os.path.join(eval_path, 'metadata.yaml'), 'w') as meta_file:
                                            yaml.dump(metadata, meta_file)
                                        st.rerun()

                                else:
                                    st.error('Classification file not found in the rejected submission.')

                # Check for submissions
                user_submissions = check_submission_status(st.session_state['username'])
                if user_submissions:
                    notify_user(user_submissions)

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

                # Set a radio button to select the upload files or scan a directory
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
                            # Make the dataframe editable, so user can change the classification if needed using selectbox in the Classification column
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
                            
                            # Send to manager button
                            if st.button('Send to manager for evaluation'):
                                # Save the edited_df and documents to a directory
                                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                evaluation_dir = f"evaluations/{st.session_state['username']}_{timestamp}"
                                if not os.path.exists(evaluation_dir):
                                    os.makedirs(evaluation_dir)
                                # Save the edited DataFrame
                                edited_df.to_csv(os.path.join(evaluation_dir, 'classification.csv'), index=False)
                                # Save the files
                                for doc in st.session_state.documents:
                                    shutil.copy(doc, evaluation_dir)
                                # Create a metadata file
                                with open(os.path.join(evaluation_dir, 'metadata.yaml'), 'w') as meta_file:
                                    yaml.dump({'status': 'pending', 'user': st.session_state['username']}, meta_file)
                                st.success('Your evaluation has been sent to the manager for review.')

                        else:
                            st.warning(t("no_valid_documents"))

                else:
                    uploaded_files = st.file_uploader(t("upload_document"), accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'doc', 'pptx', 'xlsx', 'xls', 'csv', 'psv'])

                    if st.button(t("upload_files"), key="upload") or st.session_state.showresults:
                        if not st.session_state.showresults or st.session_state.upload:
                            st.session_state.showresults = True
                            st.session_state.results = []
                            st.session_state.uploaded_files = uploaded_files  # Store the uploaded files
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

                                # Send to manager button
                                if st.button('Send to manager for evaluation'):
                                    # Save the results and files
                                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                    evaluation_dir = f"evaluations/{st.session_state['username']}_{timestamp}"
                                    if not os.path.exists(evaluation_dir):
                                        os.makedirs(evaluation_dir)
                                    # Save the results DataFrame
                                    results_df.to_csv(os.path.join(evaluation_dir, 'classification.csv'), index=False)
                                    # Save the files
                                    for i, file in enumerate(st.session_state.uploaded_files):
                                        file_path = os.path.join(evaluation_dir, file.name)
                                        with open(file_path, 'wb') as f:
                                            f.write(file.getbuffer())
                                    # Create a metadata file
                                    with open(os.path.join(evaluation_dir, 'metadata.yaml'), 'w') as meta_file:
                                        yaml.dump({'status': 'pending', 'user': st.session_state['username']}, meta_file)
                                    st.success('Your evaluation has been sent to the manager for review.')

                            else:
                                st.subheader(t("classification_results"))
                                # Make the dataframe editable, so user can change the classification if needed using selectbox in the Classification column
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
                                csv = edited_df.to_csv(index=False)
                                st.download_button(
                                    label=t("download_results"),
                                    data=csv,
                                    file_name="classification_results.csv",
                                    mime="text/csv",
                                )

                                if st.button(t("generate_dataset")):
                                    with st.spinner(t("generating_dataset")):
                                        training_data = generate_training_dataset(edited_df, st.session_state.uploaded_files)
                                        csv = training_data.to_csv(index=False)
                                        st.download_button(
                                            label="Download Train Dataset",
                                            data=csv,
                                            file_name="train_dataset.csv",
                                            mime="text/csv",
                                        )

                                if st.button(t("download_classified_zip")):
                                    zip_file = create_classified_zip(st.session_state.results, st.session_state.uploaded_files)
                                    st.download_button(
                                        label=t("download_classified_documents"),
                                        data=zip_file,
                                        file_name="classifications.zip",
                                        mime="application/zip",
                                    )
                                
                                # Send to manager button
                                if st.button('Send to manager for evaluation'):
                                    # Save the edited_df and files
                                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                    evaluation_id = f"{st.session_state['username']}_{timestamp}"
                                    evaluation_dir = f"evaluations/{evaluation_id}"
                                    if not os.path.exists(evaluation_dir):
                                        os.makedirs(evaluation_dir)
                                    # Save the edited DataFrame
                                    edited_df.to_csv(os.path.join(evaluation_dir, 'classification.csv'), index=False)
                                    # Save the files
                                    for i, file in enumerate(st.session_state.uploaded_files):
                                        file_path = os.path.join(evaluation_dir, file.name)
                                        with open(file_path, 'wb') as f:
                                            f.write(file.getbuffer())
                                    # Create a metadata file
                                    with open(os.path.join(evaluation_dir, 'metadata.yaml'), 'w') as meta_file:
                                        yaml.dump({'status': 'pending', 'user': st.session_state['username']}, meta_file)
                                    st.success('Your evaluation has been sent to the manager for review.')

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

            elif role == 'manager':
                # Content for manager
                st.title('Manager Dashboard')
                # List evaluations submitted by users
                evaluation_dirs = glob.glob('evaluations/*')
                if evaluation_dirs:
                    st.write('Evaluations submitted by users:')
                    # Create a list of display names for the selectbox
                    evaluation_display_names = [os.path.basename(d) for d in evaluation_dirs]
                    evaluation_selection = dict(zip(evaluation_display_names, evaluation_dirs))
                    selected_display_name = st.selectbox('Select an evaluation to review:', evaluation_display_names)
                    selected_dir = evaluation_selection[selected_display_name]
                    # Load the classification DataFrame
                    class_file = os.path.join(selected_dir, 'classification.csv')
                    if os.path.exists(class_file):
                        eval_df = pd.read_csv(class_file)
                        st.write('Evaluation details:')
                        st.dataframe(eval_df)
                        # List files included in the evaluation
                        st.write('Files included in this evaluation:')
                        files_in_eval = [f for f in os.listdir(selected_dir) if f != 'classification.csv' and f != 'metadata.yaml']
                        for file in files_in_eval:
                            file_path = os.path.join(selected_dir, file)
                            with open(file_path, 'rb') as f:
                                file_bytes = f.read()
                            st.download_button(
                                label=f'Download {file}',
                                data=file_bytes,
                                file_name=file,
                            )
                        # Approve or Disapprove buttons with notes
                        st.write('Please add your notes or comments if disapproving the submission:')
                        manager_notes = st.text_area('Manager Notes', '')
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button('Approve', type='primary'):
                                # Update metadata
                                with open(os.path.join(selected_dir, 'metadata.yaml'), 'r') as meta_file:
                                    metadata = yaml.load(meta_file, Loader=SafeLoader)
                                metadata['status'] = 'approved'
                                metadata['manager_notes'] = ''
                                metadata['notified'] = False
                                with open(os.path.join(selected_dir, 'metadata.yaml'), 'w') as meta_file:
                                    yaml.dump(metadata, meta_file)
                                # Move the evaluation directory to the 'approved' directory
                                if not os.path.exists('approved'):
                                    os.makedirs('approved')
                                shutil.move(selected_dir, os.path.join('approved', os.path.basename(selected_dir)))
                                st.success('Evaluation approved.')
                                st.rerun()
                        with col2:
                            if st.button('Disapprove', type='primary'):
                                if not manager_notes.strip():
                                    st.error('Please provide notes explaining why the submission was disapproved.')
                                else:
                                    # Update metadata
                                    with open(os.path.join(selected_dir, 'metadata.yaml'), 'r') as meta_file:
                                        metadata = yaml.load(meta_file, Loader=SafeLoader)
                                    metadata['status'] = 'rejected'
                                    metadata['manager_notes'] = manager_notes
                                    metadata['notified'] = False
                                    with open(os.path.join(selected_dir, 'metadata.yaml'), 'w') as meta_file:
                                        yaml.dump(metadata, meta_file)
                                    # Move the evaluation directory to the 'disapproved' directory
                                    if not os.path.exists('disapproved'):
                                        os.makedirs('disapproved')
                                    shutil.move(selected_dir, os.path.join('disapproved', os.path.basename(selected_dir)))
                                    st.warning('Evaluation disapproved with notes.')
                                    st.rerun()
                    else:
                        st.error('Classification file not found in this evaluation.')
                else:
                    st.write('No evaluations pending for review.')

        elif st.session_state.get("authentication_status") == False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") == None:
            st.warning('Please enter your username and password')

if __name__ == "__main__":
    main()
