import tkinter as tk
from tkinter import messagebox
from tkinter import StringVar
import yaml
import streamlit_authenticator as stauth
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def add_user_to_data(data, username, email, name, password):
    data['credentials']['usernames'][username] = {
        'email': email,
        'name': name,
        'password': password,
        'first_time_login': True
    }
    stauth.Hasher.hash_passwords(data['credentials'])

class UserAdminApp:
    def __init__(self, master):
        self.master = master
        self.master.title("User Administration")
        self.master.geometry("500x400")
        self.file_path = 'credentials.yaml'
        self.data = load_yaml(self.file_path)

        # Choose a theme (e.g., 'flatly', 'superhero', 'cyborg', 'darkly')
        self.style = ttk.Style(theme='flatly')

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="20 20 20 20")
        main_frame.pack(fill=BOTH, expand=True)

        title_label = ttk.Label(
            main_frame, 
            text="Add New User", 
            font=("Helvetica", 20, "bold")
        )
        title_label.pack(pady=(0, 20))

        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=X, expand=True)

        # Username
        username_label = ttk.Label(form_frame, text="Username:")
        username_label.grid(row=0, column=0, sticky=W, pady=10)
        self.username_entry = ttk.Entry(form_frame)
        self.username_entry.grid(row=0, column=1, sticky=EW, pady=10)
        
        # Email
        email_label = ttk.Label(form_frame, text="Email:")
        email_label.grid(row=1, column=0, sticky=W, pady=10)
        self.email_entry = ttk.Entry(form_frame)
        self.email_entry.grid(row=1, column=1, sticky=EW, pady=10)
        
        # Name
        name_label = ttk.Label(form_frame, text="Name:")
        name_label.grid(row=2, column=0, sticky=W, pady=10)
        self.name_entry = ttk.Entry(form_frame)
        self.name_entry.grid(row=2, column=1, sticky=EW, pady=10)
        
        # Password
        password_label = ttk.Label(form_frame, text="Password:")
        password_label.grid(row=3, column=0, sticky=W, pady=10)
        self.password_entry = ttk.Entry(form_frame, show="*")
        self.password_entry.grid(row=3, column=1, sticky=EW, pady=10)
        
        # Configure grid weights
        form_frame.columnconfigure(1, weight=1)

        # Add User Button
        add_button = ttk.Button(
            main_frame, 
            text="Add User", 
            command=self.add_user, 
            style="success.TButton"
        )
        add_button.pack(pady=20)

    def add_user(self):
        username = self.username_entry.get()
        email = self.email_entry.get()
        name = self.name_entry.get()
        password = self.password_entry.get()

        if username and email and name and password:
            add_user_to_data(self.data, username, email, name, password)
            save_yaml(self.file_path, self.data)
            messagebox.showinfo("Success", "User added successfully!")
            self.clear_entries()
        else:
            messagebox.showerror("Error", "Please fill in all fields!")

    def clear_entries(self):
        self.username_entry.delete(0, tk.END)
        self.email_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)

def main():
    root = ttk.Window(themename='flatly')
    # Make window centered
    root.eval('tk::PlaceWindow . center')
    app = UserAdminApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
