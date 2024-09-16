import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

load_dotenv()

def setup():
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")

    client = OpenAI(api_key=openai_api_key)
    genai.configure(api_key=gemini_api_key)

    with open("prompts/system_prompt_en.txt", "r") as f:
        system_prompt_en = f.read()

    with open("prompts/system_prompt_tr.txt", "r") as f:
        system_prompt_tr = f.read()

    with open("prompts/explain_prompt_en.txt", "r") as f:
        explain_prompt_en = f.read()

    with open("prompts/explain_prompt_tr.txt", "r") as f:
        explain_prompt_tr = f.read()

    return OLLAMA_HOST, client, genai, system_prompt_en, system_prompt_tr, explain_prompt_en, explain_prompt_tr

