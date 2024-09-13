from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import httpx
import asyncio
import aiohttp

app = FastAPI()
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(api_key=openai_api_key)
genai.configure(api_key=gemini_api_key)

with open("system_prompt_en.txt", "r") as f:
    system_prompt_en = f.read()

with open("system_prompt_tr.txt", "r") as f:
    system_prompt_tr = f.read()

with open("explain_prompt_en.txt", "r") as f:
    explain_prompt_en = f.read()

with open("explain_prompt_tr.txt", "r") as f:
    explain_prompt_tr = f.read()

class InputText(BaseModel):
    text: str
    model: str = "gemma2"
    lang: str = "en"

class ExplainInputText(InputText):
    classification: str

class PullModel(BaseModel):
    model: str


async def query_ollama(session, model, prompt):
    await asyncio.sleep(0.5)
    async with session.post(
        url=f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "keep_alive":-1}
    ) as response:
        result = await response.json()
        return model, result.get("response").strip().lower()

async def classify_document(input_text: str, model: str, lang: str):
    system_prompt = system_prompt_en if lang == "en" else system_prompt_tr

    prompt = f"{system_prompt}\n\nDocument content: {input_text}\n\nClassification:"

    if model in ["gemma2", "phi3", "llama3", "mistral", "llama3.1", "mistral-nemo", "mertergun/phi3_finetuned", "gemma2", "qwen2"]:
        async with aiohttp.ClientSession() as session:
            _, result = await query_ollama(session, model, prompt)
        return result

    elif model in ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
        response = client.chat.completions.create(model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])
        return response.choices[0].message.content.strip().lower()

    elif model in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]:
        model = genai.GenerativeModel(model)
        response = model.generate_content(
            prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        return response.text.strip().lower()

    else:
        raise HTTPException(status_code=400, detail="Unsupported model")
    

async def explain_classification(input_text: str, model: str, classification: str,lang: str):
    explain_prompt = explain_prompt_en if lang == "en" else explain_prompt_tr

    prompt = f"{explain_prompt}\n\nDocument content: {input_text}\n\nClassification:{classification}\n\nExplanation:"

    if model in ["gemma2", "phi3", "llama3", "mistral", "llama3.1", "mistral-nemo", "mertergun/phi3_finetuned", "gemma2", "qwen2"]:
        async with aiohttp.ClientSession() as session:
            _, result = await query_ollama(session, model, prompt)
        return result

    elif model in ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
        response = client.chat.completions.create(model=model,
        messages=[
            {"role": "system", "content": explain_prompt},
            {"role": "user", "content": prompt}
        ])
        return response.choices[0].message.content.strip().lower()

    elif model in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]:
        model = genai.GenerativeModel(model)
        response = model.generate_content(
            prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        return response.text.strip().lower()

    else:
        raise HTTPException(status_code=400, detail="Unsupported model")


@app.post("/classify")
async def classify_document_api(input_text: InputText):
    try:
        result = await classify_document(input_text.text, input_text.model, input_text.lang)
        print(result)
        if result not in ["top secret", "secret", "confidential", "restricted", "unclassified"] and result not in ["çok gizli", "gizli", "hizmete özel", "kısıtlı", "sınıflandırılmamış"]:
            result = "ERROR!!"  # Default to unclassified if the model output is unexpected
            print("Defaulting to unclassified")
        return {"classification": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/explain")
async def explain_classification_api(input_text: InputText):
    try:
        result = await explain_classification(input_text.text, input_text.model, input_text.classification, input_text.lang)
        return {"explanation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pull_model")
async def pull_model(model: PullModel):
    async def pull_model_stream():
        async with httpx.AsyncClient() as client:
            async with client.stream('POST', f"{OLLAMA_HOST}/api/pull", json={"name": model.model}) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield line + '\n'

    return StreamingResponse(pull_model_stream(), media_type="text/event-stream")

@app.get("/tags")
async def get_tags():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        return response.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)