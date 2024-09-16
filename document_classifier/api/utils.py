from document_classifier.api.config import setup
from document_classifier.models.models import InputText, ExplainInputText, PullModel
import asyncio
import aiohttp
import re
from fastapi import HTTPException
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import httpx
from fastapi.responses import StreamingResponse

OLLAMA_HOST, client, genai, system_prompt_en, system_prompt_tr, explain_prompt_en, explain_prompt_tr = setup()

async def query_ollama(session, model, prompt, lowercase=True):
    async with session.post(
        url=f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "keep_alive": -1}
    ) as response:
        result = await response.json()
        if lowercase:
            return result.get("response", "").strip().lower()
        return model, result.get("response", "").strip()

async def classify_document_util(input_text: str, model: str, lang: str):
    system_prompt = system_prompt_en if lang == "en" else system_prompt_tr
    prompt = f"{system_prompt}\n\nDocument content: {input_text}\n\nClassification:"

    if model in ["gemma2", "phi3", "llama3", "mistral", "llama3.1", "mistral-nemo", "mertergun/phi3_finetuned", "qwen2"]:
        async with aiohttp.ClientSession() as session:
            result = await query_ollama(session, model, prompt)
        return result

    elif model in ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
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

async def explain_classification_util(input_text: str, model: str, classification: str, lang: str):
    explain_prompt = explain_prompt_en if lang == "en" else explain_prompt_tr

    prompt = f"{explain_prompt}\n\nDocument content: {input_text}\n\nClassification:{classification}\n\nExplanation:"

    if model in ["gemma2", "phi3", "llama3", "mistral", "llama3.1", "mistral-nemo", "mertergun/phi3_finetuned", "qwen2"]:
        async with aiohttp.ClientSession() as session:
            _, result = await query_ollama(session, model, prompt, lowercase=False)
        return result

    elif model in ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
        response = client.chat.completions.create(model=model,
        messages=[
            {"role": "system", "content": explain_prompt},
            {"role": "user", "content": prompt}
        ])
        return response.choices[0].message.content.strip()

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
        return response.text.strip()

    else:
        raise HTTPException(status_code=400, detail="Unsupported model")

async def classify_document(input_text: InputText):
    try:
        result = await classify_document_util(input_text.text, input_text.model, input_text.lang)
        valid_classifications = ["top secret", "secret", "confidential", "restricted", "unclassified", 
                                 "çok gizli", "gizli", "hizmete özel", "kısıtlı", "sınıflandırılmamış"]
        if result not in valid_classifications:
            result = "unclassified" if input_text.lang == "en" else "sınıflandırılmamış"
        return {"classification": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def explain_classification(input_text: ExplainInputText):
    try:
        result = await explain_classification_util(input_text.text, input_text.model, input_text.classification, input_text.lang)
        return {"explanation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def pull_model_stream(model: PullModel):
    async def stream():
        async with httpx.AsyncClient() as client:
            async with client.stream('POST', f"{OLLAMA_HOST}/api/pull", json={"name": model.model}) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield line + '\n'

    return StreamingResponse(stream(), media_type="text/event-stream")

async def get_ollama_tags():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        return response.json()