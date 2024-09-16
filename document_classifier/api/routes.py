from fastapi import APIRouter
from document_classifier.models.models import InputText, ExplainInputText, PullModel
from document_classifier.api.utils import (
    classify_document, 
    explain_classification, 
    pull_model_stream,
    get_ollama_tags
)

router = APIRouter()

@router.post("/classify")
async def classify(input_text: InputText):
    return await classify_document(input_text)

@router.post("/explain")
async def explain(input_text: ExplainInputText):
    return await explain_classification(input_text)

@router.post("/pull")
async def pull_model(pull_model: PullModel):
    return await pull_model_stream(pull_model)

@router.get("/tags")
async def get_tags():
    return await get_ollama_tags()