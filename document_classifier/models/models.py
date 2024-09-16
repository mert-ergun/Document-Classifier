from pydantic import BaseModel

class InputText(BaseModel):
    text: str
    model: str = "gemma2"
    lang: str = "en"

class ExplainInputText(InputText):
    classification: str

class PullModel(BaseModel):
    model: str
