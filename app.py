from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the data model for the API request
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    """
    Pydantic model for the response body.
    Returns the predicted 'label' and 'confidence'.
    """
    label: str
    confidence: float

# Initialize the FastAPI app
app = FastAPI(
    title="SvaraAI Reply Classifier",
    description="A simple API for classifying sales email replies.",
    version="1.0.0"
)

try:
    model_path = "./my_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except OSError as e:
    raise RuntimeError(f"Could not load the model from {model_path}. "
                       "Ensure the directory exists and contains the model files.") from e

label_mapping = {
    0: "positive",
    1: "negative",
    2: "neutral"
}

# API Endpoints 
@app.get("/")
def read_root():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"message": "API is running!"}

@app.post("/predict", response_model=PredictionResponse)

def predict(request: PredictionRequest):
  
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class_id].item()
    
    predicted_label = label_mapping[predicted_class_id]

    return {"label": predicted_label, "confidence": confidence}
