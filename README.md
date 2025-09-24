# SvaraAI Reply Classification Project
This repository contains the code and documentation for the SvaraAI reply classification project. The goal of this assignment was to design, implement, and deploy a simple NLP pipeline to classify email replies from sales prospects.

# Project Deliverables
notebook.ipynb: Contains the complete code for the ML/NLP pipeline (Part A), including data preprocessing, baseline model training, and fine-tuning a Transformer model.

app.py: The FastAPI service for the deployment task (Part B), which exposes a /predict endpoint for sentiment classification.

answers.md: A markdown file with the short answers for the reasoning questions (Part C).

README.md: This file, which provides an overview and instructions.

my_model/: The directory containing the fine-tuned DistilBERT model and its tokenizer, which is loaded by the API.

# Part A: ML/NLP Pipeline
The pipeline involved the following steps:

Data Preprocessing: Cleaning the text and handling duplicates.

Baseline Model: Training a Logistic Regression model with TF-IDF features to establish a benchmark.

Transformer Model: Fine-tuning a distilbert-base-uncased model for superior performance on the classification task.

$ Part B: API Deployment
The API is built with FastAPI and provides a single endpoint for predictions.

Running the API Locally
Install Dependencies: Install the required libraries using the provided requirements.txt file (or install them manually).

pip install -r requirements.txt

Run the Server: Navigate to the project directory in your terminal and run the following command:

uvicorn app:app --reload


# Part C: Reasoning Answers
The detailed answers to the reasoning questions are provided in the answers.md file.

Bonus: Requirements File
The requirements.txt file contains all the necessary libraries to run the project.

fastapi
pydantic
uvicorn
torch
transformers
datasets
scikit-learn
pandas
