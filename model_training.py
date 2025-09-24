import pandas as pd
# Reading csv
df = pd.read_csv("reply_classification_dataset.csv")
df.head(10)

# drop rows with missing values
df.dropna(subset=['reply', 'label'], inplace=True)

print("\nDataFrame after handling missing values:")
print(df.info())

import re

def clean_text(text):
    text = str(text).lower()

    #[^a-z\s] character that is NOT a lowercase letter or a whitespace character.
    text = re.sub(r'[^a-z\s]', '', text)
    
    #Remove extra whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


df['reply'] = df['reply'].apply(clean_text)
df['label'] = df['label'].str.lower()


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X = df['reply'] 
y = df['label']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


#removing duplicate rows
initial_rows = len(df)
df.drop_duplicates(subset=['reply', 'label'], inplace=True)
final_rows = len(df)

print(f"Removed {initial_rows - final_rows} duplicate rows.")
print(f"Remaining rows in the dataset: {final_rows}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

#Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)

#predictions
y_pred_logistic = logistic_model.predict(X_test_tfidf)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic, average='weighted')

print(f"Logistic Regression Model Accuracy: {accuracy_logistic:.4f}")
print(f"Logistic Regression Model F1 Score: {f1_logistic:.4f}")

def predict_sentiment(text):
    preprocessed_text = clean_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prediction = logistic_model.predict(text_tfidf)
    return prediction[0]

# Assuming your clean_text() function and your models are in the environmen

# Example 1: 
reply1 = "This looks great, I'm ready to move forward."
print(f"Reply: '{reply1}' -> Predicted Label: {predict_sentiment(reply1)}")

# Example 2: 
reply2 = "Thanks for your email, but we're not interested at this time."
print(f"Reply: '{reply2}' -> Predicted Label: {predict_sentiment(reply2)}")

# Example 3: 
reply3 = "Could you send me some more details about your pricing plans?"
print(f"Reply: '{reply3}' -> Predicted Label: {predict_sentiment(reply3)}")

# Example 4: 
reply4 = "The demo was good, but I need to consult with my team."
print(f"Reply: '{reply4}' -> Predicted Label: {predict_sentiment(reply4)}")

import pandas as pd
from datasets import Dataset, ClassLabel, Features, Value

label_mapping = {"positive": 0, "negative": 1, "neutral": 2}
df["label_int"] = df["label"].map(label_mapping)

#Hugging Face Dataset
dataset = Dataset.from_pandas(df)

dataset = dataset.cast_column(
    "label_int", ClassLabel(names=["positive", "negative", "neutral"])
)

dataset = dataset.rename_column("reply", "text")
dataset = dataset.remove_columns(["label"])
dataset = dataset.rename_column("label_int", "label")

# Split the dataset 
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

from transformers import AutoTokenizer

#DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy and f1 score
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {"accuracy": accuracy, "f1": f1}

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",          
    eval_strategy="epoch",     
    learning_rate=2e-5,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    num_train_epochs=3,              
    weight_decay=0.01,               
    report_to="none",                
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=tokenized_train_dataset, 
    eval_dataset=tokenized_test_dataset,   
    compute_metrics=compute_metrics,     
)


trainer.train()

# Evaluate the fine-tuned model on the test set
transformer_results = trainer.evaluate()

# Print the results
print("Transformer Model Evaluation Results:")
print(f"Accuracy: {transformer_results['eval_accuracy']:.4f}")
print(f"F1 Score: {transformer_results['eval_f1']:.4f}")
