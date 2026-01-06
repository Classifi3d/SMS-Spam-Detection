!pip install transformers datasets scikit-learn pandas matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Data upload
url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
df = pd.read_csv(url, encoding='latin-1')

# Curatare dataset
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Conversie etichete ham = 0 spam = 1
df['label_id'] = df['label'].map({'ham': 0, 'spam': 1})

print("Distribuția datelor:")
print(df['label'].value_counts())

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_id'], test_size=0.2, random_state=42
)

# Vectorizare
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Antrenare Baseline (Naive Bayes)
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Baseline evaluation
predictions = nb_model.predict(X_test_tfidf)
print("\n--- Raport de Performanță (Baseline: Naive Bayes) ---")
print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))

# Vizualizare Matrice de Confuzie
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Baseline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Pregatirea Tokenizer-ului
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Formatarea datelor pentru PyTorch/DistilBERT
class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenizarea datelor
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

# Crearea dataset-ului
train_dataset = SMSDataset(train_encodings, list(y_train))
test_dataset = SMSDataset(test_encodings, list(y_test))

# Incarcarea Modelului preantrenat
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Calcularea metricilor in timpul antrenarii
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Configurare Antrenament (Trainer API)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Antrenare
print("Incepe antrenarea DistilBERT...")
trainer.train()

# Evaluare Finala DistilBERT
print("\n--- Rezultate Finale DistilBERT ---")
results = trainer.evaluate()
print(results)

def predict_spam(text):
    # Pregatim textul
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Mutam inputul pe masina locala
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predictie
    with torch.no_grad():
        logits = model(**inputs).logits

    # Interpretare
    predicted_class_id = logits.argmax().item()
    label = "SPAM" if predicted_class_id == 1 else "HAM (Mesaj legitim)"

    return label


# Testing
text_mesaj = "Congratulations! You have won a $1000 Walmart gift card. Call to claim now."
rezultat = predict_spam(text_mesaj)

print(f"Mesaj: '{text_mesaj}'")
print(f"Predicție Model: ** {rezultat} **")