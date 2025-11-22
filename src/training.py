import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import evaluate

DATASET_PATH = "data/URL_dataset.csv"
OUTPUT_PATH = "models/fine_tuned_model"
MODEL_NAME = "distilbert-base-uncased"

df = pd.read_csv(DATASET_PATH)
print("-----------------Dataset Info-----------------")
print(f"Rows: {len(df)}")
print(df["type"].value_counts())

train_data, valid_test_data = train_test_split(df, test_size=0.3, stratify=df["type"], random_state=42)
validation_data, test_data = train_test_split(valid_test_data, test_size=0.333, stratify=valid_test_data["type"], random_state=42)

train_dataset = Dataset.from_pandas(train_data)
validation_dataset = Dataset.from_pandas(validation_data)
test_dataset = Dataset.from_pandas(test_data)

print("\nThe data is now split into training, validation, and testing")
print(f"Training Length: {len(train_dataset)}, Validation Length: {len(validation_data)}, Testing Length: {len(test_dataset)}")


dataset = DatasetDict({
    "train" : train_dataset,
    "validation" : validation_dataset,
    "test" : test_dataset
    })

type_to_id = {"legitimate": 0, "phishing": 1}
num_types = len(type_to_id)

def num_labels(sample):
    sample["label"] = type_to_id[sample["type"]]
    return sample

dataset = dataset.map(num_labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_data(batch):
    tokenized = tokenizer(
        batch["url"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized["labels"] = batch["label"]
    return tokenized

dataset = dataset.map(tokenize_data, batched=True)
print("\nTokenization is complete")

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, correct_type = eval_pred
    predicted_type = np.argmax(logits, axis=-1)

    accuracy_metric = accuracy.compute(predictions=predicted_type, references=correct_type)["accuracy"]
    f1_metric = f1.compute(predictions=predicted_type, references=correct_type, average="macro")["f1"]
    return {"accuracy" : accuracy_metric, "f1" : f1_metric}

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_arguments = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"\nModel saved to {OUTPUT_PATH}")


