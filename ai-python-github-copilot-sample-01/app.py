import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, IntervalStrategy
from transformers import DistilBertTokenizerFast, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
import time
import os
import logging
from transformers.trainer_utils import SaveStrategy

# Basic console logging for script-style execution.
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Part 1: Building an MVP from Scratch
# ---------------------------------------

logger.info("# Part 1: Building an MVP from Scratch")

# Create a small dataset for demonstration
texts = [
    "I love this product, it's amazing!",
    "This is the best purchase I've made all year",
    "Highly recommend this to everyone",
    "Works exactly as described and exceeded expectations",
    "Very happy with my purchase",
    "This product is terrible, avoid at all costs",
    "Waste of money, doesn't work properly",
    "Disappointed with the quality, would not buy again",
    "Broke after two weeks of use",
    "Customer service was unhelpful when I had issues"
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative

# Create a DataFrame
df = pd.DataFrame({"text": texts, "label": labels})
logger.info("Dataset Overview:\n%s", df)

# Split the data
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
logger.info(f"Training set: {len(train_df)} samples")
logger.info(f"Test set: {len(test_df)} samples")
logger.info("Dataset Train Overview:\n%s", train_df)
logger.info("Dataset Test Overview:\n%s", test_df)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
logger.info("Hugging Face Dataset Train Overview:\n%s", train_dataset)

# Initialize tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


# Apply tokenization
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define training arguments - explicitly disable wandb
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     weight_decay=0.01,
#     #    evaluation_strategy=IntervalStrategy.EPOCH, #"epoch",
#     #    save_strategy=SaveStrategy.EPOCH, #"epoch",
#     #    save_strategy="best",
#     load_best_model_at_end=True,
#     report_to="none",  # Disable all integrations including wandb
#     logging_dir="./logs",
#     logging_strategy="epoch"
# )

# Define the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_test,
#     tokenizer=tokenizer,
# )

# Train the model
# print("\nTraining the model...")
# trainer.train()

# Evaluate the model
# print("\nEvaluating the model...")
# eval_results = trainer.evaluate()
# print(f"Evaluation results: {eval_results}")

# Make predictions on test set
# predictions = trainer.predict(tokenized_test)
# preds = np.argmax(predictions.predictions, axis=-1)

# Calculate accuracy
# accuracy = accuracy_score(test_df["label"].values, preds)
# print(f"\nAccuracy on test set: {accuracy:.4f}")

# Classification report
# print("\nClassification Report:")
# print(classification_report(test_df["label"].values, preds))

# Save the model for later use
# output_dir = "./saved_model"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)
# print(f"\nModel saved to {output_dir}")
