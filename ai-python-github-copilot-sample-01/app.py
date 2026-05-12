import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import Dataset
import time
import os
import logging

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
