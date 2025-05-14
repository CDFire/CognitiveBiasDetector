import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import transformers 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from bias_definitions import label2id, id2label, CLASSIFIER_LABELS

# --- Configuration ---
MODEL_NAME = "bert-base-uncased"
DATA_FILE = "generated_bias_data.csv"
OUTPUT_DIR = "./fine_tuned_bert_model"
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 8 # **** BERT is large, potentially needs smaller batch size ****
TEST_SIZE = 0.15

# --- Helper Functions (load_and_prepare_data, tokenize_function, compute_metrics) ---
def load_and_prepare_data(data_file):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}. Run synthetic_data_generator.py")
    df = pd.read_csv(data_file)
    df.dropna(subset=['text', 'label'], inplace=True)
    df = df[df['label'].isin(CLASSIFIER_LABELS)]
    if df['label'].nunique() < 2:
        raise ValueError("Need at least two distinct classes in the data to train.")
    df['label'] = df['label'].map(label2id)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    ds = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })
    return ds

def tokenize_function(examples, tokenizer):
    # max_length=128 might be short for some BERT use cases
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
# --- Main Training Logic ---
def train_model():
    print(transformers.__version__)
    """Loads data, tokenizer, BERT model, and trains the classifier."""
    print("Starting BERT model training...")

    # Load and Prepare Data
    try:
        dataset = load_and_prepare_data(DATA_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return

    # Load Tokenizer and BERT model
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSIFIER_LABELS),
        id2label=id2label,
        label2id=label2id
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Tokenize Data
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"]) # Remove pandas index if present
    tokenized_datasets.set_format("torch")

    # Define Training Arguments
    print("Setting up training arguments...")
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Evaluate on Test Set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("\n--- Test Set Evaluation Results ---")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    print("-----------------------------------\n")

    # Save Model and Tokenizer
    print(f"Saving fine-tuned BERT model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    train_model()
