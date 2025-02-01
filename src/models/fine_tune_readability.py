from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load CodeBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Load dataset
dataset = load_dataset("csv", data_files="src/data/readability_dataset.csv")

# Tokenize dataset
def preprocess_data(examples):
    return tokenizer(examples["code_snippet"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Load model for fine-tuning
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
save_path = "src/models/fine_tuned_codebert"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Fine-tuned model saved at {save_path}")
