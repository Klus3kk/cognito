"""
HuggingFace Model Trainer for Cognito - Complete Implementation
Trains models and uploads both datasets and models to HuggingFace Hub
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

# HuggingFace imports
from datasets import Dataset, DatasetDict, Features, Value
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, 
    DataCollatorWithPadding, pipeline
)
from huggingface_hub import HfApi, login, create_repo
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Local imports
import sys
sys.path.append('src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training and upload."""
    # Model settings
    model_name: str = "microsoft/codebert-base"
    task_name: str = "code-readability-classification"
    num_labels: int = 5  # Readability scores 1-5
    max_length: int = 512
    
    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # HuggingFace settings
    hub_model_id: str = "cognito-ai/code-readability-classifier"
    hub_dataset_id: str = "cognito-ai/code-readability-dataset"
    hub_organization: str = "cognito-ai"
    private: bool = False
    
    # Local paths
    dataset_path: str = "src/data/readability_dataset.csv"
    model_output_dir: str = "models/cognito-readability"
    cache_dir: str = "cache/huggingface"


class HuggingFaceModelTrainer:
    """Complete HuggingFace model trainer with dataset and model upload."""
    
    def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
        """
        Initialize the HuggingFace trainer.
        
        Args:
            config: Model configuration
            hf_token: HuggingFace token for uploading
        """
        self.config = config
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Initialize HuggingFace API
        if self.hf_token:
            login(token=self.hf_token)
            self.hf_api = HfApi()
        else:
            logger.warning("No HuggingFace token provided. Upload features will be disabled.")
            self.hf_api = None
        
        # Create directories
        os.makedirs(self.config.model_output_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.dataset = None
        
    def load_and_prepare_dataset(self) -> bool:
        """Load dataset and prepare for HuggingFace training."""
        logger.info("Loading and preparing dataset...")
        
        try:
            # Load CSV dataset
            if not os.path.exists(self.config.dataset_path):
                logger.error(f"Dataset not found at {self.config.dataset_path}")
                logger.info("Please run: python src/data/dataset_loader.py first")
                return False
            
            df = pd.read_csv(self.config.dataset_path)
            logger.info(f"Loaded {len(df)} samples from CSV")
            
            # Validate required columns
            required_columns = ['code_snippet', 'readability_score']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Expected: {required_columns}")
                return False
            
            # Clean and prepare data
            df = self._clean_dataset(df)
            
            # Convert to HuggingFace dataset format
            dataset_dict = self._create_huggingface_dataset(df)
            self.dataset = dataset_dict
            
            logger.info(f"Dataset prepared: {len(dataset_dict['train'])} train, {len(dataset_dict['test'])} test samples")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return False
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset."""
        logger.info("Cleaning dataset...")
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=['code_snippet'])
        logger.info(f"Removed {initial_len - len(df)} duplicate samples")
        
        # Clean code snippets
        df['code_snippet'] = df['code_snippet'].apply(self._clean_code_snippet)
        
        # Ensure readability scores are in valid range (1-5)
        df['readability_score'] = df['readability_score'].clip(1, 5)
        df['labels'] = df['readability_score'] - 1  # Convert to 0-4 for classification
        
        # Remove empty or very short code snippets
        df = df[df['code_snippet'].str.len() >= 10]
        
        # Add metadata
        df['text_length'] = df['code_snippet'].str.len()
        df['line_count'] = df['code_snippet'].str.count('\n') + 1
        
        logger.info(f"Final dataset size: {len(df)} samples")
        logger.info(f"Label distribution: {df['readability_score'].value_counts().sort_index().to_dict()}")
        
        return df
    
    def _clean_code_snippet(self, code: str) -> str:
        """Clean individual code snippet."""
        if pd.isna(code):
            return ""
        
        # Convert escaped newlines
        code = str(code).replace('\\n', '\n').replace('\\t', '\t')
        
        # Remove excessive whitespace
        lines = code.split('\n')
        lines = [line.rstrip() for line in lines]
        code = '\n'.join(lines)
        
        # Truncate if too long
        if len(code) > 2000:
            code = code[:2000] + "..."
        
        return code.strip()
    
    def _create_huggingface_dataset(self, df: pd.DataFrame) -> DatasetDict:
        """Convert pandas DataFrame to HuggingFace DatasetDict."""
        logger.info("Converting to HuggingFace dataset format...")
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42,
            stratify=df['labels'] if len(df['labels'].unique()) > 1 else None
        )
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df[['code_snippet', 'labels', 'readability_score']])
        test_dataset = Dataset.from_pandas(test_df[['code_snippet', 'labels', 'readability_score']])
        
        # Define features schema
        features = Features({
            'code_snippet': Value('string'),
            'labels': Value('int32'),
            'readability_score': Value('int32')
        })
        
        train_dataset = train_dataset.cast(features)
        test_dataset = test_dataset.cast(features)
        
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    
    def upload_dataset_to_hub(self) -> bool:
        """Upload dataset to HuggingFace Hub."""
        if not self.hf_api or not self.dataset:
            logger.warning("Cannot upload dataset: missing HF token or dataset")
            return False
        
        try:
            logger.info(f"Uploading dataset to {self.config.hub_dataset_id}...")
            
            # Create repository if it doesn't exist
            try:
                create_repo(
                    repo_id=self.config.hub_dataset_id,
                    token=self.hf_token,
                    repo_type="dataset",
                    private=self.config.private
                )
            except Exception as e:
                if "already exists" not in str(e):
                    logger.warning(f"Repo creation warning: {e}")
            
            # Upload dataset
            self.dataset.push_to_hub(
                self.config.hub_dataset_id,
                token=self.hf_token,
                private=self.config.private
            )
            
            # Create dataset card
            self._create_dataset_card()
            
            logger.info(f"✅ Dataset uploaded successfully to: https://huggingface.co/datasets/{self.config.hub_dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading dataset: {e}")
            return False
    
    def _create_dataset_card(self):
        """Create a dataset card (README) for the uploaded dataset."""
        card_content = f"""---
license: mit
task_categories:
- text-classification
language:
- code
tags:
- code-analysis
- readability
- software-engineering
- python
pretty_name: Code Readability Classification Dataset
size_categories:
- 1K<n<10K
---

# Code Readability Classification Dataset

## Dataset Description

This dataset contains code snippets labeled with readability scores for training code analysis models.

### Dataset Summary

- **Task**: Code readability classification
- **Language**: Primarily Python code snippets
- **Size**: {len(self.dataset['train']) + len(self.dataset['test'])} samples
- **Labels**: 5-class classification (1-5 readability score)

### Data Fields

- `code_snippet`: The source code text
- `labels`: Classification label (0-4, representing readability scores 1-5)
- `readability_score`: Original readability score (1-5)

### Data Splits

| Split | Size |
|-------|------|
| Train | {len(self.dataset['train'])} |
| Test  | {len(self.dataset['test'])} |

### Usage

```python
from datasets import load_dataset

dataset = load_dataset("{self.config.hub_dataset_id}")
```

### Citation

If you use this dataset, please cite:

```
@misc{{cognito-readability-dataset,
  author = {{Cognito AI Team}},
  title = {{Code Readability Classification Dataset}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{self.config.hub_dataset_id}}}}}
}}
```

### License

MIT License - See LICENSE file for details.
"""
        
        # Save dataset card
        card_path = os.path.join(self.config.model_output_dir, "dataset_card.md")
        with open(card_path, 'w') as f:
            f.write(card_content)
        
        # Upload to hub
        try:
            self.hf_api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=self.config.hub_dataset_id,
                repo_type="dataset",
                token=self.hf_token
            )
        except Exception as e:
            logger.warning(f"Could not upload dataset card: {e}")
    
    def initialize_model(self) -> bool:
        """Initialize tokenizer and model."""
        try:
            logger.info(f"Initializing model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
                cache_dir=self.config.cache_dir
            )
            
            logger.info("✅ Model and tokenizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False
    
    def tokenize_dataset(self) -> bool:
        """Tokenize the dataset for training."""
        if not self.dataset or not self.tokenizer:
            logger.error("Dataset or tokenizer not available")
            return False
        
        try:
            logger.info("Tokenizing dataset...")
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['code_snippet'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
            
            # Tokenize datasets
            self.dataset = self.dataset.map(
                tokenize_function,
                batched=True,
                desc="Tokenizing dataset"
            )
            
            # Set format for PyTorch
            self.dataset.set_format(
                type='torch',
                columns=['input_ids', 'attention_mask', 'labels']
            )
            
            logger.info("✅ Dataset tokenized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error tokenizing dataset: {e}")
            return False
    
    def train_model(self) -> bool:
        """Train the model using HuggingFace Trainer."""
        if not self.model or not self.dataset:
            logger.error("Model or dataset not available")
            return False
        
        try:
            logger.info("Starting model training...")
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.model_output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                logging_dir=f"{self.config.model_output_dir}/logs",
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
                greater_is_better=True,
                push_to_hub=bool(self.hf_token),
                hub_model_id=self.config.hub_model_id if self.hf_token else None,
                hub_token=self.hf_token,
                report_to=["tensorboard"],
                dataloader_pin_memory=False  # Avoid memory issues
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['test'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics
            )
            
            # Train model
            logger.info("🚀 Starting training...")
            train_result = self.trainer.train()
            
            # Log training results
            logger.info(f"✅ Training completed!")
            logger.info(f"📊 Training loss: {train_result.training_loss:.4f}")
            
            # Evaluate model
            logger.info("📊 Evaluating model...")
            eval_result = self.trainer.evaluate()
            logger.info(f"📊 Evaluation results: {eval_result}")
            
            # Save model locally
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.model_output_dir)
            
            logger.info(f"💾 Model saved to: {self.config.model_output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def upload_model_to_hub(self) -> bool:
        """Upload trained model to HuggingFace Hub."""
        if not self.hf_api or not self.trainer:
            logger.warning("Cannot upload model: missing HF token or trained model")
            return False
        
        try:
            logger.info(f"Uploading model to {self.config.hub_model_id}...")
            
            # Push to hub
            self.trainer.push_to_hub(
                commit_message="Upload Cognito code readability model"
            )
            
            # Create model card
            self._create_model_card()
            
            logger.info(f"✅ Model uploaded successfully to: https://huggingface.co/models/{self.config.hub_model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return False
    
    def _create_model_card(self):
        """Create a model card for the uploaded model."""
        card_content = f"""---
license: mit
base_model: {self.config.model_name}
tags:
- code-analysis
- readability
- classification
- software-engineering
- cognito
widget:
- text: "def calculate_sum(a, b): return a + b"
  example_title: "Simple Function"
- text: "def f(x,y): return x*y"
  example_title: "Poor Naming"
---

# Cognito Code Readability Classifier

## Model Description

This model classifies code snippets by readability score (1-5) using a fine-tuned CodeBERT architecture.

### Model Details

- **Base Model**: {self.config.model_name}
- **Task**: Multi-class classification (5 classes)
- **Languages**: Primarily Python code
- **License**: MIT

### Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="{self.config.hub_model_id}"
)

code = "def calculate_sum(a, b): return a + b"
result = classifier(code)
print(result)
```

### Training Data

The model was trained on the [{self.config.hub_dataset_id}](https://huggingface.co/datasets/{self.config.hub_dataset_id}) dataset.

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| F1-Score | TBD |

### Citation

```
@misc{{cognito-readability-classifier,
  author = {{Cognito AI Team}},
  title = {{Code Readability Classification Model}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/models/{self.config.hub_model_id}}}}}
}}
```
"""
        
        # Save model card
        card_path = os.path.join(self.config.model_output_dir, "model_card.md")
        with open(card_path, 'w') as f:
            f.write(card_content)
        
        # Upload to hub
        try:
            self.hf_api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=self.config.hub_model_id,
                repo_type="model",
                token=self.hf_token
            )
        except Exception as e:
            logger.warning(f"Could not upload model card: {e}")
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete training and upload pipeline."""
        logger.info("🚀 Starting complete HuggingFace pipeline...")
        
        # Step 1: Load and prepare dataset
        if not self.load_and_prepare_dataset():
            return False
        
        # Step 2: Upload dataset to HF Hub
        if self.hf_token:
            self.upload_dataset_to_hub()
        
        # Step 3: Initialize model
        if not self.initialize_model():
            return False
        
        # Step 4: Tokenize dataset
        if not self.tokenize_dataset():
            return False
        
        # Step 5: Train model
        if not self.train_model():
            return False
        
        # Step 6: Upload model to HF Hub
        if self.hf_token:
            self.upload_model_to_hub()
        
        logger.info("🎉 Complete pipeline finished successfully!")
        return True


def main():
    """Main function to run the HuggingFace training pipeline."""
    print("=" * 80)
    print("🤖 COGNITO HUGGINGFACE MODEL TRAINER")
    print("=" * 80)
    
    # Get HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        hf_token = input("Enter your HuggingFace token (or press Enter to skip upload): ").strip()
        if not hf_token:
            print("⚠️  No HuggingFace token provided. Models will be trained locally only.")
    
    # Configuration
    config = ModelConfig()
    
    # Ask user preferences
    print(f"\n📋 Current configuration:")
    print(f"   Dataset: {config.dataset_path}")
    print(f"   Base model: {config.model_name}")
    print(f"   HF model ID: {config.hub_model_id}")
    print(f"   HF dataset ID: {config.hub_dataset_id}")
    print(f"   Training epochs: {config.num_train_epochs}")
    
    proceed = input(f"\nProceed with training? [Y/n]: ").lower()
    if proceed.startswith('n'):
        print("Training cancelled.")
        return
    
    # Initialize trainer
    trainer = HuggingFaceModelTrainer(config, hf_token)
    
    # Run complete pipeline
    success = trainer.run_complete_pipeline()
    
    if success:
        print(f"\n🎉 SUCCESS! Model training completed!")
        print(f"📁 Local model: {config.model_output_dir}")
        if hf_token:
            print(f"🤗 HuggingFace model: https://huggingface.co/{config.hub_model_id}")
            print(f"🗂️  HuggingFace dataset: https://huggingface.co/datasets/{config.hub_dataset_id}")
        print(f"\n🔬 Next steps:")
        print(f"   1. Test model: python src/models/test_huggingface_model.py")
        print(f"   2. Integrate with Cognito: Update readability_analyzer.py")
    else:
        print(f"\n❌ Training failed. Check logs for details.")


if __name__ == "__main__":
    main()