from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_readability_model():
    """Load the fine-tuned CodeBERT model for readability analysis."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")
    return tokenizer, model
