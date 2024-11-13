from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 

# Loading BERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

def analyze_readability(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 0:
        return "Consider improving readability: Rename variables, simplify functions, or add comments."
    else:
        return "Code readability looks good."
