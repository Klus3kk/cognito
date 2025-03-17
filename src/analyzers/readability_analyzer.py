from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import warnings
import os
from huggingface_hub import login
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Global variables to track model status
MODEL_LOADED = False
tokenizer = None
model = None

# Try to authenticate with Hugging Face
try:
    # Check if token is in environment variables
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        login(token)
        logger.info("Authenticated with Hugging Face using environment token")
    else:
        logger.warning("No Hugging Face token found in environment variables")
except Exception as e:
    logger.error(f"Error authenticating with Hugging Face: {e}")

# Model loading with better error handling
try:
    # If token is available, try to load models
    if os.environ.get("HUGGINGFACE_TOKEN"):
        # Load tokenizer with authentication options
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base", 
            use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        
        # Load model with the same options
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/codebert-base", 
            use_auth_token=os.environ.get("HUGGINGFACE_TOKEN"),
            num_labels=2
        )
        
        MODEL_LOADED = True
        logger.info("Successfully loaded CodeBERT model and tokenizer")
    else:
        # No token, don't try to load model
        logger.info("Skipping CodeBERT model loading due to missing token")
        MODEL_LOADED = False
        
except Exception as e:
    logger.error(f"Error loading CodeBERT model: {e}")
    MODEL_LOADED = False

def analyze_readability(code_snippet):
    """
    Analyze code readability using CodeBERT if available, or fallback to basic analysis.
    
    Args:
        code_snippet (str): Code to analyze
        
    Returns:
        str: Readability analysis feedback
    """
    if MODEL_LOADED and tokenizer is not None and model is not None:
        try:
            # Use the ML model for analysis
            inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
            if predicted_class == 0:
                return "Consider improving readability: Rename variables, simplify functions, or add comments."
            else:
                return "Code readability looks good."
                
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            # Fall back to basic analysis if model inference fails
            return fallback_readability_analysis(code_snippet)
    else:
        # Use fallback function when model isn't loaded
        logger.info("Using fallback readability analysis (ML model not available)")
        return fallback_readability_analysis(code_snippet)

def fallback_readability_analysis(code_snippet):
    """
    Basic readability analysis without ML models.
    
    Args:
        code_snippet (str): Code to analyze
        
    Returns:
        str: Readability feedback
    """
    # Check line length
    long_lines = 0
    for line in code_snippet.split('\n'):
        if len(line) > 80:
            long_lines += 1
    
    # Check comment frequency
    comment_lines = 0
    for line in code_snippet.split('\n'):
        if line.strip().startswith('#') or line.strip().startswith('//'):
            comment_lines += 1
    
    total_lines = len(code_snippet.split('\n'))
    comment_ratio = comment_lines / max(total_lines, 1)
    
    # Check indentation consistency
    indentation_levels = set()
    for line in code_snippet.split('\n'):
        if line.strip():
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces > 0:
                indentation_levels.add(leading_spaces)
    
    # Generate feedback
    issues = []
    
    if long_lines > 3:
        issues.append(f"{long_lines} lines exceed recommended length (80 chars)")
    
    if comment_ratio < 0.1 and total_lines > 20:
        issues.append("Low comment ratio, consider adding more documentation")
    
    if len(indentation_levels) > 2:
        issues.append("Inconsistent indentation detected")
    
    if not issues:
        return "Code readability looks good."
    else:
        return "Consider improving readability: " + "; ".join(issues)