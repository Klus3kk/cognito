import warnings
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Global variables to track model status
MODEL_LOADED = False
ml_model = None

def load_ml_readability_model():
    """Load the trained ML readability model."""
    global MODEL_LOADED, ml_model
    
    try:
        # Import the trainer class
        import sys
        sys.path.append('src/models')
        from fine_tune_readability import ReadabilityModelTrainer
        
        ml_model = ReadabilityModelTrainer()
        if ml_model.load_model():
            MODEL_LOADED = True
            logger.info("Successfully loaded ML readability model")
            return True
        else:
            logger.warning("Failed to load ML readability model")
            return False
            
    except Exception as e:
        logger.error(f"Error loading ML readability model: {e}")
        return False

def analyze_readability(code_snippet):
    """
    Analyze code readability using ML model if available, or fallback to basic analysis.
    
    Args:
        code_snippet (str): Code to analyze
        
    Returns:
        str: Readability analysis feedback
    """
    global MODEL_LOADED, ml_model
    
    # Try to use ML model first
    if not MODEL_LOADED:
        load_ml_readability_model()
    
    if MODEL_LOADED and ml_model is not None:
        try:
            # Use the ML model for analysis
            score, explanation = ml_model.predict_readability(code_snippet)
            
            if score is not None:
                return explanation
            else:
                logger.warning("ML model returned None, falling back to heuristics")
                
        except Exception as e:
            logger.error(f"Error during ML model inference: {e}")
            # Fall back to basic analysis if model inference fails
    
    # Use fallback function when model isn't available
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
    
    # FIXED: Better indentation consistency check
    indentation_levels = []
    lines_with_content = []
    
    for line in code_snippet.split('\n'):
        if line.strip():  # Only check non-empty lines
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces > 0:  # Only count indented lines
                indentation_levels.append(leading_spaces)
                lines_with_content.append(line)
    
    # Check for mixed tabs and spaces (real inconsistency)
    has_mixed_whitespace = False
    for line in lines_with_content:
        leading_whitespace = line[:len(line) - len(line.lstrip())]
        if '\t' in leading_whitespace and ' ' in leading_whitespace:
            has_mixed_whitespace = True
            break
    
    # Check if indentation levels follow a reasonable pattern
    is_inconsistent_indentation = False
    if indentation_levels:
        unique_levels = sorted(set(indentation_levels))
        
        # Only flag if we have more than 3 different indentation levels
        # and they don't follow a consistent pattern (multiples of 2 or 4)
        if len(unique_levels) > 3:
            # Check for 4-space pattern (most common in Python)
            if not all(level % 4 == 0 for level in unique_levels):
                # Check for 2-space pattern
                if not all(level % 2 == 0 for level in unique_levels):
                    is_inconsistent_indentation = True
    
    # Generate feedback
    issues = []
    
    if long_lines > 3:
        issues.append(f"{long_lines} lines exceed recommended length (80 chars)")
    
    if comment_ratio < 0.1 and total_lines > 20:
        issues.append("Low comment ratio, consider adding more documentation")
    
    # FIXED: Only flag real indentation issues
    if has_mixed_whitespace:
        issues.append("Mixed tabs and spaces detected - use consistent whitespace")
    elif is_inconsistent_indentation:
        issues.append("Inconsistent indentation pattern detected")
    
    # Check for documentation
    if '"""' in code_snippet or "'''" in code_snippet:
        issues = [issue for issue in issues if issue]  # Keep other issues but note good docs
        if not issues:
            return "Code readability looks good."
    
    # Check for good naming
    import re
    good_names = len(re.findall(r'def\s+[a-z][a-z0-9_]*\s*\(', code_snippet))
    poor_names = len(re.findall(r'def\s+[a-zA-Z]\s*\(', code_snippet)) - good_names
    
    if poor_names > good_names:
        issues.append("Consider using more descriptive function names")
    
    if not issues:
        return "Code readability looks good."
    else:
        return "Consider improving readability: " + "; ".join(issues)

# Maintain backwards compatibility
def analyze_readability_detailed(code_snippet):
    """
    Detailed readability analysis (for backwards compatibility).
    
    Args:
        code_snippet (str): Code to analyze
        
    Returns:
        dict: Detailed analysis results
    """
    result = analyze_readability(code_snippet)
    
    # Determine score based on result
    if "looks good" in result.lower():
        score = 4
    elif "excellent" in result.lower():
        score = 5
    elif "poor" in result.lower() or "improve" in result.lower():
        score = 2
    else:
        score = 3
    
    return {
        'score': score,
        'feedback': result,
        'model_used': 'ML' if MODEL_LOADED else 'heuristic'
    }