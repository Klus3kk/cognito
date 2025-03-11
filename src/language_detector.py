import re
from collections import Counter
import os

class LanguageDetector:
    """Detects the programming language of code snippets."""
    
    def __init__(self):
        """Initialize the language detector with language signatures."""
        # Define language signatures
        self.signatures = {
            'python': {
                'extensions': ['.py', '.pyw', '.pyc'],
                'keywords': ['def', 'class', 'import', 'from', 'as', 'with', 'if', 'elif', 'else', 'for', 'while', 
                             'try', 'except', 'finally', 'raise', 'assert', 'lambda', 'yield', 'return', 'and', 'or', 'not'],
                'patterns': [
                    r'^\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\):',  # Function definition
                    r'^\s*class\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(\(.*\))?:',  # Class definition
                    r'^\s*from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import',  # Import statement
                    r'^\s*import\s+[a-zA-Z_][a-zA-Z0-9_.]*',  # Import statement
                    r'^\s*#.*$',  # Python comments
                    r'\s*""".*?"""',  # Python docstrings
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '**', '//', '==', '!=', '>', '<', '>=', '<=', 'and', 'or', 'not', 'in', 'is']
            },
            'c': {
                'extensions': ['.c', '.h'],
                'keywords': ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
                             'int', 'char', 'float', 'double', 'void', 'struct', 'union', 'typedef', 'static', 'const', 'volatile'],
                'patterns': [
                    r'#include\s*[<"][a-zA-Z0-9_.]+[>"]',  # Include directive
                    r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\)\s*\{',  # Function definition
                    r'^\s*//.*$',  # Single line comments
                    r'/\*.*?\*/',  # Multi-line comments
                    r'malloc\s*\(',  # Memory allocation
                    r'free\s*\(',  # Memory deallocation
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--']
            }
        }
    
    def detect_language(self, code, filename=None):
        """
        Detect the programming language of a code snippet.
        
        Args:
            code (str): Code snippet to analyze
            filename (str, optional): Original filename, if available
        
        Returns:
            dict: Detection results with language and confidence score
        """
        # Initialize scores
        scores = {lang: 0 for lang in self.signatures}
        
        # Check file extension if filename is provided
        if filename:
            _, ext = os.path.splitext(filename.lower())
            for lang, sig in self.signatures.items():
                if ext in sig['extensions']:
                    scores[lang] += 50  # High weight for file extension
        
        # Clean the code (remove extra whitespace)
        code = code.strip()
        lines = code.split('\n')
        
        # Check for language-specific patterns
        for lang, sig in self.signatures.items():
            # Check for language keywords
            word_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            words = re.findall(word_pattern, code)
            word_counter = Counter(words)
            
            keyword_count = sum(word_counter[kw] for kw in sig['keywords'] if kw in word_counter)
            scores[lang] += keyword_count * 2
            
            # Check for language-specific patterns in the code
            for pattern in sig['patterns']:
                matches = sum(1 for line in lines if re.search(pattern, line))
                scores[lang] += matches * 5
            
            # Check for language-specific operators
            for op in sig['operators']:
                scores[lang] += code.count(op)
        
        # Determine the language with the highest score
        max_score = max(scores.values())
        detected_langs = [lang for lang, score in scores.items() if score == max_score]
        
        # If multiple languages have the same score, choose based on specific indicators
        if len(detected_langs) > 1:
            # Python specific indicators
            if 'def ' in code or 'class ' in code or 'import ' in code:
                if 'python' in detected_langs:
                    detected_langs = ['python']
            
            # C specific indicators
            if ('#include' in code or 'int main' in code) and '{' in code and '}' in code and ';' in code:
                if 'c' in detected_langs:
                    detected_langs = ['c']
        
        # Calculate confidence (normalize to 0-100%)
        total_scores = sum(scores.values())
        confidence = (max_score / total_scores * 100) if total_scores > 0 else 0
        
        # Default to Python if scores are very low or tied
        detected_lang = detected_langs[0] if detected_langs else 'unknown'
        
        return {
            'language': detected_lang,
            'confidence': min(round(confidence, 1), 100.0),
            'scores': scores
        }


def detect_code_language(code, filename=None):
    """
    Convenient function to detect the language of a code snippet.
    
    Args:
        code (str): Code snippet to analyze
        filename (str, optional): Original filename, if available
    
    Returns:
        str: Detected language name
    """
    detector = LanguageDetector()
    result = detector.detect_language(code, filename)
    return result['language']