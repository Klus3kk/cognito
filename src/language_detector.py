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
            },
            'javascript': {
                'extensions': ['.js', '.jsx', '.mjs', '.ts', '.tsx'],
                'keywords': ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'do',
                           'switch', 'case', 'break', 'continue', 'return', 'try', 'catch', 'finally',
                           'throw', 'async', 'await', 'class', 'extends', 'import', 'export'],
                'patterns': [
                    r'function\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(',
                    r'^\s*const\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*=',
                    r'^\s*let\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*=',
                    r'^\s*var\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*=',
                    r'^\s*//.*$',
                    r'/\*.*?\*/',
                    r'=>',
                    r'console\.(log|error|warn)',
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '===', '!=', '!==', '>', '<', '>=', '<=', '&&', '||', '!']
            },
            'java': {
                'extensions': ['.java'],
                'keywords': ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements',
                           'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
                           'return', 'try', 'catch', 'finally', 'throw', 'throws', 'import', 'package'],
                'patterns': [
                    r'public\s+class\s+[A-Z][a-zA-Z0-9_]*',
                    r'public\s+static\s+void\s+main\s*\(',
                    r'^\s*import\s+[a-zA-Z_][a-zA-Z0-9_.]*;',
                    r'^\s*package\s+[a-zA-Z_][a-zA-Z0-9_.]*;',
                    r'^\s*//.*$',
                    r'/\*.*?\*/',
                    r'System\.out\.print',
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!']
            },
            'cpp': {
                'extensions': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh'],
                'keywords': ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
                           'int', 'char', 'float', 'double', 'void', 'bool', 'class', 'struct', 'public', 'private',
                           'protected', 'virtual', 'override', 'namespace', 'using', 'template', 'typename'],
                'patterns': [
                    r'#include\s*[<"][a-zA-Z0-9_.]+[>"]',
                    r'^\s*class\s+[A-Z][a-zA-Z0-9_]*',
                    r'^\s*namespace\s+[a-zA-Z_][a-zA-Z0-9_]*',
                    r'std::[a-zA-Z_]+',
                    r'^\s*//.*$',
                    r'/\*.*?\*/',
                    r'cout\s*<<',
                    r'cin\s*>>',
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--', '::']
            },
            'csharp': {
                'extensions': ['.cs'],
                'keywords': ['public', 'private', 'protected', 'internal', 'class', 'interface', 'namespace',
                           'using', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
                           'return', 'try', 'catch', 'finally', 'throw', 'var', 'string', 'int', 'bool'],
                'patterns': [
                    r'^\s*using\s+[A-Z][a-zA-Z0-9_.]*;',
                    r'^\s*namespace\s+[A-Z][a-zA-Z0-9_.]*',
                    r'^\s*public\s+class\s+[A-Z][a-zA-Z0-9_]*',
                    r'^\s*//.*$',
                    r'/\*.*?\*/',
                    r'Console\.(WriteLine|Write)',
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!']
            },
            'go': {
                'extensions': ['.go'],
                'keywords': ['package', 'import', 'func', 'var', 'const', 'type', 'struct', 'interface',
                           'if', 'else', 'for', 'switch', 'case', 'break', 'continue', 'return',
                           'go', 'defer', 'chan', 'select', 'range'],
                'patterns': [
                    r'^\s*package\s+[a-zA-Z_][a-zA-Z0-9_]*',
                    r'^\s*import\s*\(',
                    r'^\s*func\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
                    r'^\s*type\s+[A-Z][a-zA-Z0-9_]*\s+struct',
                    r'^\s*//.*$',
                    r'/\*.*?\*/',
                    r'fmt\.(Print|Sprint)',
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', ':=']
            },
            'rust': {
                'extensions': ['.rs'],
                'keywords': ['fn', 'let', 'mut', 'const', 'static', 'struct', 'enum', 'impl', 'trait',
                           'if', 'else', 'match', 'for', 'while', 'loop', 'break', 'continue', 'return',
                           'pub', 'mod', 'use', 'crate', 'super', 'self'],
                'patterns': [
                    r'^\s*fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
                    r'^\s*let\s+(mut\s+)?[a-zA-Z_][a-zA-Z0-9_]*',
                    r'^\s*struct\s+[A-Z][a-zA-Z0-9_]*',
                    r'^\s*use\s+[a-zA-Z_][a-zA-Z0-9_:]*;',
                    r'^\s*//.*$',
                    r'/\*.*?\*/',
                    r'println!\s*\(',
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '=>']
            },
            'php': {
                'extensions': ['.php'],
                'keywords': ['function', 'class', 'interface', 'trait', 'namespace', 'use', 'if', 'else',
                           'elseif', 'for', 'foreach', 'while', 'do', 'switch', 'case', 'break', 'continue',
                           'return', 'try', 'catch', 'finally', 'throw', 'public', 'private', 'protected'],
                'patterns': [
                    r'<\?php',
                    r'^\s*function\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
                    r'^\s*class\s+[A-Z][a-zA-Z0-9_]*',
                    r'\$[a-zA-Z_][a-zA-Z0-9_]*',
                    r'^\s*//.*$',
                    r'/\*.*?\*/',
                    r'echo\s+',
                    r'print\s+',
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '===', '!=', '!==', '>', '<', '>=', '<=', '&&', '||', '!', '.']
            },
            'ruby': {
                'extensions': ['.rb'],
                'keywords': ['def', 'class', 'module', 'if', 'elsif', 'else', 'unless', 'case', 'when',
                           'for', 'while', 'until', 'break', 'next', 'return', 'yield', 'begin', 'rescue',
                           'ensure', 'end', 'true', 'false', 'nil', 'and', 'or', 'not'],
                'patterns': [
                    r'^\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*',
                    r'^\s*class\s+[A-Z][a-zA-Z0-9_]*',
                    r'^\s*module\s+[A-Z][a-zA-Z0-9_]*',
                    r'^\s*#.*$',
                    r'puts\s+',
                    r'print\s+',
                    r'@[a-zA-Z_][a-zA-Z0-9_]*',
                    r'@@[a-zA-Z_][a-zA-Z0-9_]*',
                ],
                'operators': ['=', '+', '-', '*', '/', '%', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '<<', '>>' ]
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
            word_pattern = r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b'
            words = re.findall(word_pattern, code)
            word_counter = Counter(words)
            
            keyword_count = sum(word_counter[kw] for kw in sig['keywords'] if kw in word_counter)
            scores[lang] += keyword_count * 2
            
            # Check for language-specific patterns in the code
            for pattern in sig['patterns']:
                matches = sum(1 for line in lines if re.search(pattern, line, re.MULTILINE))
                scores[lang] += matches * 5
            
            # Check for language-specific operators
            for op in sig['operators']:
                scores[lang] += code.count(op) * 0.5
        
        # Apply language-specific heuristics
        scores = self._apply_language_heuristics(code, scores)
        
        # Determine the language with the highest score
        max_score = max(scores.values())
        detected_langs = [lang for lang, score in scores.items() if score == max_score]
        
        # If multiple languages have the same score, choose based on specific indicators
        if len(detected_langs) > 1:
            detected_langs = self._resolve_language_conflicts(code, detected_langs)
        
        # Calculate confidence (normalize to 0-100%)
        total_scores = sum(scores.values())
        confidence = (max_score / total_scores * 100) if total_scores > 0 else 0
        
        # Default to most likely if scores are very low
        detected_lang = detected_langs[0] if detected_langs else 'unknown'
        
        return {
            'language': detected_lang,
            'confidence': min(round(confidence, 1), 100.0),
            'scores': scores,
            'alternatives': [lang for lang, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[1:4]]
        }
    
    def _apply_language_heuristics(self, code, scores):
        """Apply specific heuristics to improve language detection."""
        
        # Python specific heuristics
        if ':' in code and 'def ' in code:
            scores['python'] += 10
        if 'import ' in code or 'from ' in code:
            scores['python'] += 8
        
        # JavaScript/TypeScript heuristics
        if 'function(' in code.replace(' ', '') or '=>' in code:
            scores['javascript'] += 10
        if 'console.' in code or 'document.' in code or 'window.' in code:
            scores['javascript'] += 15
        
        # Java heuristics
        if 'public class' in code and 'public static void main' in code:
            scores['java'] += 20
        if 'System.out.' in code:
            scores['java'] += 10
        
        # C/C++ heuristics
        if '#include' in code and ('{' in code and '}' in code and ';' in code):
            if 'std::' in code or 'cout' in code or 'cin' in code:
                scores['cpp'] += 15
            else:
                scores['c'] += 15
        if 'malloc(' in code and 'free(' in code:
            scores['c'] += 10
        
        # C# heuristics
        if 'using System' in code or 'Console.WriteLine' in code:
            scores['csharp'] += 15
        
        # Go heuristics
        if 'package main' in code and 'func main()' in code:
            scores['go'] += 20
        if 'fmt.Print' in code or 'fmt.Sprint' in code:
            scores['go'] += 10
        
        # Rust heuristics
        if 'fn main()' in code or 'println!' in code:
            scores['rust'] += 15
        if 'let mut' in code or 'impl ' in code:
            scores['rust'] += 10
        
        # PHP heuristics
        if '<?php' in code or '$' in code:
            scores['php'] += 15
        
        # Ruby heuristics
        if 'puts ' in code or '@' in code or 'end' in code:
            scores['ruby'] += 10
        
        return scores
    
    def _resolve_language_conflicts(self, code, detected_langs):
        """Resolve conflicts when multiple languages have similar scores."""
        
        # C vs C++ conflict
        if 'c' in detected_langs and 'cpp' in detected_langs:
            if 'std::' in code or 'cout' in code or 'class ' in code:
                return ['cpp']
            else:
                return ['c']
        
        # JavaScript vs TypeScript conflict
        if 'javascript' in detected_langs:
            if ': string' in code or ': number' in code or 'interface ' in code:
                return ['typescript'] if 'typescript' in self.signatures else ['javascript']
            else:
                return ['javascript']
        
        # Return first language as default
        return [detected_langs[0]]


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


def get_language_info(language):
    """
    Get information about a supported language.
    
    Args:
        language (str): Language name
        
    Returns:
        dict: Language information including extensions and features
    """
    detector = LanguageDetector()
    if language.lower() in detector.signatures:
        lang_data = detector.signatures[language.lower()]
        return {
            'name': language.title(),
            'extensions': lang_data['extensions'],
            'keywords_count': len(lang_data['keywords']),
            'patterns_count': len(lang_data['patterns'])
        }
    return None


def get_supported_languages():
    """
    Get list of all supported languages.
    
    Returns:
        list: List of supported language names
    """
    detector = LanguageDetector()
    return list(detector.signatures.keys())