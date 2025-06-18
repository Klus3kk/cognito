"""
Universal analyzer that integrates all language-specific analyzers.
This module provides a unified interface for analyzing code in any supported language.
"""

import os
import sys
from typing import Dict, Any, Optional

# Add the analyzers directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import all language analyzers
try:
    from analyzers.python_analyzer import analyze_python
    from analyzers.c_analyzer import analyze_c_code
    from analyzers.javascript_analyzer import analyze_javascript_code
    from analyzers.java_analyzer import analyze_java_code
    from analyzers.cpp_analyzer import analyze_cpp_code
except ImportError:
    # Fallback imports if analyzers are in the same directory
    try:
        from python_analyzer import analyze_python
        from c_analyzer import analyze_c_code
        from javascript_analyzer import analyze_javascript_code
        from java_analyzer import analyze_java_code
        from cpp_analyzer import analyze_cpp_code
    except ImportError as e:
        print(f"Warning: Could not import all analyzers: {e}")

class UniversalAnalyzer:
    """Universal code analyzer that supports multiple programming languages."""
    
    def __init__(self):
        """Initialize the universal analyzer with language mappings."""
        self.language_analyzers = {
            'python': self._analyze_python_wrapper,
            'c': self._analyze_c_wrapper,
            'javascript': self._analyze_javascript_wrapper,
            'typescript': self._analyze_javascript_wrapper,  # TypeScript uses JS analyzer
            'java': self._analyze_java_wrapper,
            'cpp': self._analyze_cpp_wrapper,
            'c++': self._analyze_cpp_wrapper,
        }
        
        # Languages with full analyzer support
        self.fully_supported = ['python', 'c', 'javascript', 'typescript', 'java', 'cpp', 'c++']
        
        # Languages with basic support (using generic analyzer)
        self.basic_supported = ['go', 'rust', 'php', 'ruby', 'csharp', 'c#']
        
    def analyze_code(self, code: str, language: str, filename: str = "") -> Dict[str, Any]:
        """
        Analyze code using the appropriate language analyzer.
        
        Args:
            code: Source code to analyze
            language: Programming language
            filename: Original filename (optional)
            
        Returns:
            Analysis results dictionary
        """
        language = language.lower()
        
        # Use language-specific analyzer if available
        if language in self.language_analyzers:
            try:
                return self.language_analyzers[language](code, filename)
            except Exception as e:
                return self._create_error_result(language, str(e))
        
        # Use generic analyzer for basic supported languages
        elif language in self.basic_supported:
            return self._analyze_generic(code, language, filename)
        
        # Unknown language
        else:
            return self._analyze_unknown(code, language, filename)
    
    def _analyze_python_wrapper(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Wrapper for Python analyzer."""
        try:
            result = analyze_python(code)
            return self._standardize_result(result, 'python')
        except Exception as e:
            return self._create_fallback_result(code, 'python', str(e))
    
    def _analyze_c_wrapper(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Wrapper for C analyzer."""
        try:
            result = analyze_c_code(code)
            return self._standardize_result(result, 'c')
        except Exception as e:
            return self._create_fallback_result(code, 'c', str(e))
    
    def _analyze_javascript_wrapper(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Wrapper for JavaScript/TypeScript analyzer."""
        try:
            result = analyze_javascript_code(code)
            return self._standardize_result(result, 'javascript')
        except Exception as e:
            return self._create_fallback_result(code, 'javascript', str(e))
    
    def _analyze_java_wrapper(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Wrapper for Java analyzer."""
        try:
            result = analyze_java_code(code)
            return self._standardize_result(result, 'java')
        except Exception as e:
            return self._create_fallback_result(code, 'java', str(e))
    
    def _analyze_cpp_wrapper(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Wrapper for C++ analyzer."""
        try:
            result = analyze_cpp_code(code)
            return self._standardize_result(result, 'cpp')
        except Exception as e:
            return self._create_fallback_result(code, 'cpp', str(e))
    
    def _analyze_generic(self, code: str, language: str, filename: str = "") -> Dict[str, Any]:
        """Generic analyzer for basic supported languages."""
        issues = []
        metrics = {}
        
        # Basic analysis for any language
        lines = code.split('\n')
        
        # Line length check
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            issues.append({
                'type': 'style',
                'message': f"Lines exceeding 120 characters: {len(long_lines)} lines",
                'priority': 'low'
            })
        
        # Basic complexity estimation
        complexity_indicators = [
            'if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch'
        ]
        complexity_score = sum(code.lower().count(indicator) for indicator in complexity_indicators)
        
        if complexity_score > 20:
            issues.append({
                'type': 'complexity',
                'message': f"High complexity detected (score: {complexity_score})",
                'priority': 'medium'
            })
        
        # Language-specific checks
        if language == 'go':
            issues.extend(self._analyze_go_specific(code))
        elif language == 'rust':
            issues.extend(self._analyze_rust_specific(code))
        elif language == 'php':
            issues.extend(self._analyze_php_specific(code))
        elif language == 'ruby':
            issues.extend(self._analyze_ruby_specific(code))
        elif language in ['csharp', 'c#']:
            issues.extend(self._analyze_csharp_specific(code))
        
        # Calculate basic metrics
        metrics = {
            'lines_of_code': len([line for line in lines if line.strip()]),
            'complexity_score': complexity_score,
            'long_lines': len(long_lines),
            'maintainability_index': max(0, 100 - complexity_score - len(long_lines))
        }
        
        return {
            'language': language,
            'issues': issues,
            'metrics': metrics,
            'summary': {
                'issue_count': len(issues),
                'maintainability': {
                    'index': metrics['maintainability_index'],
                    'rating': self._get_maintainability_rating(metrics['maintainability_index'])
                },
                'analyzer_type': 'generic'
            }
        }
    
    def _analyze_go_specific(self, code: str) -> list:
        """Go-specific analysis patterns."""
        issues = []
        
        # Check for proper error handling
        if 'err != nil' not in code and 'error' in code:
            issues.append({
                'type': 'best_practice',
                'message': "Consider proper error handling with 'if err != nil'",
                'priority': 'medium'
            })
        
        # Check for goroutine usage without context
        if 'go ' in code and 'context' not in code:
            issues.append({
                'type': 'best_practice',
                'message': "Consider using context for goroutine management",
                'priority': 'low'
            })
        
        return issues
    
    def _analyze_rust_specific(self, code: str) -> list:
        """Rust-specific analysis patterns."""
        issues = []
        
        # Check for unwrap() usage
        unwrap_count = code.count('.unwrap()')
        if unwrap_count > 0:
            issues.append({
                'type': 'best_practice',
                'message': f"Found {unwrap_count} .unwrap() calls. Consider proper error handling",
                'priority': 'medium'
            })
        
        # Check for unsafe blocks
        unsafe_count = code.count('unsafe')
        if unsafe_count > 0:
            issues.append({
                'type': 'security',
                'message': f"Found {unsafe_count} unsafe blocks. Ensure memory safety",
                'priority': 'high'
            })
        
        return issues
    
    def _analyze_php_specific(self, code: str) -> list:
        """PHP-specific analysis patterns."""
        issues = []
        
        # Check for SQL injection risks
        if '$_GET' in code or '$_POST' in code:
            if 'mysql_query' in code or 'mysqli_query' in code:
                issues.append({
                    'type': 'security',
                    'message': "Potential SQL injection risk with user input",
                    'priority': 'high'
                })
        
        # Check for error reporting
        if 'error_reporting' not in code and 'ini_set' not in code:
            issues.append({
                'type': 'best_practice',
                'message': "Consider setting proper error reporting",
                'priority': 'low'
            })
        
        return issues
    
    def _analyze_ruby_specific(self, code: str) -> list:
        """Ruby-specific analysis patterns."""
        issues = []
        
        # Check for eval usage
        if 'eval(' in code:
            issues.append({
                'type': 'security',
                'message': "Use of eval() detected - security risk",
                'priority': 'high'
            })
        
        # Check for proper exception handling
        if 'rescue' in code and 'ensure' not in code:
            issues.append({
                'type': 'best_practice',
                'message': "Consider using 'ensure' for cleanup in exception handling",
                'priority': 'low'
            })
        
        return issues
    
    def _analyze_csharp_specific(self, code: str) -> list:
        """C#-specific analysis patterns."""
        issues = []
        
        # Check for proper using statements
        if 'new ' in code and 'IDisposable' in code and 'using(' not in code:
            issues.append({
                'type': 'best_practice',
                'message': "Consider using 'using' statements for IDisposable objects",
                'priority': 'medium'
            })
        
        # Check for async/await patterns
        if 'async' in code and 'ConfigureAwait(false)' not in code:
            issues.append({
                'type': 'performance',
                'message': "Consider using ConfigureAwait(false) in library code",
                'priority': 'low'
            })
        
        return issues
    
    def _analyze_unknown(self, code: str, language: str, filename: str = "") -> Dict[str, Any]:
        """Analysis for unknown languages."""
        lines = code.split('\n')
        
        return {
            'language': language,
            'issues': [{
                'type': 'info',
                'message': f"Language '{language}' not fully supported. Basic analysis only.",
                'priority': 'info'
            }],
            'metrics': {
                'lines_of_code': len([line for line in lines if line.strip()]),
                'total_lines': len(lines),
                'estimated_complexity': 'unknown'
            },
            'summary': {
                'issue_count': 1,
                'maintainability': {
                    'index': 50,
                    'rating': 'Unknown'
                },
                'analyzer_type': 'basic'
            }
        }
    
    def _standardize_result(self, result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Standardize analyzer results to common format."""
        if not isinstance(result, dict):
            return self._create_fallback_result("", language, "Invalid result format")
        
        # Ensure all required fields exist
        standardized = {
            'language': language,
            'issues': result.get('issues', []),
            'metrics': result.get('metrics', {}),
            'summary': result.get('summary', {}),
            'analyzer_type': 'specialized'
        }
        
        # Ensure summary has required fields
        if 'issue_count' not in standardized['summary']:
            standardized['summary']['issue_count'] = len(standardized['issues'])
        
        if 'maintainability' not in standardized['summary']:
            mi = standardized['metrics'].get('maintainability_index', 50)
            standardized['summary']['maintainability'] = {
                'index': mi,
                'rating': self._get_maintainability_rating(mi)
            }
        
        return standardized
    
    def _create_error_result(self, language: str, error: str) -> Dict[str, Any]:
        """Create error result when analyzer fails."""
        return {
            'language': language,
            'issues': [{
                'type': 'error',
                'message': f"Analysis failed: {error}",
                'priority': 'high'
            }],
            'metrics': {'error': error},
            'summary': {
                'issue_count': 1,
                'maintainability': {'index': 0, 'rating': 'Error'},
                'analyzer_type': 'error'
            }
        }
    
    def _create_fallback_result(self, code: str, language: str, error: str) -> Dict[str, Any]:
        """Create fallback result with basic analysis."""
        lines = code.split('\n') if code else []
        
        return {
            'language': language,
            'issues': [{
                'type': 'warning',
                'message': f"Specialized analyzer failed ({error}), using basic analysis",
                'priority': 'low'
            }],
            'metrics': {
                'lines_of_code': len([line for line in lines if line.strip()]),
                'analyzer_error': error
            },
            'summary': {
                'issue_count': 1,
                'maintainability': {'index': 50, 'rating': 'Unknown'},
                'analyzer_type': 'fallback'
            }
        }
    
    def _get_maintainability_rating(self, index: float) -> str:
        """Convert maintainability index to rating."""
        if index >= 85:
            return "Excellent"
        elif index >= 65:
            return "Good"
        elif index >= 40:
            return "Fair"
        else:
            return "Poor"
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages with their support level."""
        return {
            **{lang: 'full' for lang in self.fully_supported},
            **{lang: 'basic' for lang in self.basic_supported}
        }
    
    def is_language_supported(self, language: str) -> tuple:
        """Check if language is supported and return support level."""
        language = language.lower()
        if language in self.fully_supported:
            return True, 'full'
        elif language in self.basic_supported:
            return True, 'basic'
        else:
            return False, 'none'


# Convenience function for external use
def analyze_code_universal(code: str, language: str, filename: str = "") -> Dict[str, Any]:
    """
    Universal code analysis function.
    
    Args:
        code: Source code to analyze
        language: Programming language
        filename: Original filename (optional)
        
    Returns:
        Analysis results dictionary
    """
    analyzer = UniversalAnalyzer()
    return analyzer.analyze_code(code, language, filename)


# Export the main analyzer class and function
__all__ = ['UniversalAnalyzer', 'analyze_code_universal']