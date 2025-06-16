"""
Cognito Analyzer - Core module for code analysis

This module integrates all analyzers and provides a unified interface for code analysis.
"""

from language_detector import detect_code_language
from analyzers.python_analyzer import analyze_python, get_python_analysis_summary
from analyzers.c_analyzer import analyze_c_code
from analyzers.readability_analyzer import analyze_readability
from analyzers.performance_analyzer import analyze_complexity, analyze_memory_usage
from analyzers.security_analyzer import analyze_security, generate_security_suggestion
from llm.integration import LLMIntegration

class CodeAnalyzer:
    """Main class for unified code analysis across languages - FIXED RECURSION."""
    
    def __init__(self):
        """Initialize the code analyzer."""
        pass
    
    def analyze(self, code, filename=None, language=None, use_llm=False, llm_integration=None):
        """
        Analyze code and generate comprehensive feedback - FIXED RECURSION.
        
        Args:
            code (str): Code snippet to analyze
            filename (str, optional): Original filename, if available
            language (str, optional): Force a specific language for analysis
            use_llm (bool): Whether to enhance results with LLM capabilities
            llm_integration: Custom LLM integration to use (optional)
        
        Returns:
            dict: Analysis results with language-specific insights
        """
        # FIXED: Import locally to prevent circular imports
        from language_detector import LanguageDetector
        
        # Detect language if not specified
        if not language:
            detector = LanguageDetector()
            detection_result = detector.detect_language(code, filename)
            language = detection_result['language']
        
        # Initialize results
        results = {
            'language': language,
            'analysis': {},
            'summary': {},
            'suggestions': []
        }
        
        # FIXED: Use try-except to handle import errors gracefully
        try:
            # Language-specific analysis
            if language == 'python':
                results['analysis'] = self._analyze_python(code)
                # Import locally to prevent recursion
                from analyzers.python_analyzer import get_python_analysis_summary
                results['summary'] = get_python_analysis_summary(results['analysis'])
            elif language == 'c':
                from analyzers.c_analyzer import analyze_c_code
                c_analysis = analyze_c_code(code)
                results['analysis'] = c_analysis
                results['summary'] = c_analysis.get('summary', {})
            else:
                # Fallback to general analysis for unsupported languages
                results['analysis'] = self._perform_general_analysis(code)
                results['summary'] = {
                    'message': 'Limited analysis available for this language',
                    'readability': 'Unknown',
                    'complexity': 'Unknown',
                    'security': 'Unknown'
                }
        except Exception as e:
            # Graceful fallback if analysis fails
            results['analysis'] = {
                'readability': f"Analysis error: {str(e)[:100]}",
                'performance': "Could not complete performance analysis",
                'security': "Could not complete security analysis"
            }
            results['summary'] = {
                'error': str(e)[:100],
                'status': 'failed'
            }
        
        # Generate suggestions from analysis
        results['suggestions'] = self._generate_suggestions(results)
        
        # FIXED: Simplified LLM enhancement to prevent recursion
        if use_llm:
            try:
                if llm_integration:
                    results = llm_integration.enhance_analysis(code, results)
                else:
                    # Simple LLM enhancement without circular imports
                    results["llm_enhanced"] = False
                    results["llm_note"] = "LLM enhancement available but not configured"
            except Exception as e:
                results["llm_error"] = str(e)[:100]
        
        return results
    
    def _analyze_python(self, code):
        """
        Perform comprehensive Python code analysis.
        
        Args:
            code (str): Python code to analyze
        
        Returns:
            dict: Analysis results
        """
        # Use the Python analyzer for detailed analysis
        python_analysis = analyze_python(code)
        
        # Add general analysis
        general_analysis = self._perform_general_analysis(code)
        python_analysis.update(general_analysis)
        
        return python_analysis
    
    def _perform_general_analysis(self, code):
        """
        Perform language-agnostic analysis.
        
        Args:
            code (str): Code to analyze
        
        Returns:
            dict: General analysis results
        """
        results = {}
        
        # Readability analysis
        results['readability'] = analyze_readability(code)
        
        # Performance analysis
        results['performance'] = analyze_complexity(code)
        
        # Security analysis
        security_issues = analyze_security(code)
        results['security'] = generate_security_suggestion(security_issues)
        
        return results
    
    def _generate_suggestions(self, results):
        """
        Generate prioritized suggestions based on analysis results.
        
        Args:
            results (dict): Analysis results
        
        Returns:
            list: Prioritized suggestions
        """
        suggestions = []
        
        # Extract suggestions from analysis based on language
        if results['language'] == 'python':
            # Add style suggestions
            for suggestion in results['analysis'].get('style', {}).get('suggestions', []):
                if 'No style issues detected' not in suggestion:
                    suggestions.append({
                        'category': 'Style',
                        'message': suggestion,
                        'priority': 'medium'
                    })
            
            # Add best practices suggestions
            for pattern in results['analysis'].get('best_practices', []):
                if 'No common anti-patterns detected' not in pattern:
                    suggestions.append({
                        'category': 'Best Practices',
                        'message': pattern,
                        'priority': 'high'
                    })
        
        elif results['language'] == 'c':
            # Add C-specific suggestions from issues
            for issue in results['analysis'].get('issues', []):
                suggestions.append({
                    'category': issue['type'].capitalize(),
                    'message': issue['message'],
                    'priority': issue['priority']
                })
        
        # Add general suggestions for all languages
        
        # Security suggestions
        if 'security' in results['analysis']:
            security_suggestion = results['analysis']['security']
            if isinstance(security_suggestion, str) and 'No common security issues detected' not in security_suggestion:
                suggestions.append({
                    'category': 'Security',
                    'message': security_suggestion,
                    'priority': 'high'
                })
            elif isinstance(security_suggestion, list):
                for issue in security_suggestion:
                    if 'No common security issues detected' not in issue:
                        suggestions.append({
                            'category': 'Security',
                            'message': issue,
                            'priority': 'high'
                        })
        
        # Performance suggestions
        if 'performance' in results['analysis']:
            performance = results['analysis']['performance']
            if isinstance(performance, str) and 'looks good' not in performance:
                suggestions.append({
                    'category': 'Performance',
                    'message': performance,
                    'priority': 'medium'
                })
            elif isinstance(performance, list):
                for issue in performance:
                    if 'looks good' not in issue:
                        suggestions.append({
                            'category': 'Performance',
                            'message': issue,
                            'priority': 'medium'
                        })
        
        # Readability suggestions
        if 'readability' in results['analysis']:
            readability = results['analysis']['readability']
            if 'looks good' not in readability:
                suggestions.append({
                    'category': 'Readability',
                    'message': readability,
                    'priority': 'low'
                })
        
        # Sort suggestions by priority
        priority_map = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_map.get(x['priority'], 3))
        
        return suggestions


def analyze_code(code, filename=None, language=None, use_llm=False, llm_integration=None):
    """
    Convenient function to analyze code.
    
    Args:
        code (str): Code to analyze
        filename (str, optional): Original filename, if available
        language (str, optional): Force a specific language for analysis
        use_llm (bool): Whether to enhance results with LLM capabilities
        llm_integration: Custom LLM integration to use (optional)
    
    Returns:
        dict: Analysis results
    """
    analyzer = CodeAnalyzer()
    results = analyzer.analyze(code, filename, language, use_llm=use_llm, llm_integration=llm_integration)
    return results