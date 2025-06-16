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
    """Main class for unified code analysis across languages - FIXED C integration."""
    
    def analyze(self, code, filename=None, language=None, use_llm=False, llm_integration=None):
        """
        Analyze code and generate comprehensive feedback - FIXED C analysis.
        """
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
        
        try:
            # Language-specific analysis
            if language == 'python':
                results['analysis'] = self._analyze_python(code)
                from analyzers.python_analyzer import get_python_analysis_summary
                results['summary'] = get_python_analysis_summary(results['analysis'])
                
            elif language == 'c':
                # FIXED: Call the comprehensive C analyzer
                results = self._analyze_c_comprehensive(code)
                return results  # Return early with comprehensive C results
                
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
        
        # LLM enhancement
        if use_llm:
            try:
                if llm_integration:
                    results = llm_integration.enhance_analysis(code, results)
                else:
                    results["llm_enhanced"] = False
                    results["llm_note"] = "LLM enhancement available but not configured"
            except Exception as e:
                results["llm_error"] = str(e)[:100]
        
        return results
    
    def _analyze_c_comprehensive(self, code):
        """
        Comprehensive C code analysis using the full C analyzer.
        
        Args:
            code (str): C code to analyze
        
        Returns:
            dict: Comprehensive C analysis results
        """
        try:
            # Import and use the full C analyzer
            from analyzers.c_analyzer import analyze_c_code
            
            # Get comprehensive C analysis
            c_analysis = analyze_c_code(code)
            
            # Transform C analyzer results to match expected format
            suggestions = []
            
            # Add issues as suggestions
            for issue in c_analysis.get('issues', []):
                suggestions.append({
                    'category': issue.get('type', 'General').title(),
                    'message': issue.get('message', ''),
                    'priority': issue.get('priority', 'medium')
                })
            
            # Add security suggestions
            security_metrics = c_analysis.get('metrics', {}).get('memory_safety', {})
            if security_metrics.get('malloc_count', 0) > security_metrics.get('free_count', 0):
                suggestions.append({
                    'category': 'Security',
                    'message': 'Potential memory leak detected - check malloc/free pairs',
                    'priority': 'high'
                })
            
            # Add complexity suggestions
            complexity_metrics = c_analysis.get('metrics', {}).get('complexity', {})
            if complexity_metrics.get('max_nesting', 0) > 4:
                suggestions.append({
                    'category': 'Performance',
                    'message': f"High nesting complexity (depth: {complexity_metrics['max_nesting']})",
                    'priority': 'medium'
                })
            
            # Add maintainability suggestions
            maintainability_index = c_analysis.get('metrics', {}).get('maintainability_index', 50)
            if maintainability_index < 40:
                suggestions.append({
                    'category': 'Maintainability',
                    'message': f"Low maintainability index ({maintainability_index:.1f})",
                    'priority': 'medium'
                })
            
            # Build comprehensive results in expected format
            results = {
                'language': 'c',
                'analysis': {
                    'security': self._format_c_security_analysis(c_analysis),
                    'performance': self._format_c_performance_analysis(c_analysis),
                    'readability': self._format_c_readability_analysis(c_analysis),
                    'style': c_analysis.get('issues', [])
                },
                'summary': c_analysis.get('summary', {}),
                'suggestions': suggestions,
                'metrics': c_analysis.get('metrics', {}),
                'c_analysis_raw': c_analysis  # Keep original for debugging
            }
            
            return results
            
        except ImportError:
            # Fallback if C analyzer not available
            return {
                'language': 'c',
                'analysis': {
                    'security': 'C security analysis not available',
                    'performance': 'C performance analysis not available',
                    'readability': 'C readability analysis not available'
                },
                'suggestions': [
                    {'category': 'System', 'message': 'C analyzer not fully available', 'priority': 'low'}
                ],
                'summary': {'status': 'limited'}
            }
        except Exception as e:
            # Fallback on error
            return {
                'language': 'c',
                'analysis': {
                    'security': f'C analysis error: {str(e)[:50]}',
                    'performance': 'Could not complete C performance analysis',
                    'readability': 'Could not complete C readability analysis'
                },
                'suggestions': [
                    {'category': 'System', 'message': f'C analysis failed: {str(e)[:50]}', 'priority': 'medium'}
                ],
                'summary': {'error': str(e)[:100]}
            }
    
    def _format_c_security_analysis(self, c_analysis):
        """Format C security analysis results."""
        security_issues = [issue for issue in c_analysis.get('issues', []) 
                          if issue.get('type') == 'security']
        
        if not security_issues:
            return "C Security Analysis: No major security vulnerabilities detected."
        
        issue_descriptions = []
        for issue in security_issues[:3]:  # Show top 3
            issue_descriptions.append(f"- {issue.get('message', 'Security issue detected')}")
        
        return f"C Security Analysis: {len(security_issues)} security issues found:\n" + "\n".join(issue_descriptions)
    
    def _format_c_performance_analysis(self, c_analysis):
        """Format C performance analysis results."""
        complexity_metrics = c_analysis.get('metrics', {}).get('complexity', {})
        
        performance_issues = []
        
        max_nesting = complexity_metrics.get('max_nesting', 0)
        if max_nesting > 3:
            performance_issues.append(f"High nesting complexity (depth: {max_nesting})")
        
        cyclomatic = complexity_metrics.get('cyclomatic_complexity', 0)
        if cyclomatic > 10:
            performance_issues.append(f"High cyclomatic complexity ({cyclomatic})")
        
        goto_count = complexity_metrics.get('control_structures', {}).get('goto', 0)
        if goto_count > 2:
            performance_issues.append(f"Excessive goto usage ({goto_count} statements)")
        
        if not performance_issues:
            return "C Performance Analysis: Code complexity within acceptable limits."
        
        return "C Performance Analysis: " + "; ".join(performance_issues)
    
    def _format_c_readability_analysis(self, c_analysis):
        """Format C readability analysis results."""
        style_issues = [issue for issue in c_analysis.get('issues', []) 
                       if issue.get('type') == 'style']
        
        maintainability_index = c_analysis.get('metrics', {}).get('maintainability_index', 50)
        
        if maintainability_index >= 70 and len(style_issues) <= 2:
            return f"C Readability Analysis: Good maintainability (index: {maintainability_index:.1f})"
        elif maintainability_index >= 40:
            return f"C Readability Analysis: Moderate maintainability (index: {maintainability_index:.1f}). {len(style_issues)} style issues detected."
        else:
            return f"C Readability Analysis: Poor maintainability (index: {maintainability_index:.1f}). Consider refactoring."


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