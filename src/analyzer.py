"""
Fixed analyzer.py with missing _generate_suggestions method added.
This fixes the 'CodeAnalyzer' object has no attribute '_generate_suggestions' error.
"""

from language_detector import detect_code_language
from analyzers.universal_analyzer import UniversalAnalyzer, analyze_code_universal
from analyzers.readability_analyzer import analyze_readability
from analyzers.performance_analyzer import analyze_complexity, analyze_memory_usage
from analyzers.security_analyzer import analyze_security, generate_security_suggestion

# Import specific analyzers for fallback
try:
    from analyzers.python_analyzer import analyze_python, get_python_analysis_summary
    from analyzers.c_analyzer import analyze_c_code
    from analyzers.javascript_analyzer import analyze_javascript_code
    from analyzers.java_analyzer import analyze_java_code
    from analyzers.cpp_analyzer import analyze_cpp_code
except ImportError as e:
    print(f"Warning: Some analyzers not available: {e}")


class CodeAnalyzer:
    """Enhanced main class for unified code analysis across all supported languages."""
    
    def __init__(self):
        """Initialize the code analyzer with universal language support."""
        self.universal_analyzer = UniversalAnalyzer()
        
    def analyze(self, code, filename=None, language=None, use_llm=False, llm_integration=None):
        """
        Analyze code with comprehensive language support and generate feedback.
        
        Args:
            code (str): Code to analyze
            filename (str, optional): Original filename
            language (str, optional): Force specific language
            use_llm (bool): Whether to enhance with LLM
            llm_integration: Custom LLM integration instance
            
        Returns:
            dict: Comprehensive analysis results
        """
        try:
            # Detect language if not specified
            if not language:
                from language_detector import LanguageDetector
                detector = LanguageDetector()
                detection_result = detector.detect_language(code, filename)
                language = detection_result['language']
                confidence = detection_result['confidence']
            else:
                confidence = 100.0
            
            # Check if language is supported
            is_supported, support_level = self.universal_analyzer.is_language_supported(language)
            
            # Initialize comprehensive results structure
            results = {
                'language': language,
                'confidence': confidence,
                'supported': is_supported,
                'support_level': support_level,
                'analysis': {},
                'suggestions': [],
                'summary': {},
                'metrics': {}
            }
            
            # Perform comprehensive analysis based on language support
            if language.lower() == 'python':
                results = self._analyze_python_comprehensive(code, results, use_llm, llm_integration)
            elif language.lower() in ['c', 'cpp', 'c++']:
                results = self._analyze_c_comprehensive(code, results, language)
            elif language.lower() in ['javascript', 'typescript']:
                results = self._analyze_javascript_comprehensive(code, results)
            elif language.lower() == 'java':
                results = self._analyze_java_comprehensive(code, results)
            else:
                # Use universal analyzer for other languages
                results = self._analyze_universal(code, results, language)
            
            # Generate suggestions from all analysis results
            if not results.get('suggestions'):
                results['suggestions'] = self._generate_suggestions(results)
            
            # Apply LLM enhancement if requested and available
            if use_llm and llm_integration:
                try:
                    results = llm_integration.enhance_analysis(code, results)
                except Exception as e:
                    print(f"LLM enhancement failed: {e}")
            
            return results
            
        except Exception as e:
            # Graceful error handling
            return self._handle_analysis_error(code, language or 'unknown', e, {
                'language': language or 'unknown',
                'analysis': {},
                'suggestions': [],
                'summary': {},
                'metrics': {}
            })
    
    def _generate_suggestions(self, results):
        """
        Generate suggestions based on analysis results.
        
        Args:
            results (dict): Analysis results containing various analysis outputs
            
        Returns:
            list: List of suggestion dictionaries
        """
        suggestions = []
        
        try:
            analysis = results.get('analysis', {})
            language = results.get('language', 'unknown')
            
            # Generate suggestions from readability analysis
            if 'readability' in analysis:
                readability = analysis['readability']
                if isinstance(readability, str):
                    if 'consider improving' in readability.lower():
                        suggestions.append({
                            'category': 'Readability',
                            'message': readability,
                            'priority': 'medium'
                        })
                    elif 'good readability' in readability.lower():
                        suggestions.append({
                            'category': 'Readability', 
                            'message': readability,
                            'priority': 'low'
                        })
            
            # Generate suggestions from performance analysis
            if 'performance' in analysis:
                performance = analysis['performance']
                if isinstance(performance, str):
                    if 'complexity' in performance.lower() or 'inefficient' in performance.lower():
                        suggestions.append({
                            'category': 'Performance',
                            'message': performance,
                            'priority': 'high' if 'high complexity' in performance.lower() else 'medium'
                        })
                elif isinstance(performance, list):
                    for item in performance:
                        if isinstance(item, str) and 'looks good' not in item.lower():
                            suggestions.append({
                                'category': 'Performance',
                                'message': item,
                                'priority': 'medium'
                            })
            
            # Generate suggestions from security analysis
            if 'security' in analysis:
                security = analysis['security']
                if isinstance(security, str):
                    if 'security issue' in security.lower() or 'vulnerability' in security.lower():
                        priority = 'high' if any(word in security.lower() for word in ['critical', 'severe', 'high']) else 'medium'
                        suggestions.append({
                            'category': 'Security',
                            'message': security,
                            'priority': priority
                        })
                elif isinstance(security, list):
                    for item in security:
                        if isinstance(item, str) and 'no' not in item.lower() and 'passes' not in item.lower():
                            suggestions.append({
                                'category': 'Security',
                                'message': item,
                                'priority': 'high'
                            })
            
            # Generate suggestions from style analysis
            if 'style' in analysis:
                style = analysis['style']
                if isinstance(style, dict) and 'suggestions' in style:
                    for suggestion in style['suggestions'][:3]:  # Limit to top 3
                        if isinstance(suggestion, str):
                            suggestions.append({
                                'category': 'Style',
                                'message': suggestion,
                                'priority': 'low'
                            })
                elif isinstance(style, list):
                    for item in style[:3]:  # Limit to top 3
                        if isinstance(item, str):
                            suggestions.append({
                                'category': 'Style',
                                'message': item,
                                'priority': 'low'
                            })
            
            # Generate suggestions from best practices analysis
            if 'best_practices' in analysis:
                best_practices = analysis['best_practices']
                if isinstance(best_practices, list):
                    for item in best_practices[:2]:  # Limit to top 2
                        if isinstance(item, str) and 'no' not in item.lower():
                            suggestions.append({
                                'category': 'Best Practices',
                                'message': item,
                                'priority': 'medium'
                            })
            
            # If no specific suggestions generated, create a general one
            if not suggestions:
                suggestions.append({
                    'category': 'Overall',
                    'message': f"Code analysis looks good for {language.title()}!",
                    'priority': 'info'
                })
            
            return suggestions
            
        except Exception as e:
            # Return error suggestion if generation fails
            return [{
                'category': 'System',
                'message': f'Analysis completed with limitations: {str(e)}',
                'priority': 'info'
            }]
    
    def _analyze_python_comprehensive(self, code, results, use_llm=False, llm_integration=None):
        """Comprehensive Python analysis using all available analyzers."""
        try:
            # Use specialized Python analyzer
            python_results = analyze_python(code)
            
            # Extract and standardize results
            results['analysis'] = {
                'readability': self._extract_readability_analysis(python_results),
                'performance': self._extract_performance_analysis(python_results),
                'security': self._extract_security_analysis(python_results),
                'style': python_results.get('style', {}),
                'best_practices': python_results.get('best_practices', [])
            }
            
            # Generate summary
            if 'summary' in python_results:
                results['summary'] = python_results['summary']
            else:
                results['summary'] = self._generate_summary(results['analysis'], 'python')
            
            # Extract metrics
            results['metrics'] = python_results.get('metrics', self._calculate_general_metrics(code))
            
            return results
            
        except Exception as e:
            # Fallback to universal analyzer
            return self._analyze_universal(code, results, 'python')
    
    def _analyze_c_comprehensive(self, code, results, language):
        """Comprehensive C/C++ analysis."""
        try:
            if language.lower() in ['cpp', 'c++']:
                c_results = analyze_cpp_code(code)
            else:
                c_results = analyze_c_code(code)
            
            # Standardize C analysis results
            results['analysis'] = {
                'readability': c_results.get('readability', 'C code readability analysis completed'),
                'performance': c_results.get('performance', 'C performance analysis completed'),
                'security': c_results.get('security', 'C security analysis completed')
            }
            
            results['summary'] = c_results.get('summary', {'language': language, 'status': 'analyzed'})
            results['metrics'] = c_results.get('metrics', self._calculate_general_metrics(code))
            
            return results
            
        except Exception as e:
            return self._analyze_universal(code, results, language)
    
    def _analyze_javascript_comprehensive(self, code, results):
        """Comprehensive JavaScript analysis."""
        try:
            js_results = analyze_javascript_code(code)
            
            results['analysis'] = {
                'readability': js_results.get('readability', 'JavaScript readability analysis completed'),
                'performance': js_results.get('performance', 'JavaScript performance analysis completed'),
                'security': js_results.get('security', 'JavaScript security analysis completed')
            }
            
            results['summary'] = js_results.get('summary', {'language': 'javascript', 'status': 'analyzed'})
            results['metrics'] = js_results.get('metrics', self._calculate_general_metrics(code))
            
            return results
            
        except Exception as e:
            return self._analyze_universal(code, results, 'javascript')
    
    def _analyze_java_comprehensive(self, code, results):
        """Comprehensive Java analysis."""
        try:
            java_results = analyze_java_code(code)
            
            results['analysis'] = {
                'readability': java_results.get('readability', 'Java readability analysis completed'),
                'performance': java_results.get('performance', 'Java performance analysis completed'),
                'security': java_results.get('security', 'Java security analysis completed')
            }
            
            results['summary'] = java_results.get('summary', {'language': 'java', 'status': 'analyzed'})
            results['metrics'] = java_results.get('metrics', self._calculate_general_metrics(code))
            
            return results
            
        except Exception as e:
            return self._analyze_universal(code, results, 'java')
    
    def _analyze_universal(self, code, results, language):
        """Universal analysis for any language using the universal analyzer."""
        try:
            universal_results = self.universal_analyzer.analyze_code(code, language)
            
            # Standardize universal analyzer results
            results['analysis'] = {
                'readability': f"Generic readability analysis completed for {language}",
                'performance': f"Generic performance analysis completed for {language}",
                'security': f"Generic security analysis completed for {language}"
            }
            
            if 'issues' in universal_results:
                # Convert issues to suggestions
                results['suggestions'] = []
                for issue in universal_results['issues']:
                    if isinstance(issue, dict):
                        results['suggestions'].append({
                            'category': issue.get('type', 'General').title(),
                            'message': issue.get('message', ''),
                            'priority': issue.get('priority', 'medium')
                        })
            
            results['summary'] = universal_results.get('summary', {'language': language, 'status': 'analyzed'})
            results['metrics'] = universal_results.get('metrics', self._calculate_general_metrics(code))
            
            return results
            
        except Exception as e:
            # Final fallback
            results['analysis'] = {
                'error': f'Analysis failed: {str(e)[:100]}',
                'fallback_used': True
            }
            results['summary'] = {'language': language, 'status': 'failed'}
            results['metrics'] = self._calculate_general_metrics(code)
            return results
    
    def _extract_readability_analysis(self, python_results):
        """Extract readability analysis from Python results."""
        if 'readability_score' in python_results:
            score = python_results['readability_score']
            return f"Readability score: {score}/5 - {'Good' if score >= 4 else 'Fair' if score >= 3 else 'Needs improvement'}"
        elif 'style' in python_results and 'suggestions' in python_results['style']:
            suggestions = python_results['style']['suggestions']
            if suggestions and len(suggestions) > 0:
                return f"Readability suggestions available: {len(suggestions)} items to consider"
        return "Readability analysis completed"
    
    def _extract_performance_analysis(self, python_results):
        """Extract performance analysis from Python results."""
        if 'performance' in python_results:
            return python_results['performance']
        elif 'complexity' in python_results:
            complexity = python_results['complexity']
            if isinstance(complexity, dict) and 'rating' in complexity:
                return f"Complexity rating: {complexity['rating']}"
        return "Performance analysis completed"
    
    def _extract_security_analysis(self, python_results):
        """Extract security analysis from Python results."""
        if 'security' in python_results:
            return python_results['security']
        return "Security analysis completed"
    
    def _generate_summary(self, analysis, language):
        """Generate a summary from analysis results."""
        return {
            'language': language,
            'analysis_completed': True,
            'readability_checked': 'readability' in analysis,
            'performance_checked': 'performance' in analysis,
            'security_checked': 'security' in analysis
        }
    
    def _calculate_general_metrics(self, code):
        """Calculate general code metrics that work for any language."""
        lines = code.split('\n')
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'avg_line_length': sum(len(line) for line in lines) / max(len(lines), 1),
            'max_line_length': max(len(line) for line in lines) if lines else 0
        }
    
    def _handle_analysis_error(self, code, language, error, results):
        """Handle analysis errors gracefully."""
        results['analysis'] = {
            'error': str(error)[:200],
            'fallback_used': True
        }
        
        results['summary'] = {
            'error': 'Analysis failed',
            'language': language,
            'overall_score': 0
        }
        
        results['suggestions'] = [{
            'category': 'System',
            'message': f'Analysis completed with limitations: {str(error)[:100]}',
            'priority': 'info'
        }]
        
        results['metrics'] = self._calculate_general_metrics(code)
        
        return results


def analyze_code(code, filename=None, language=None, use_llm=False, llm_integration=None):
    """
    Enhanced convenient function to analyze code with full language support.
    
    Args:
        code (str): Code to analyze
        filename (str, optional): Original filename
        language (str, optional): Force specific language
        use_llm (bool): Whether to enhance with LLM
        llm_integration: Custom LLM integration
    
    Returns:
        dict: Comprehensive analysis results
    """
    analyzer = CodeAnalyzer()
    return analyzer.analyze(code, filename, language, use_llm=use_llm, llm_integration=llm_integration)


# For backward compatibility with existing code
def get_supported_languages():
    """Get list of all supported languages with their support levels."""
    universal_analyzer = UniversalAnalyzer()
    return universal_analyzer.get_supported_languages()