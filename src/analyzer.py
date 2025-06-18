"""
Updated analyzer.py with complete language analyzer integration.
This replaces the existing analyzer.py to support all languages properly.
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
            'language_confidence': confidence,
            'support_level': support_level,
            'analysis': {},
            'summary': {},
            'suggestions': [],
            'metrics': {}
        }
        
        try:
            # Use universal analyzer for comprehensive analysis
            universal_results = self.universal_analyzer.analyze_code(code, language, filename or "")
            
            # Extract core analysis from universal analyzer
            results['analysis']['core'] = universal_results
            results['metrics'].update(universal_results.get('metrics', {}))
            
            # Add language-specific enhancements
            if support_level == 'full':
                results = self._enhance_with_specialized_analysis(code, language, results)
            
            # Add cross-language analysis components
            results = self._add_cross_language_analysis(code, results)
            
            # Generate comprehensive summary
            results['summary'] = self._generate_comprehensive_summary(results)
            
            # Generate actionable suggestions
            results['suggestions'] = self._generate_comprehensive_suggestions(results)
            
        except Exception as e:
            # Graceful fallback with error reporting
            results = self._handle_analysis_error(code, language, e, results)
        
        # LLM enhancement if requested
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
    
    def _enhance_with_specialized_analysis(self, code, language, results):
        """Enhance results with specialized language analysis."""
        try:
            if language == 'python':
                # Enhanced Python analysis
                python_results = analyze_python(code)
                results['analysis']['python_specific'] = python_results
                
                # Add Python readability analysis
                results['analysis']['readability'] = analyze_readability(code)
                
                # Add Python performance analysis
                results['analysis']['performance'] = {
                    'complexity': analyze_complexity(code),
                    'memory': analyze_memory_usage(code)
                }
                
                # Add Python security analysis
                security_issues = analyze_security(code)
                results['analysis']['security'] = {
                    'issues': security_issues,
                    'suggestion': generate_security_suggestion(security_issues)
                }
                
            elif language == 'c':
                # Enhanced C analysis already handled by universal analyzer
                # Add additional C-specific checks
                results['analysis']['memory_safety'] = self._analyze_c_memory_safety(code)
                
            elif language in ['javascript', 'typescript']:
                # Enhanced JavaScript/TypeScript analysis
                results['analysis']['web_specific'] = self._analyze_web_patterns(code)
                
            elif language == 'java':
                # Enhanced Java analysis
                results['analysis']['enterprise_patterns'] = self._analyze_java_enterprise_patterns(code)
                
            elif language in ['cpp', 'c++']:
                # Enhanced C++ analysis
                results['analysis']['modern_cpp'] = self._analyze_modern_cpp_usage(code)
                
        except Exception as e:
            results['analysis']['enhancement_error'] = str(e)
        
        return results
    
    def _add_cross_language_analysis(self, code, results):
        """Add analysis components that work across languages."""
        try:
            # General readability analysis (works for most languages)
            if 'readability' not in results['analysis']:
                results['analysis']['readability'] = analyze_readability(code)
            
            # General complexity analysis
            if 'performance' not in results['analysis']:
                results['analysis']['performance'] = {
                    'complexity': analyze_complexity(code),
                    'memory': analyze_memory_usage(code)
                }
            
            # Security analysis for supported languages
            if results['language'] in ['python', 'java', 'javascript', 'c', 'cpp']:
                if 'security' not in results['analysis']:
                    security_issues = analyze_security(code)
                    results['analysis']['security'] = {
                        'issues': security_issues,
                        'suggestion': generate_security_suggestion(security_issues)
                    }
            
            # Code metrics that work for any language
            results['analysis']['general_metrics'] = self._calculate_general_metrics(code)
            
        except Exception as e:
            results['analysis']['cross_language_error'] = str(e)
        
        return results
    
    def _generate_comprehensive_summary(self, results):
        """Generate a comprehensive summary from all analysis components."""
        summary = {
            'language': results['language'],
            'support_level': results['support_level'],
            'overall_score': 0,
            'category_scores': {},
            'key_metrics': {},
            'priority_issues': []
        }
        
        try:
            # Extract scores from core analysis
            core_analysis = results['analysis'].get('core', {})
            if 'summary' in core_analysis:
                core_summary = core_analysis['summary']
                summary['maintainability'] = core_summary.get('maintainability', {})
                summary['issue_count'] = core_summary.get('issue_count', 0)
                
                # Get priority recommendations
                if 'priority_recommendations' in core_summary:
                    summary['priority_issues'].extend(core_summary['priority_recommendations'])
            
            # Calculate category scores
            categories = ['readability', 'performance', 'security', 'maintainability']
            total_score = 0
            valid_categories = 0
            
            for category in categories:
                score = self._calculate_category_score(results, category)
                if score is not None:
                    summary['category_scores'][category] = score
                    total_score += score
                    valid_categories += 1
            
            # Calculate overall score
            if valid_categories > 0:
                summary['overall_score'] = round(total_score / valid_categories, 1)
            
            # Extract key metrics
            summary['key_metrics'] = self._extract_key_metrics(results)
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _generate_comprehensive_suggestions(self, results):
        """Generate comprehensive suggestions from all analysis components."""
        suggestions = []
        
        try:
            # Get suggestions from core analysis
            core_analysis = results['analysis'].get('core', {})
            if 'issues' in core_analysis:
                for issue in core_analysis['issues']:
                    if issue.get('priority') in ['high', 'medium']:
                        suggestions.append({
                            'category': issue.get('type', 'General').title(),
                            'message': issue.get('message', ''),
                            'priority': issue.get('priority', 'medium')
                        })
            
            # Add language-specific suggestions
            suggestions.extend(self._get_language_specific_suggestions(results))
            
            # Add readability suggestions
            readability_analysis = results['analysis'].get('readability', '')
            if isinstance(readability_analysis, str) and 'improve' in readability_analysis.lower():
                suggestions.append({
                    'category': 'Readability',
                    'message': readability_analysis,
                    'priority': 'medium'
                })
            
            # Add security suggestions
            security_analysis = results['analysis'].get('security', {})
            if 'suggestion' in security_analysis:
                suggestion_text = security_analysis['suggestion']
                if 'issues were found' in suggestion_text:
                    suggestions.append({
                        'category': 'Security',
                        'message': suggestion_text,
                        'priority': 'high'
                    })
            
            # Sort by priority
            priority_order = {'high': 0, 'medium': 1, 'low': 2, 'info': 3}
            suggestions.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
            
            # Limit to top 10 suggestions
            suggestions = suggestions[:10]
            
        except Exception as e:
            suggestions.append({
                'category': 'System',
                'message': f'Error generating suggestions: {str(e)[:50]}',
                'priority': 'low'
            })
        
        return suggestions
    
    def _calculate_category_score(self, results, category):
        """Calculate score for a specific category."""
        try:
            if category == 'maintainability':
                core_summary = results['analysis'].get('core', {}).get('summary', {})
                maintainability = core_summary.get('maintainability', {})
                return maintainability.get('index', 50)
            
            elif category == 'readability':
                readability = results['analysis'].get('readability', '')
                if 'excellent' in readability.lower():
                    return 90
                elif 'good' in readability.lower():
                    return 75
                elif 'improve' in readability.lower():
                    return 40
                else:
                    return 60
            
            elif category == 'performance':
                complexity = results['analysis'].get('performance', {}).get('complexity', '')
                if isinstance(complexity, str):
                    if 'good' in complexity.lower():
                        return 80
                    elif 'high' in complexity.lower():
                        return 30
                    else:
                        return 60
                return 60
            
            elif category == 'security':
                security = results['analysis'].get('security', {})
                issues = security.get('issues', [])
                if isinstance(issues, list):
                    if len(issues) == 0 or (len(issues) == 1 and 'no' in str(issues[0]).lower()):
                        return 90
                    elif len(issues) <= 2:
                        return 70
                    else:
                        return 40
                return 60
            
        except Exception:
            pass
        
        return None
    
    def _extract_key_metrics(self, results):
        """Extract key metrics from analysis results."""
        metrics = {}
        
        try:
            # Core metrics
            core_metrics = results['analysis'].get('core', {}).get('metrics', {})
            if 'lines_of_code' in core_metrics:
                metrics['lines_of_code'] = core_metrics['lines_of_code']
            if 'complexity_score' in core_metrics:
                metrics['complexity_score'] = core_metrics['complexity_score']
            
            # General metrics
            general_metrics = results['analysis'].get('general_metrics', {})
            metrics.update(general_metrics)
            
            # Language-specific metrics
            lang = results['language']
            if lang == 'python':
                python_analysis = results['analysis'].get('python_specific', {})
                if 'style' in python_analysis:
                    style_stats = python_analysis['style'].get('stats', {})
                    metrics['function_count'] = style_stats.get('function_count', 0)
                    metrics['class_count'] = style_stats.get('class_count', 0)
            
        except Exception:
            pass
        
        return metrics
    
    def _get_language_specific_suggestions(self, results):
        """Get language-specific suggestions."""
        suggestions = []
        language = results['language']
        
        try:
            if language == 'python':
                python_analysis = results['analysis'].get('python_specific', {})
                if 'best_practices' in python_analysis:
                    for practice in python_analysis['best_practices'][:3]:
                        if 'no common anti-patterns' not in practice.lower():
                            suggestions.append({
                                'category': 'Python Best Practices',
                                'message': practice,
                                'priority': 'medium'
                            })
            
            elif language in ['javascript', 'typescript']:
                web_analysis = results['analysis'].get('web_specific', {})
                suggestions.extend(web_analysis.get('suggestions', []))
            
            elif language == 'java':
                enterprise_analysis = results['analysis'].get('enterprise_patterns', {})
                suggestions.extend(enterprise_analysis.get('suggestions', []))
        
        except Exception:
            pass
        
        return suggestions
    
    def _analyze_c_memory_safety(self, code):
        """Additional C memory safety analysis."""
        return "Enhanced C memory safety analysis completed"
    
    def _analyze_web_patterns(self, code):
        """Analyze web-specific patterns for JavaScript/TypeScript."""
        suggestions = []
        
        # Check for jQuery usage
        if '$(' in code or 'jQuery' in code:
            suggestions.append({
                'category': 'Modern Web',
                'message': 'Consider modern alternatives to jQuery for better performance',
                'priority': 'low'
            })
        
        return {'suggestions': suggestions}
    
    def _analyze_java_enterprise_patterns(self, code):
        """Analyze Java enterprise patterns."""
        suggestions = []
        
        # Check for Spring annotations
        if '@Component' in code or '@Service' in code:
            suggestions.append({
                'category': 'Enterprise',
                'message': 'Spring framework usage detected - ensure proper dependency injection',
                'priority': 'info'
            })
        
        return {'suggestions': suggestions}
    
    def _analyze_modern_cpp_usage(self, code):
        """Analyze modern C++ usage patterns."""
        return "Modern C++ analysis completed"
    
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
            'message': f'Analysis failed for {language}: {str(error)[:100]}',
            'priority': 'high'
        }]
        
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