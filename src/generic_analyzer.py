"""
Generic analyzer that can provide basic analysis for any programming language.
"""

import re
from collections import Counter
from typing import Dict, List, Any

class GenericCodeAnalyzer:
    """Generic code analyzer that works across multiple programming languages."""
    
    def __init__(self, language: str = 'unknown'):
        """Initialize the generic analyzer."""
        self.language = language.lower()
        self.language_config = self._get_language_config()
    
    def _get_language_config(self) -> Dict[str, Any]:
        """Get language-specific configuration for analysis."""
        configs = {
            'python': {
                'comment_patterns': [r'^\s*#.*$'],
                'function_patterns': [r'^\s*def\s+([a-zA-Z_]\w*)\s*\('],
                'class_patterns': [r'^\s*class\s+([A-Z]\w*)\s*[:\(]'],
                'string_patterns': [r'["\'].*?["\']'],
                'line_comment': '#',
                'block_comment': ('"""', '"""'),
                'max_line_length': 79,
                'indentation_size': 4
            },
            'javascript': {
                'comment_patterns': [r'^\s*//.*$', r'/\*.*?\*/'],
                'function_patterns': [r'function\s+([a-zA-Z_$]\w*)\s*\(', r'([a-zA-Z_$]\w*)\s*=\s*\([^)]*\)\s*=>'],
                'class_patterns': [r'^\s*class\s+([A-Z]\w*)\s*[{]'],
                'string_patterns': [r'["\'].*?["\']', r'`.*?`'],
                'line_comment': '//',
                'block_comment': ('/*', '*/'),
                'max_line_length': 80,
                'indentation_size': 2
            },
            'java': {
                'comment_patterns': [r'^\s*//.*$', r'/\*.*?\*/'],
                'function_patterns': [r'^\s*(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_]\w*)\s*\('],
                'class_patterns': [r'^\s*(?:public|private)?\s*class\s+([A-Z]\w*)\s*[{]'],
                'string_patterns': [r'".*?"'],
                'line_comment': '//',
                'block_comment': ('/*', '*/'),
                'max_line_length': 120,
                'indentation_size': 4
            },
            'c': {
                'comment_patterns': [r'^\s*//.*$', r'/\*.*?\*/'],
                'function_patterns': [r'^\s*\w+\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*{'],
                'class_patterns': [],  # C doesn't have classes
                'string_patterns': [r'".*?"'],
                'line_comment': '//',
                'block_comment': ('/*', '*/'),
                'max_line_length': 80,
                'indentation_size': 4
            },
            'cpp': {
                'comment_patterns': [r'^\s*//.*$', r'/\*.*?\*/'],
                'function_patterns': [r'^\s*(?:\w+\s+)*([a-zA-Z_]\w*)\s*\([^)]*\)\s*{'],
                'class_patterns': [r'^\s*class\s+([A-Z]\w*)\s*[{:]'],
                'string_patterns': [r'".*?"'],
                'line_comment': '//',
                'block_comment': ('/*', '*/'),
                'max_line_length': 80,
                'indentation_size': 4
            },
            'go': {
                'comment_patterns': [r'^\s*//.*$', r'/\*.*?\*/'],
                'function_patterns': [r'^\s*func\s+([a-zA-Z_]\w*)\s*\('],
                'class_patterns': [r'^\s*type\s+([A-Z]\w*)\s+struct'],
                'string_patterns': [r'".*?"', r'`.*?`'],
                'line_comment': '//',
                'block_comment': ('/*', '*/'),
                'max_line_length': 100,
                'indentation_size': 4
            },
            'rust': {
                'comment_patterns': [r'^\s*//.*$', r'/\*.*?\*/'],
                'function_patterns': [r'^\s*fn\s+([a-zA-Z_]\w*)\s*\('],
                'class_patterns': [r'^\s*struct\s+([A-Z]\w*)\s*[{]'],
                'string_patterns': [r'".*?"'],
                'line_comment': '//',
                'block_comment': ('/*', '*/'),
                'max_line_length': 100,
                'indentation_size': 4
            }
        }
        
        # Return specific config or generic config
        return configs.get(self.language, {
            'comment_patterns': [r'^\s*//.*$', r'^\s*#.*$', r'/\*.*?\*/'],
            'function_patterns': [r'^\s*(?:function|def|fn|func)\s+([a-zA-Z_]\w*)\s*\('],
            'class_patterns': [r'^\s*(?:class|struct)\s+([A-Z]\w*)\s*[{:]'],
            'string_patterns': [r'["\'].*?["\']'],
            'line_comment': '//',
            'block_comment': ('/*', '*/'),
            'max_line_length': 80,
            'indentation_size': 4
        })
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """
        Perform generic code analysis.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        lines = code.split('\n')
        
        analysis = {
            'language': self.language,
            'metrics': self._calculate_metrics(code, lines),
            'style_issues': self._analyze_style(code, lines),
            'complexity': self._analyze_complexity(code, lines),
            'maintainability': self._analyze_maintainability(code, lines),
            'suggestions': []
        }
        
        # Generate suggestions based on analysis
        analysis['suggestions'] = self._generate_suggestions(analysis)
        
        return analysis
    
    def _calculate_metrics(self, code: str, lines: List[str]) -> Dict[str, Any]:
        """Calculate basic code metrics."""
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = []
        
        # Count comment lines
        for pattern in self.language_config.get('comment_patterns', []):
            comment_lines.extend([line for line in lines if re.search(pattern, line)])
        
        # Extract functions and classes
        functions = []
        for pattern in self.language_config.get('function_patterns', []):
            functions.extend(re.findall(pattern, code, re.MULTILINE))
        
        classes = []
        for pattern in self.language_config.get('class_patterns', []):
            classes.extend(re.findall(pattern, code, re.MULTILINE))
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'comment_lines': len(comment_lines),
            'comment_ratio': len(comment_lines) / max(len(non_empty_lines), 1),
            'function_count': len(functions),
            'class_count': len(classes),
            'avg_line_length': sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1),
            'max_line_length': max(len(line) for line in lines) if lines else 0,
            'functions': functions,
            'classes': classes
        }
    
    def _analyze_style(self, code: str, lines: List[str]) -> List[str]:
        """Analyze code style issues."""
        issues = []
        max_line_length = self.language_config.get('max_line_length', 80)
        expected_indent = self.language_config.get('indentation_size', 4)
        
        # Check line length
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > max_line_length]
        if long_lines:
            issues.append(f"Lines exceed {max_line_length} characters: {len(long_lines)} lines")
        
        # Check indentation consistency
        indentations = []
        for line in lines:
            if line.strip():  # Only check non-empty lines
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 0:
                    indentations.append(leading_spaces)
        
        if indentations:
            # Check if indentations are consistent with expected size
            inconsistent = [indent for indent in indentations if indent % expected_indent != 0]
            if len(inconsistent) > len(indentations) * 0.3:  # More than 30% inconsistent
                issues.append(f"Inconsistent indentation detected (expected {expected_indent} spaces)")
        
        # Check for mixed tabs and spaces
        has_tabs = any('\t' in line for line in lines)
        has_spaces = any('    ' in line for line in lines)
        if has_tabs and has_spaces:
            issues.append("Mixed tabs and spaces detected")
        
        return issues
    
    def _analyze_complexity(self, code: str, lines: List[str]) -> Dict[str, Any]:
        """Analyze code complexity."""
        # Count control structures (generic patterns)
        control_patterns = {
            'if_statements': [r'\bif\b', r'\belif\b', r'\belse\s+if\b'],
            'loops': [r'\bfor\b', r'\bwhile\b', r'\bdo\b'],
            'switches': [r'\bswitch\b', r'\bmatch\b', r'\bcase\b'],
            'try_catch': [r'\btry\b', r'\bcatch\b', r'\bexcept\b']
        }
        
        complexity_metrics = {}
        total_complexity = 1  # Base complexity
        
        for category, patterns in control_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, code, re.IGNORECASE))
            complexity_metrics[category] = count
            total_complexity += count
        
        # Calculate nesting depth
        max_nesting = self._calculate_max_nesting(lines)
        
        # Calculate cyclomatic complexity (simplified)
        cyclomatic_complexity = total_complexity
        
        return {
            'cyclomatic_complexity': cyclomatic_complexity,
            'max_nesting_depth': max_nesting,
            'control_structures': complexity_metrics,
            'complexity_rating': self._rate_complexity(cyclomatic_complexity, max_nesting)
        }
    
    def _calculate_max_nesting(self, lines: List[str]) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        current_depth = 0
        
        # Simple brace/indentation counting
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Count opening braces or control keywords
            if '{' in line:
                current_depth += line.count('{')
            elif any(keyword in stripped for keyword in ['if', 'for', 'while', 'def', 'function', 'class']):
                if stripped.endswith(':') or stripped.endswith('{'):
                    current_depth += 1
            
            max_depth = max(max_depth, current_depth)
            
            # Count closing braces
            if '}' in line:
                current_depth -= line.count('}')
                current_depth = max(0, current_depth)
        
        return max_depth
    
    def _rate_complexity(self, cyclomatic: int, nesting: int) -> str:
        """Rate complexity as low, medium, or high."""
        if cyclomatic <= 5 and nesting <= 3:
            return "Low"
        elif cyclomatic <= 10 and nesting <= 5:
            return "Medium"
        else:
            return "High"
    
    def _analyze_maintainability(self, code: str, lines: List[str]) -> Dict[str, Any]:
        """Analyze code maintainability factors."""
        metrics = self._calculate_metrics(code, lines)
        complexity = self._analyze_complexity(code, lines)
        
        # Calculate maintainability score (0-100)
        score = 100
        
        # Penalize for low comment ratio
        if metrics['comment_ratio'] < 0.1:
            score -= 20
        
        # Penalize for high complexity
        if complexity['cyclomatic_complexity'] > 10:
            score -= 30
        elif complexity['cyclomatic_complexity'] > 5:
            score -= 15
        
        # Penalize for long functions (estimate)
        avg_function_length = metrics['non_empty_lines'] / max(metrics['function_count'], 1)
        if avg_function_length > 50:
            score -= 20
        elif avg_function_length > 25:
            score -= 10
        
        # Penalize for long lines
        if metrics['max_line_length'] > 120:
            score -= 15
        elif metrics['max_line_length'] > 100:
            score -= 10
        
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'rating': self._rate_maintainability(score),
            'factors': {
                'comment_ratio': metrics['comment_ratio'],
                'complexity': complexity['cyclomatic_complexity'],
                'avg_function_length': avg_function_length,
                'max_line_length': metrics['max_line_length']
            }
        }
    
    def _rate_maintainability(self, score: int) -> str:
        """Rate maintainability based on score."""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate suggestions based on analysis results."""
        suggestions = []
        
        # Style suggestions
        for issue in analysis['style_issues']:
            suggestions.append({
                'category': 'Style',
                'message': issue,
                'priority': 'medium'
            })
        
        # Complexity suggestions
        complexity = analysis['complexity']
        if complexity['complexity_rating'] == 'High':
            suggestions.append({
                'category': 'Complexity',
                'message': f"High code complexity detected (cyclomatic: {complexity['cyclomatic_complexity']}). Consider breaking down functions.",
                'priority': 'high'
            })
        
        if complexity['max_nesting_depth'] > 4:
            suggestions.append({
                'category': 'Complexity',
                'message': f"Deep nesting detected (depth: {complexity['max_nesting_depth']}). Consider refactoring to reduce nesting.",
                'priority': 'medium'
            })
        
        # Maintainability suggestions
        maintainability = analysis['maintainability']
        if maintainability['rating'] in ['Poor', 'Fair']:
            factors = maintainability['factors']
            
            if factors['comment_ratio'] < 0.1:
                suggestions.append({
                    'category': 'Documentation',
                    'message': f"Low comment ratio ({factors['comment_ratio']:.1%}). Consider adding more documentation.",
                    'priority': 'medium'
                })
            
            if factors['avg_function_length'] > 25:
                suggestions.append({
                    'category': 'Structure',
                    'message': f"Functions are quite long (avg: {factors['avg_function_length']:.1f} lines). Consider breaking them down.",
                    'priority': 'medium'
                })
        
        # General suggestions
        metrics = analysis['metrics']
        if metrics['function_count'] == 0 and metrics['non_empty_lines'] > 10:
            suggestions.append({
                'category': 'Structure',
                'message': "No functions detected. Consider organizing code into functions for better structure.",
                'priority': 'low'
            })
        
        # If no issues found
        if not suggestions:
            suggestions.append({
                'category': 'Overall',
                'message': f"Code analysis looks good for {self.language.title()}!",
                'priority': 'info'
            })
        
        return suggestions


def analyze_generic_code(code: str, language: str = 'unknown') -> Dict[str, Any]:
    """
    Analyze code using generic analyzer.
    
    Args:
        code: Source code to analyze
        language: Programming language (if known)
        
    Returns:
        Analysis results dictionary
    """
    analyzer = GenericCodeAnalyzer(language)
    return analyzer.analyze(code)


def get_language_support_info() -> Dict[str, List[str]]:
    """
    Get information about language support levels.
    
    Returns:
        Dictionary mapping support levels to language lists
    """
    return {
        'full_support': ['python', 'c'],  # Languages with dedicated analyzers
        'enhanced_support': ['javascript', 'java', 'cpp', 'go', 'rust'],  # Languages with detailed configs
        'basic_support': ['csharp', 'php', 'ruby'],  # Languages with basic configs
        'generic_support': ['*']  # Any language using generic patterns
    }