"""
JavaScript/TypeScript code analyzer for comprehensive code analysis.
"""

import re
import json
from typing import List, Dict, Any
from collections import defaultdict

class JavaScriptAnalyzer:
    """Analyzer for JavaScript/TypeScript code quality, style, and best practices."""
    
    def __init__(self, code):
        """
        Initialize the JavaScript code analyzer.
        
        Args:
            code (str): JavaScript/TypeScript code to analyze
        """
        self.code = code
        self.lines = code.split('\n')
        self.issues = []
        self.metrics = {}
        self.is_typescript = self._detect_typescript()
        
    def _detect_typescript(self):
        """Detect if the code is TypeScript based on type annotations."""
        typescript_patterns = [
            r':\s*(string|number|boolean|any|void|object)',
            r'interface\s+\w+',
            r'type\s+\w+\s*=',
            r'<[A-Z][a-zA-Z]*>',
            r'as\s+\w+',
            r'implements\s+\w+',
            r'enum\s+\w+'
        ]
        
        for pattern in typescript_patterns:
            if re.search(pattern, self.code):
                return True
        return False
        
    def analyze(self):
        """
        Perform comprehensive analysis on the JavaScript/TypeScript code.
        
        Returns:
            dict: Analysis results with issues and metrics
        """
        self._analyze_style()
        self._analyze_best_practices()
        self._analyze_security()
        self._analyze_complexity()
        self._analyze_performance()
        
        # Calculate maintainability index
        self.metrics['maintainability_index'] = self._calculate_maintainability_index()
        
        return {
            'issues': self.issues,
            'metrics': self.metrics,
            'summary': self._generate_summary(),
            'language': 'typescript' if self.is_typescript else 'javascript'
        }
    
    def _analyze_style(self):
        """Analyze JavaScript/TypeScript code style and conventions."""
        # Check line length
        long_lines = []
        for i, line in enumerate(self.lines):
            if len(line) > 100:  # JavaScript common limit
                long_lines.append(i + 1)
        
        if long_lines:
            self.issues.append({
                'type': 'style',
                'message': f"Lines exceeding 100 characters found on lines: {', '.join(map(str, long_lines[:5]))}{'...' if len(long_lines) > 5 else ''}",
                'priority': 'low'
            })
        
        # Check indentation consistency
        indent_pattern = re.compile(r'^(\s+)\S')
        indents = [match.group(1) for line in self.lines 
                  for match in [indent_pattern.search(line)] if match]
        
        has_tabs = any('\t' in indent for indent in indents)
        has_spaces = any(' ' in indent for indent in indents)
        
        if has_tabs and has_spaces:
            self.issues.append({
                'type': 'style',
                'message': "Inconsistent indentation: mix of tabs and spaces used",
                'priority': 'medium'
            })
        
        # Check for consistent spacing
        spacing_issues = []
        if re.search(r'[a-zA-Z0-9_]\{', self.code):
            spacing_issues.append("Missing space before opening brace")
        if re.search(r'[,;][a-zA-Z0-9_]', self.code):
            spacing_issues.append("Missing space after comma/semicolon")
        
        if spacing_issues:
            self.issues.append({
                'type': 'style',
                'message': f"Spacing issues: {', '.join(spacing_issues)}",
                'priority': 'low'
            })
        
        # Check naming conventions
        # camelCase for variables and functions
        var_pattern = re.compile(r'\b(?:var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)')
        variables = var_pattern.findall(self.code)
        
        non_camel_case = []
        for var in variables:
            if not re.match(r'^[a-z$_][a-zA-Z0-9$_]*$', var) and not var.isupper():
                non_camel_case.append(var)
        
        if non_camel_case:
            self.issues.append({
                'type': 'style',
                'message': f"Variables not following camelCase convention: {', '.join(non_camel_case[:5])}{'...' if len(non_camel_case) > 5 else ''}",
                'priority': 'low'
            })
        
        # Store style metrics
        self.metrics['style'] = {
            'long_lines': len(long_lines),
            'mixed_indentation': has_tabs and has_spaces,
            'non_camel_case_vars': len(non_camel_case)
        }
    
    def _analyze_best_practices(self):
        """Analyze JavaScript/TypeScript best practices."""
        # Check for var usage (should use let/const)
        var_usage = len(re.findall(r'\bvar\s+', self.code))
        if var_usage > 0:
            self.issues.append({
                'type': 'best_practice',
                'message': f"Found {var_usage} 'var' declarations. Use 'let' or 'const' instead for block scoping",
                'priority': 'medium'
            })
        
        # Check for == vs ===
        loose_equality = len(re.findall(r'[^=!]==[^=]|[^!]!=[^=]', self.code))
        if loose_equality > 0:
            self.issues.append({
                'type': 'best_practice',
                'message': "Use strict equality (=== and !==) instead of loose equality (== and !=)",
                'priority': 'medium'
            })
        
        # Check for console.log in production code
        console_logs = len(re.findall(r'console\.log\s*\(', self.code))
        if console_logs > 0:
            self.issues.append({
                'type': 'best_practice',
                'message': f"Found {console_logs} console.log statements. Remove for production code",
                'priority': 'low'
            })
        
        # Check for proper error handling
        try_blocks = len(re.findall(r'\btry\s*\{', self.code))
        catch_blocks = len(re.findall(r'\bcatch\s*\(', self.code))
        
        if try_blocks != catch_blocks:
            self.issues.append({
                'type': 'best_practice',
                'message': "Unmatched try/catch blocks detected",
                'priority': 'high'
            })
        
        # Check for proper async/await usage
        async_without_await = False
        async_functions = re.findall(r'async\s+(?:function\s+)?[a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*\)\s*\{([^}]*)\}', self.code, re.DOTALL)
        for func_body in async_functions:
            if 'await' not in func_body and 'return' in func_body:
                async_without_await = True
                break
        
        if async_without_await:
            self.issues.append({
                'type': 'best_practice',
                'message': "Async function detected without await usage. Consider if async is necessary",
                'priority': 'medium'
            })
        
        # Store best practices metrics
        self.metrics['best_practices'] = {
            'var_usage': var_usage,
            'loose_equality': loose_equality,
            'console_logs': console_logs,
            'try_catch_mismatch': try_blocks != catch_blocks
        }
    
    def _analyze_security(self):
        """Analyze JavaScript/TypeScript code for security vulnerabilities."""
        # Check for eval usage
        eval_usage = len(re.findall(r'\beval\s*\(', self.code))
        if eval_usage > 0:
            self.issues.append({
                'type': 'security',
                'message': "Use of eval() detected - code injection risk. Consider safer alternatives",
                'priority': 'high'
            })
        
        # Check for innerHTML usage (XSS risk)
        inner_html = len(re.findall(r'\.innerHTML\s*=', self.code))
        if inner_html > 0:
            self.issues.append({
                'type': 'security',
                'message': "Use of innerHTML detected - XSS risk. Use textContent or sanitize input",
                'priority': 'high'
            })
        
        # Check for document.write usage
        document_write = len(re.findall(r'document\.write\s*\(', self.code))
        if document_write > 0:
            self.issues.append({
                'type': 'security',
                'message': "Use of document.write() detected - XSS risk and performance issue",
                'priority': 'medium'
            })
        
        # Check for setTimeout/setInterval with string argument
        timer_string = len(re.findall(r'set(?:Timeout|Interval)\s*\(\s*["\']', self.code))
        if timer_string > 0:
            self.issues.append({
                'type': 'security',
                'message': "setTimeout/setInterval with string argument - code injection risk. Use function instead",
                'priority': 'medium'
            })
        
        # Check for crypto usage without proper randomness
        weak_random = len(re.findall(r'Math\.random\s*\(\s*\)', self.code))
        if weak_random > 0 and ('password' in self.code.lower() or 'token' in self.code.lower()):
            self.issues.append({
                'type': 'security',
                'message': "Math.random() used in security context - use crypto.getRandomValues() for cryptographic randomness",
                'priority': 'high'
            })
        
        # Store security metrics
        self.metrics['security'] = {
            'eval_usage': eval_usage,
            'inner_html_usage': inner_html,
            'document_write_usage': document_write,
            'weak_random_usage': weak_random
        }
    
    def _analyze_complexity(self):
        """Analyze JavaScript/TypeScript code complexity."""
        # Count nested functions/blocks
        brace_nesting = []
        current_nesting = 0
        max_nesting = 0
        
        for char in self.code:
            if char == '{':
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == '}':
                current_nesting = max(0, current_nesting - 1)
        
        if max_nesting > 5:
            self.issues.append({
                'type': 'complexity',
                'message': f"High nesting complexity detected (max nesting level: {max_nesting})",
                'priority': 'medium'
            })
        
        # Count control structures for cyclomatic complexity
        control_patterns = {
            'if': r'\bif\s*\(',
            'for': r'\bfor\s*\(',
            'while': r'\bwhile\s*\(',
            'switch': r'\bswitch\s*\(',
            'case': r'\bcase\s+',
            'catch': r'\bcatch\s*\(',
            'ternary': r'\?\s*.*?\s*:'
        }
        
        control_counts = {}
        for control, pattern in control_patterns.items():
            control_counts[control] = len(re.findall(pattern, self.code))
        
        # Calculate cyclomatic complexity
        total_complexity = sum(control_counts.values()) + 1  # +1 for base complexity
        
        # Count functions to get average complexity
        function_pattern = re.compile(r'function\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(|[a-zA-Z_$][a-zA-Z0-9_$]*\s*:\s*function\s*\(|[a-zA-Z_$][a-zA-Z0-9_$]*\s*=>\s*\{')
        function_count = len(function_pattern.findall(self.code))
        
        avg_complexity = total_complexity / max(function_count, 1)
        
        if avg_complexity > 10:
            self.issues.append({
                'type': 'complexity',
                'message': f"High average cyclomatic complexity ({avg_complexity:.1f})",
                'priority': 'medium'
            })
        
        # Store complexity metrics
        self.metrics['complexity'] = {
            'max_nesting': max_nesting,
            'cyclomatic_complexity': total_complexity,
            'avg_complexity': avg_complexity,
            'function_count': function_count,
            'control_structures': control_counts
        }
    
    def _analyze_performance(self):
        """Analyze JavaScript/TypeScript code for performance issues."""
        # Check for inefficient loops
        nested_loops = 0
        for_pattern = re.compile(r'\bfor\s*\([^)]*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', re.DOTALL)
        for match in for_pattern.finditer(self.code):
            loop_body = match.group(1)
            if re.search(r'\bfor\s*\(', loop_body):
                nested_loops += 1
        
        if nested_loops > 0:
            self.issues.append({
                'type': 'performance',
                'message': f"Nested loops detected ({nested_loops}) - consider optimization for better performance",
                'priority': 'medium'
            })
        
        # Check for DOM queries in loops
        dom_in_loop_pattern = re.compile(r'for\s*\([^)]*\)[^{]*\{[^}]*(?:document\.querySelector|document\.getElementById|getElementsBy)[^}]*\}', re.DOTALL)
        dom_in_loop = len(dom_in_loop_pattern.findall(self.code))
        
        if dom_in_loop > 0:
            self.issues.append({
                'type': 'performance',
                'message': "DOM queries inside loops detected - cache selectors outside loops",
                'priority': 'medium'
            })
        
        # Check for string concatenation in loops
        string_concat_loop = False
        for_loops = re.findall(r'for\s*\([^)]*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', self.code, re.DOTALL)
        for loop_body in for_loops:
            if re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*\s*\+=\s*["\']', loop_body):
                string_concat_loop = True
                break
        
        if string_concat_loop:
            self.issues.append({
                'type': 'performance',
                'message': "String concatenation in loop detected - use array.join() for better performance",
                'priority': 'medium'
            })
        
        # Store performance metrics
        self.metrics['performance'] = {
            'nested_loops': nested_loops,
            'dom_queries_in_loops': dom_in_loop,
            'string_concat_in_loops': string_concat_loop
        }
    
    def _calculate_maintainability_index(self):
        """
        Calculate maintainability index for JavaScript/TypeScript code.
        
        Returns:
            float: Maintainability index (0-100 scale)
        """
        import math
        
        # Count lines of code (excluding empty lines and comments)
        loc = len([line for line in self.lines 
                  if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('/*')])
        
        if loc == 0:
            return 100.0
        
        # Get complexity from previous analysis
        avg_complexity = self.metrics.get('complexity', {}).get('avg_complexity', 5)
        
        # Calculate Halstead volume (simplified)
        operators = set()
        for op in ['+', '-', '*', '/', '%', '=', '==', '===', '!=', '!==', '>', '<', '>=', '<=', '&&', '||', '!']:
            if op in self.code:
                operators.add(op)
        
        # Count identifiers
        identifier_pattern = re.compile(r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b')
        identifiers = set(identifier_pattern.findall(self.code))
        
        # Remove JavaScript keywords
        keywords = {'var', 'let', 'const', 'function', 'if', 'else', 'for', 'while', 'do', 'switch', 
                   'case', 'break', 'continue', 'return', 'try', 'catch', 'finally', 'throw', 
                   'async', 'await', 'class', 'extends', 'import', 'export', 'default'}
        operands = identifiers - keywords
        
        # Calculate Halstead metrics
        n1 = len(operators)
        n2 = len(operands)
        N1 = sum(self.code.count(op) for op in operators)
        N2 = sum(len(re.findall(r'\b' + re.escape(operand) + r'\b', self.code)) for operand in operands)
        
        # Prevent division by zero
        n1 = max(n1, 1)
        n2 = max(n2, 1)
        N1 = max(N1, 1)
        N2 = max(N2, 1)
        
        # Calculate Halstead volume
        volume = (N1 + N2) * math.log2(n1 + n2)
        
        # Calculate maintainability index
        halstead_volume = 0 if volume <= 0 else math.log(volume)
        maintainability = 171 - 5.2 * halstead_volume - 0.23 * avg_complexity - 16.2 * math.log(loc)
        
        # Normalize to 0-100 scale
        maintainability = max(0, min(100, maintainability))
        
        return round(maintainability, 2)
    
    def _generate_summary(self):
        """
        Generate a summary of the analysis results.
        
        Returns:
            dict: Summary of analysis results
        """
        from collections import Counter
        
        # Count issues by priority
        priority_counts = Counter(issue['priority'] for issue in self.issues)
        
        # Categorize maintainability
        maintainability_index = self.metrics.get('maintainability_index', 50)
        if maintainability_index >= 85:
            maintainability_rating = "Excellent"
        elif maintainability_index >= 65:
            maintainability_rating = "Good"
        elif maintainability_index >= 40:
            maintainability_rating = "Fair"
        else:
            maintainability_rating = "Poor"
        
        # Generate priority recommendations
        priority_issues = sorted([issue for issue in self.issues if issue['priority'] == 'high'], 
                                key=lambda x: x['type'])
        
        recommendations = [issue['message'] for issue in priority_issues[:3]]
        if not recommendations:
            medium_issues = [issue for issue in self.issues if issue['priority'] == 'medium']
            recommendations = [issue['message'] for issue in medium_issues[:3]]
        
        return {
            'issue_count': len(self.issues),
            'high_priority_issues': priority_counts.get('high', 0),
            'medium_priority_issues': priority_counts.get('medium', 0),
            'low_priority_issues': priority_counts.get('low', 0),
            'maintainability': {
                'index': maintainability_index,
                'rating': maintainability_rating
            },
            'language_detected': 'TypeScript' if self.is_typescript else 'JavaScript',
            'priority_recommendations': recommendations
        }


def analyze_javascript_code(code):
    """
    Analyze JavaScript/TypeScript code for quality, style, and best practices.
    
    Args:
        code (str): JavaScript/TypeScript code to analyze
        
    Returns:
        dict: Analysis results
    """
    analyzer = JavaScriptAnalyzer(code)
    return analyzer.analyze()