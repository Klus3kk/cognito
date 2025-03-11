import re
from collections import defaultdict, Counter

class CCodeAnalyzer:
    """Analyzer for C code quality, style, and security."""
    
    def __init__(self, code):
        """
        Initialize the C code analyzer.
        
        Args:
            code (str): C code to analyze
        """
        self.code = code
        self.lines = code.split('\n')
        self.issues = []
        self.metrics = {}
        
    def analyze(self):
        """
        Perform comprehensive analysis on the C code.
        
        Returns:
            dict: Analysis results with issues and metrics
        """
        self._analyze_style()
        self._analyze_security()
        self._analyze_complexity()
        self._analyze_memory_safety()
        
        # Calculate maintainability index
        self.metrics['maintainability_index'] = self._calculate_maintainability_index()
        
        return {
            'issues': self.issues,
            'metrics': self.metrics,
            'summary': self._generate_summary()
        }
    
    def _analyze_style(self):
        """Analyze C code style and conventions."""
        # Check line length
        long_lines = []
        for i, line in enumerate(self.lines):
            if len(line) > 80:
                long_lines.append(i + 1)
        
        if long_lines:
            self.issues.append({
                'type': 'style',
                'message': f"Lines exceeding 80 characters found on lines: {', '.join(map(str, long_lines[:5]))}{'...' if len(long_lines) > 5 else ''}",
                'priority': 'low'
            })
        
        # Check naming conventions
        # Look for function definitions
        function_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{')
        functions = function_pattern.findall(self.code)
        
        # Check function naming (snake_case is common in C)
        non_snake_case = []
        for func in functions:
            if not re.match(r'^[a-z][a-z0-9_]*$', func) and not func.startswith('_'):
                non_snake_case.append(func)
        
        if non_snake_case:
            self.issues.append({
                'type': 'style',
                'message': f"Function names not following snake_case convention: {', '.join(non_snake_case[:5])}{'...' if len(non_snake_case) > 5 else ''}",
                'priority': 'low'
            })
        
        # Check indentation consistency
        indent_pattern = re.compile(r'^(\s+)\S')
        indents = [match.group(1) for line in self.lines 
                  for match in [indent_pattern.search(line)] if match]
        
        # Check if there's a mix of tabs and spaces
        has_tabs = any('\t' in indent for indent in indents)
        has_spaces = any(' ' in indent for indent in indents)
        
        if has_tabs and has_spaces:
            self.issues.append({
                'type': 'style',
                'message': "Inconsistent indentation: mix of tabs and spaces used",
                'priority': 'medium'
            })
        
        # Check for consistent spacing around operators
        operator_pattern = re.compile(r'([a-zA-Z0-9_])([\+\-\*\/\=\!\<\>])|\b([\+\-\*\/\=\!\<\>])([a-zA-Z0-9_])')
        operator_issues = operator_pattern.findall(self.code)
        
        if operator_issues:
            self.issues.append({
                'type': 'style',
                'message': "Inconsistent spacing around operators detected",
                'priority': 'low'
            })
        
        # Store style metrics
        self.metrics['style'] = {
            'long_lines': len(long_lines),
            'non_snake_case_functions': len(non_snake_case),
            'has_mixed_indentation': has_tabs and has_spaces
        }
    
    def _analyze_security(self):
        """Analyze C code for common security vulnerabilities."""
        # Check for unsafe functions
        unsafe_functions = {
            'gets': 'Buffer overflow risk. Use fgets instead.',
            'strcpy': 'Buffer overflow risk. Use strncpy with proper bounds checking.',
            'strcat': 'Buffer overflow risk. Use strncat with proper bounds checking.',
            'sprintf': 'Buffer overflow risk. Use snprintf with size limit.',
            'scanf': 'Format string vulnerability risk if user input used directly in format.',
            'system': 'Command injection risk. Avoid or use execve with validated input.'
        }
        
        # Look for unsafe function calls
        for func, risk in unsafe_functions.items():
            pattern = re.compile(r'\b' + func + r'\s*\(')
            matches = pattern.findall(self.code)
            
            if matches:
                self.issues.append({
                    'type': 'security',
                    'message': f"Use of unsafe function '{func}': {risk}",
                    'priority': 'high'
                })
        
        # Check for format string vulnerabilities
        format_func_pattern = re.compile(r'\b(printf|fprintf|sprintf|snprintf)\s*\(([^)]*)\)')
        format_func_matches = format_func_pattern.findall(self.code)
        
        for match in format_func_matches:
            func, args = match
            # Check if there's a variable directly after the format specifier
            if re.search(r'%[^%"]*", [^"]*\b(argv|input|buf|buffer|str|string)\b', args):
                self.issues.append({
                    'type': 'security',
                    'message': f"Potential format string vulnerability in {func}() call with user input",
                    'priority': 'high'
                })
        
        # Check for integer overflow possibilities
        if re.search(r'\bsizeof\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)\s*\*', self.code):
            self.issues.append({
                'type': 'security',
                'message': "Potential integer overflow when calculating buffer sizes with sizeof() * n",
                'priority': 'medium'
            })
    
    def _analyze_complexity(self):
        """Analyze C code complexity."""
        # Count nested control structures
        open_braces = []
        max_nesting = 0
        current_nesting = 0
        
        for line in self.lines:
            # Count opening and closing braces
            open_count = line.count('{')
            close_count = line.count('}')
            
            current_nesting += open_count - close_count
            max_nesting = max(max_nesting, current_nesting)
        
        if max_nesting > 4:
            self.issues.append({
                'type': 'complexity',
                'message': f"High nesting complexity detected (max nesting level: {max_nesting})",
                'priority': 'medium'
            })
        
        # Count control structures
        control_patterns = {
            'if': r'\bif\s*\(',
            'for': r'\bfor\s*\(',
            'while': r'\bwhile\s*\(',
            'switch': r'\bswitch\s*\(',
            'goto': r'\bgoto\s+[a-zA-Z_][a-zA-Z0-9_]*;'
        }
        
        control_counts = {}
        for control, pattern in control_patterns.items():
            control_counts[control] = len(re.findall(pattern, self.code))
        
        # Check for excessive goto statements
        if control_counts.get('goto', 0) > 3:
            self.issues.append({
                'type': 'complexity',
                'message': f"Excessive use of goto statements ({control_counts['goto']} occurrences)",
                'priority': 'medium'
            })
        
        # Calculate McCabe cyclomatic complexity (simplified version)
        total_conditionals = sum(control_counts.values()) - control_counts.get('goto', 0)
        function_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{')
        function_count = len(function_pattern.findall(self.code))
        
        avg_complexity = total_conditionals / max(function_count, 1)
        if avg_complexity > 10:
            self.issues.append({
                'type': 'complexity',
                'message': f"High average cyclomatic complexity ({avg_complexity:.1f})",
                'priority': 'medium'
            })
        
        # Store complexity metrics
        self.metrics['complexity'] = {
            'max_nesting': max_nesting,
            'control_structures': control_counts,
            'cyclomatic_complexity': avg_complexity
        }
    
    def _analyze_memory_safety(self):
        """Analyze C code for memory management issues."""
        # Check for malloc without free
        malloc_count = len(re.findall(r'\bmalloc\s*\(', self.code))
        free_count = len(re.findall(r'\bfree\s*\(', self.code))
        
        if malloc_count > free_count:
            self.issues.append({
                'type': 'memory',
                'message': f"Potential memory leak: {malloc_count} malloc() calls but only {free_count} free() calls",
                'priority': 'high'
            })
        
        # Check for common pointer errors
        dereference_null_pattern = re.compile(r'if\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*!=\s*NULL\s*\)\s*{[^}]*}\s*else\s*{[^}]*\*[a-zA-Z_][a-zA-Z0-9_]*')
        dereference_null = dereference_null_pattern.search(self.code)
        
        if dereference_null:
            self.issues.append({
                'type': 'memory',
                'message': "Potential NULL pointer dereference detected",
                'priority': 'high'
            })
        
        # Check for array bounds issues
        array_access_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*([a-zA-Z_][a-zA-Z0-9_]*|[0-9]+)\s*\]')
        array_accesses = array_access_pattern.findall(self.code)
        
        bounds_check_pattern = re.compile(r'if\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*<\s*([a-zA-Z_][a-zA-Z0-9_]*)')
        bounds_checks = bounds_check_pattern.findall(self.code)
        
        # Convert to sets for comparison
        array_indices = set(index for _, index in array_accesses if not index.isdigit())
        checked_indices = set(index for index, _ in bounds_checks)
        
        unchecked_indices = array_indices - checked_indices
        if unchecked_indices:
            self.issues.append({
                'type': 'memory',
                'message': f"Potential array bounds issues: variables used as indices without bounds checking: {', '.join(unchecked_indices)}",
                'priority': 'high'
            })
        
        # Store memory safety metrics
        self.metrics['memory_safety'] = {
            'malloc_count': malloc_count,
            'free_count': free_count,
            'unchecked_array_indices': len(unchecked_indices)
        }
    
    def _calculate_maintainability_index(self):
        """
        Calculate maintainability index for C code.
        
        Returns:
            float: Maintainability index (0-100 scale)
        """
        import math
        
        # Count lines of code (excluding empty lines and comments)
        loc = len([line for line in self.lines if line.strip() and not line.strip().startswith('//')])
        
        if loc == 0:
            return 100.0  # Empty code is trivially maintainable
        
        # Use complexity metrics from previous analysis
        avg_complexity = self.metrics.get('complexity', {}).get('cyclomatic_complexity', 5)
        
        # Calculate Halstead volume (simplified)
        # Count unique operators and operands
        operators = set()
        for op in ['+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!']:
            if op in self.code:
                operators.add(op)
        
        # Count variables and function names (simplified)
        identifier_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
        identifiers = set(identifier_pattern.findall(self.code))
        
        # Remove C keywords
        keywords = {'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'return', 'int', 'char', 
                    'float', 'double', 'void', 'struct', 'union', 'const', 'static', 'sizeof', 'typedef'}
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
        # MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
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
        
        return {
            'issue_count': len(self.issues),
            'high_priority_issues': priority_counts.get('high', 0),
            'medium_priority_issues': priority_counts.get('medium', 0),
            'low_priority_issues': priority_counts.get('low', 0),
            'maintainability': {
                'index': maintainability_index,
                'rating': maintainability_rating
            },
            'priority_recommendations': recommendations
        }


def analyze_c_code(code):
    """
    Analyze C code for quality, style, and security issues.
    
    Args:
        code (str): C code to analyze
        
    Returns:
        dict: Analysis results
    """
    analyzer = CCodeAnalyzer(code)
    return analyzer.analyze()