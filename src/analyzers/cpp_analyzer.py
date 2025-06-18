"""
C++ code analyzer for comprehensive code analysis.
"""

import re
from typing import List, Dict, Any
from collections import defaultdict, Counter

class CppAnalyzer:
    """Analyzer for C++ code quality, style, and best practices."""
    
    def __init__(self, code):
        """
        Initialize the C++ code analyzer.
        
        Args:
            code (str): C++ code to analyze
        """
        self.code = code
        self.lines = code.split('\n')
        self.issues = []
        self.metrics = {}
        
    def analyze(self):
        """
        Perform comprehensive analysis on the C++ code.
        
        Returns:
            dict: Analysis results with issues and metrics
        """
        self._analyze_style()
        self._analyze_modern_cpp()
        self._analyze_memory_management()
        self._analyze_security()
        self._analyze_complexity()
        self._analyze_oop_principles()
        
        # Calculate maintainability index
        self.metrics['maintainability_index'] = self._calculate_maintainability_index()
        
        return {
            'issues': self.issues,
            'metrics': self.metrics,
            'summary': self._generate_summary()
        }
    
    def _analyze_style(self):
        """Analyze C++ code style and conventions."""
        # Check line length
        long_lines = []
        for i, line in enumerate(self.lines):
            if len(line) > 100:  # C++ common limit
                long_lines.append(i + 1)
        
        if long_lines:
            self.issues.append({
                'type': 'style',
                'message': f"Lines exceeding 100 characters found on lines: {', '.join(map(str, long_lines[:5]))}{'...' if len(long_lines) > 5 else ''}",
                'priority': 'low'
            })
        
        # Check naming conventions
        # Classes should be PascalCase
        class_pattern = re.compile(r'class\s+([A-Za-z_][A-Za-z0-9_]*)')
        classes = class_pattern.findall(self.code)
        
        non_pascal_case = []
        for class_name in classes:
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                non_pascal_case.append(class_name)
        
        if non_pascal_case:
            self.issues.append({
                'type': 'style',
                'message': f"Class names not following PascalCase convention: {', '.join(non_pascal_case)}",
                'priority': 'low'
            })
        
        # Functions should be snake_case or camelCase
        function_pattern = re.compile(r'(?:inline\s+)?(?:\w+\s+)*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:const\s*)?\{')
        functions = function_pattern.findall(self.code)
        
        inconsistent_naming = []
        for func_name in functions:
            if not (re.match(r'^[a-z][a-z0-9_]*$', func_name) or re.match(r'^[a-z][a-zA-Z0-9]*$', func_name)):
                if func_name not in ['main', 'operator']:
                    inconsistent_naming.append(func_name)
        
        if inconsistent_naming:
            self.issues.append({
                'type': 'style',
                'message': f"Functions not following naming convention: {', '.join(inconsistent_naming[:5])}{'...' if len(inconsistent_naming) > 5 else ''}",
                'priority': 'low'
            })
        
        # Check for consistent brace style
        inconsistent_braces = False
        if '{' in self.code:
            # Check for mixed K&R and Allman styles
            kr_style = len(re.findall(r'\)\s*\{', self.code))
            allman_style = len(re.findall(r'\)\s*\n\s*\{', self.code))
            
            if kr_style > 0 and allman_style > 0:
                inconsistent_braces = True
        
        if inconsistent_braces:
            self.issues.append({
                'type': 'style',
                'message': "Inconsistent brace style detected. Choose either K&R or Allman style consistently",
                'priority': 'low'
            })
        
        # Store style metrics
        self.metrics['style'] = {
            'long_lines': len(long_lines),
            'non_pascal_case_classes': len(non_pascal_case),
            'inconsistent_function_naming': len(inconsistent_naming),
            'inconsistent_braces': inconsistent_braces
        }
    
    def _analyze_modern_cpp(self):
        """Analyze usage of modern C++ features."""
        # Check for C-style casts vs C++ casts
        c_style_casts = len(re.findall(r'\([a-zA-Z_][a-zA-Z0-9_]*\s*\*?\s*\)', self.code))
        cpp_casts = len(re.findall(r'(?:static_cast|dynamic_cast|const_cast|reinterpret_cast)<', self.code))
        
        if c_style_casts > cpp_casts and c_style_casts > 2:
            self.issues.append({
                'type': 'modern_cpp',
                'message': f"C-style casts detected ({c_style_casts}). Use C++ style casts for better type safety",
                'priority': 'medium'
            })
        
        # Check for raw pointers vs smart pointers
        raw_new = len(re.findall(r'\bnew\s+', self.code))
        smart_ptrs = len(re.findall(r'(?:std::)?(?:unique_ptr|shared_ptr|weak_ptr)<', self.code))
        
        if raw_new > smart_ptrs and raw_new > 2:
            self.issues.append({
                'type': 'modern_cpp',
                'message': f"Raw 'new' usage detected ({raw_new}). Consider using smart pointers for automatic memory management",
                'priority': 'medium'
            })
        
        # Check for auto usage
        auto_usage = len(re.findall(r'\bauto\s+[a-zA-Z_]', self.code))
        explicit_types = len(re.findall(r'\b(?:int|double|float|char|bool|string)\s+[a-zA-Z_]', self.code))
        
        if explicit_types > auto_usage * 3 and explicit_types > 5:
            self.issues.append({
                'type': 'modern_cpp',
                'message': "Consider using 'auto' keyword for type deduction where appropriate",
                'priority': 'low'
            })
        
        # Check for range-based for loops
        traditional_for = len(re.findall(r'for\s*\([^:)]*;[^:)]*;[^:)]*\)', self.code))
        range_for = len(re.findall(r'for\s*\([^)]*:\s*[^)]*\)', self.code))
        
        if traditional_for > range_for * 2 and traditional_for > 3:
            self.issues.append({
                'type': 'modern_cpp',
                'message': "Consider using range-based for loops where possible for better readability",
                'priority': 'low'
            })
        
        # Check for nullptr vs NULL
        null_usage = len(re.findall(r'\bNULL\b', self.code))
        nullptr_usage = len(re.findall(r'\bnullptr\b', self.code))
        
        if null_usage > nullptr_usage and null_usage > 0:
            self.issues.append({
                'type': 'modern_cpp',
                'message': f"NULL usage detected ({null_usage}). Use 'nullptr' in C++11 and later",
                'priority': 'medium'
            })
        
        # Store modern C++ metrics
        self.metrics['modern_cpp'] = {
            'c_style_casts': c_style_casts,
            'cpp_style_casts': cpp_casts,
            'raw_new_usage': raw_new,
            'smart_pointer_usage': smart_ptrs,
            'auto_usage': auto_usage,
            'null_vs_nullptr': null_usage
        }
    
    def _analyze_memory_management(self):
        """Analyze C++ memory management patterns."""
        # Check for new/delete pairs
        new_count = len(re.findall(r'\bnew\s+', self.code))
        delete_count = len(re.findall(r'\bdelete\s+', self.code))
        delete_array_count = len(re.findall(r'\bdelete\[\]\s*', self.code))
        
        if new_count > delete_count + delete_array_count:
            self.issues.append({
                'type': 'memory',
                'message': f"Potential memory leak: {new_count} 'new' but only {delete_count + delete_array_count} 'delete' statements",
                'priority': 'high'
            })
        
        # Check for new[] vs delete[] consistency
        new_array = len(re.findall(r'\bnew\s*\[', self.code))
        if new_array > delete_array_count:
            self.issues.append({
                'type': 'memory',
                'message': f"Array allocation mismatch: {new_array} 'new[]' but only {delete_array_count} 'delete[]'",
                'priority': 'high'
            })
        
        # Check for RAII usage
        destructor_count = len(re.findall(r'~[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\)', self.code))
        constructor_count = len(re.findall(r'[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*:', self.code))
        
        # Check for potential double delete
        double_delete_pattern = re.search(r'delete\s+[a-zA-Z_][a-zA-Z0-9_]*\s*;[^}]*delete\s+\1', self.code)
        if double_delete_pattern:
            self.issues.append({
                'type': 'memory',
                'message': "Potential double delete detected",
                'priority': 'high'
            })
        
        # Check for memory leaks in exception scenarios
        if 'throw' in self.code and new_count > 0 and 'try' not in self.code:
            self.issues.append({
                'type': 'memory',
                'message': "Exception handling without proper cleanup may cause memory leaks",
                'priority': 'medium'
            })
        
        # Store memory management metrics
        self.metrics['memory_management'] = {
            'new_count': new_count,
            'delete_count': delete_count,
            'new_array_count': new_array,
            'delete_array_count': delete_array_count,
            'destructor_count': destructor_count
        }
    
    def _analyze_security(self):
        """Analyze C++ code for security vulnerabilities."""
        # Check for buffer overflow risks
        unsafe_functions = {
            'strcpy': 'Buffer overflow risk. Use strncpy or std::string',
            'strcat': 'Buffer overflow risk. Use strncat or std::string',
            'sprintf': 'Buffer overflow risk. Use snprintf',
            'gets': 'Buffer overflow risk. Use fgets',
            'scanf': 'Format string vulnerability. Use safer alternatives'
        }
        
        security_issues = 0
        for func, risk in unsafe_functions.items():
            count = len(re.findall(r'\b' + func + r'\s*\(', self.code))
            if count > 0:
                security_issues += count
                self.issues.append({
                    'type': 'security',
                    'message': f"Use of unsafe function '{func}': {risk}",
                    'priority': 'high'
                })
        
        # Check for integer overflow
        arithmetic_ops = len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*[\+\-\*]\s*[a-zA-Z_][a-zA-Z0-9_]*', self.code))
        overflow_checks = len(re.findall(r'(?:INT_MAX|UINT_MAX|LONG_MAX)', self.code))
        
        if arithmetic_ops > 5 and overflow_checks == 0:
            self.issues.append({
                'type': 'security',
                'message': "Arithmetic operations without overflow checking detected",
                'priority': 'medium'
            })
        
        # Check for uninitialized variables
        var_declarations = re.findall(r'\b(?:int|char|float|double|bool)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;', self.code)
        uninitialized_count = len(var_declarations)
        
        if uninitialized_count > 0:
            self.issues.append({
                'type': 'security',
                'message': f"Uninitialized variables detected ({uninitialized_count}). Initialize all variables",
                'priority': 'medium'
            })
        
        # Store security metrics
        self.metrics['security'] = {
            'unsafe_function_usage': security_issues,
            'uninitialized_variables': uninitialized_count,
            'arithmetic_operations': arithmetic_ops,
            'overflow_checks': overflow_checks
        }
    
    def _analyze_complexity(self):
        """Analyze C++ code complexity."""
        # Count nested blocks
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
        total_complexity = sum(control_counts.values()) + 1
        
        # Count functions to get average complexity
        function_pattern = re.compile(r'(?:inline\s+)?(?:\w+\s+)*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:const\s*)?\{')
        function_count = len(function_pattern.findall(self.code))
        
        avg_complexity = total_complexity / max(function_count, 1)
        
        if avg_complexity > 10:
            self.issues.append({
                'type': 'complexity',
                'message': f"High average cyclomatic complexity ({avg_complexity:.1f})",
                'priority': 'medium'
            })
        
        # Check for template complexity
        template_usage = len(re.findall(r'template\s*<[^>]*>', self.code))
        if template_usage > 5:
            self.issues.append({
                'type': 'complexity',
                'message': f"High template usage ({template_usage}). Consider simplifying template code",
                'priority': 'low'
            })
        
        # Store complexity metrics
        self.metrics['complexity'] = {
            'max_nesting': max_nesting,
            'cyclomatic_complexity': total_complexity,
            'avg_complexity': avg_complexity,
            'function_count': function_count,
            'template_usage': template_usage,
            'control_structures': control_counts
        }
    
    def _analyze_oop_principles(self):
        """Analyze adherence to OOP principles in C++."""
        # Check for proper encapsulation
        public_members = len(re.findall(r'public\s*:[^:]*?(?:int|char|float|double|bool|string)\s+[a-zA-Z_]', self.code, re.DOTALL))
        if public_members > 0:
            self.issues.append({
                'type': 'oop',
                'message': f"Public data members detected ({public_members}). Use private members with accessors",
                'priority': 'medium'
            })
        
        # Check for virtual destructor in base classes
        class_with_virtual = len(re.findall(r'class\s+[A-Za-z_][A-Za-z0-9_]*[^{]*\{[^}]*virtual[^}]*\}', self.code, re.DOTALL))
        virtual_destructor = len(re.findall(r'virtual\s+~[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\)', self.code))
        
        if class_with_virtual > virtual_destructor:
            self.issues.append({
                'type': 'oop',
                'message': "Classes with virtual functions should have virtual destructors",
                'priority': 'high'
            })
        
        # Check for Rule of Three/Five violations
        classes_with_destructor = set(re.findall(r'~([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)', self.code))
        classes_with_copy_constructor = set(re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*const\s+\1\s*&', self.code))
        classes_with_assignment = set(re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*&\s*operator\s*=', self.code))
        
        rule_violations = 0
        for class_name in classes_with_destructor:
            has_copy = class_name in classes_with_copy_constructor
            has_assign = class_name in classes_with_assignment
            
            if not (has_copy and has_assign):
                rule_violations += 1
        
        if rule_violations > 0:
            self.issues.append({
                'type': 'oop',
                'message': f"Rule of Three violations detected ({rule_violations} classes). Implement copy constructor and assignment operator",
                'priority': 'medium'
            })
        
        # Check for const correctness
        const_methods = len(re.findall(r'\)\s*const\s*(?:\{|;)', self.code))
        total_methods = len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(?:const\s*)?(?:\{|;)', self.code))
        
        if total_methods > 0 and const_methods / total_methods < 0.3:
            self.issues.append({
                'type': 'oop',
                'message': "Low const method usage. Consider const correctness for better design",
                'priority': 'low'
            })
        
        # Store OOP metrics
        self.metrics['oop'] = {
            'public_data_members': public_members,
            'virtual_destructor_violations': class_with_virtual - virtual_destructor,
            'rule_of_three_violations': rule_violations,
            'const_method_ratio': const_methods / max(total_methods, 1)
        }
    
    def _calculate_maintainability_index(self):
        """
        Calculate maintainability index for C++ code.
        
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
        for op in ['+', '-', '*', '/', '%', '=', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', 
                  '&', '|', '^', '~', '<<', '>>', '++', '--', '->', '.', '::', 'new', 'delete']:
            if op in self.code:
                operators.add(op)
        
        # Count identifiers
        identifier_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
        identifiers = set(identifier_pattern.findall(self.code))
        
        # Remove C++ keywords
        keywords = {'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
                   'int', 'char', 'float', 'double', 'void', 'bool', 'class', 'struct', 'public', 'private',
                   'protected', 'virtual', 'override', 'namespace', 'using', 'template', 'typename',
                   'const', 'static', 'inline', 'friend', 'operator', 'new', 'delete', 'this',
                   'try', 'catch', 'throw', 'const_cast', 'static_cast', 'dynamic_cast', 'reinterpret_cast'}
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
            'priority_recommendations': recommendations
        }


def analyze_cpp_code(code):
    """
    Analyze C++ code for quality, style, and best practices.
    
    Args:
        code (str): C++ code to analyze
        
    Returns:
        dict: Analysis results
    """
    analyzer = CppAnalyzer(code)
    return analyzer.analyze()