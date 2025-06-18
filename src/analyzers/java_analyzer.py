"""
Java code analyzer for comprehensive code analysis.
"""

import re
from typing import List, Dict, Any
from collections import defaultdict, Counter

class JavaAnalyzer:
    """Analyzer for Java code quality, style, and best practices."""
    
    def __init__(self, code):
        """
        Initialize the Java code analyzer.
        
        Args:
            code (str): Java code to analyze
        """
        self.code = code
        self.lines = code.split('\n')
        self.issues = []
        self.metrics = {}
        
    def analyze(self):
        """
        Perform comprehensive analysis on the Java code.
        
        Returns:
            dict: Analysis results with issues and metrics
        """
        self._analyze_style()
        self._analyze_best_practices()
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
        """Analyze Java code style and conventions."""
        # Check line length
        long_lines = []
        for i, line in enumerate(self.lines):
            if len(line) > 120:  # Java common limit
                long_lines.append(i + 1)
        
        if long_lines:
            self.issues.append({
                'type': 'style',
                'message': f"Lines exceeding 120 characters found on lines: {', '.join(map(str, long_lines[:5]))}{'...' if len(long_lines) > 5 else ''}",
                'priority': 'low'
            })
        
        # Check class naming convention (PascalCase)
        class_pattern = re.compile(r'(?:public\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)')
        classes = class_pattern.findall(self.code)
        
        non_pascal_case = []
        for class_name in classes:
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                non_pascal_case.append(class_name)
        
        if non_pascal_case:
            self.issues.append({
                'type': 'style',
                'message': f"Class names not following PascalCase convention: {', '.join(non_pascal_case)}",
                'priority': 'medium'
            })
        
        # Check method naming convention (camelCase)
        method_pattern = re.compile(r'(?:public|private|protected|static)*\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{')
        methods = method_pattern.findall(self.code)
        
        non_camel_case = []
        for method_name in methods:
            if not re.match(r'^[a-z][a-zA-Z0-9]*$', method_name) and method_name not in ['main', 'toString', 'equals', 'hashCode']:
                non_camel_case.append(method_name)
        
        if non_camel_case:
            self.issues.append({
                'type': 'style',
                'message': f"Method names not following camelCase convention: {', '.join(non_camel_case[:5])}{'...' if len(non_camel_case) > 5 else ''}",
                'priority': 'low'
            })
        
        # Check constant naming (UPPER_SNAKE_CASE)
        constant_pattern = re.compile(r'(?:public\s+)?(?:static\s+)?(?:final\s+)?\w+\s+([A-Z_][A-Z0-9_]*)\s*=')
        constants = constant_pattern.findall(self.code)
        
        non_constant_case = []
        for constant in constants:
            if not re.match(r'^[A-Z][A-Z0-9_]*$', constant):
                non_constant_case.append(constant)
        
        if non_constant_case:
            self.issues.append({
                'type': 'style',
                'message': f"Constants not following UPPER_SNAKE_CASE convention: {', '.join(non_constant_case)}",
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
        
        # Store style metrics
        self.metrics['style'] = {
            'long_lines': len(long_lines),
            'non_pascal_case_classes': len(non_pascal_case),
            'non_camel_case_methods': len(non_camel_case),
            'mixed_indentation': has_tabs and has_spaces
        }
    
    def _analyze_best_practices(self):
        """Analyze Java best practices."""
        # Check for proper exception handling
        try_without_finally = 0
        resource_without_try_with_resources = 0
        
        # Count try blocks without finally or try-with-resources
        try_blocks = re.findall(r'try\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}(?:\s*catch[^}]*\{[^}]*\})*(?:\s*finally\s*\{[^}]*\})?', self.code, re.DOTALL)
        
        for try_block in try_blocks:
            if 'FileInputStream' in try_block or 'FileOutputStream' in try_block or 'BufferedReader' in try_block:
                if 'new ' in try_block and '(' not in re.search(r'try\s*\(', self.code, re.IGNORECASE) or True:
                    resource_without_try_with_resources += 1
        
        if resource_without_try_with_resources > 0:
            self.issues.append({
                'type': 'best_practice',
                'message': "Consider using try-with-resources for automatic resource management",
                'priority': 'medium'
            })
        
        # Check for missing @Override annotations
        override_methods = len(re.findall(r'@Override\s*\n\s*(?:public|protected|private)', self.code))
        potential_overrides = len(re.findall(r'(?:public|protected)\s+\w+\s+(?:toString|equals|hashCode)\s*\(', self.code))
        
        if potential_overrides > override_methods:
            self.issues.append({
                'type': 'best_practice',
                'message': "Missing @Override annotations on potential override methods",
                'priority': 'low'
            })
        
        # Check for raw types usage
        raw_types = len(re.findall(r'\b(?:List|Set|Map|Collection|ArrayList|HashMap)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=', self.code))
        if raw_types > 0:
            self.issues.append({
                'type': 'best_practice',
                'message': f"Raw types usage detected ({raw_types} occurrences). Use generics for type safety",
                'priority': 'medium'
            })
        
        # Check for proper equals/hashCode implementation
        has_equals = 'equals(' in self.code
        has_hashcode = 'hashCode(' in self.code
        
        if has_equals and not has_hashcode:
            self.issues.append({
                'type': 'best_practice',
                'message': "Class overrides equals() but not hashCode(). This violates the contract",
                'priority': 'high'
            })
        elif has_hashcode and not has_equals:
            self.issues.append({
                'type': 'best_practice',
                'message': "Class overrides hashCode() but not equals(). This violates the contract",
                'priority': 'high'
            })
        
        # Check for System.out.println in production code
        system_out = len(re.findall(r'System\.out\.print', self.code))
        if system_out > 0:
            self.issues.append({
                'type': 'best_practice',
                'message': f"Found {system_out} System.out.print statements. Use logging framework instead",
                'priority': 'low'
            })
        
        # Store best practices metrics
        self.metrics['best_practices'] = {
            'resource_management_issues': resource_without_try_with_resources,
            'missing_override_annotations': potential_overrides - override_methods,
            'raw_types_usage': raw_types,
            'equals_hashcode_violation': (has_equals and not has_hashcode) or (has_hashcode and not has_equals),
            'system_out_usage': system_out
        }
    
    def _analyze_security(self):
        """Analyze Java code for security vulnerabilities."""
        # Check for SQL injection risks
        sql_injection_risk = 0
        sql_patterns = [
            r'Statement.*executeQuery\s*\([^)]*\+',
            r'PreparedStatement.*setString\s*\([^)]*\+',
            r'createStatement\(\).*executeQuery\s*\([^)]*\+'
        ]
        
        for pattern in sql_patterns:
            sql_injection_risk += len(re.findall(pattern, self.code))
        
        if sql_injection_risk > 0:
            self.issues.append({
                'type': 'security',
                'message': "Potential SQL injection vulnerability detected. Use parameterized queries",
                'priority': 'high'
            })
        
        # Check for unsafe deserialization
        unsafe_deserialization = len(re.findall(r'ObjectInputStream.*readObject\s*\(\)', self.code))
        if unsafe_deserialization > 0:
            self.issues.append({
                'type': 'security',
                'message': "Unsafe deserialization detected. Validate input before deserializing",
                'priority': 'high'
            })
        
        # Check for hardcoded credentials
        hardcoded_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'apikey\s*=\s*["\'][^"\']+["\']'
        ]
        
        hardcoded_creds = 0
        for pattern in hardcoded_patterns:
            hardcoded_creds += len(re.findall(pattern, self.code, re.IGNORECASE))
        
        if hardcoded_creds > 0:
            self.issues.append({
                'type': 'security',
                'message': "Hardcoded credentials detected. Use configuration files or environment variables",
                'priority': 'high'
            })
        
        # Check for weak random number generation
        weak_random = len(re.findall(r'Math\.random\s*\(\)', self.code))
        if weak_random > 0 and ('password' in self.code.lower() or 'token' in self.code.lower() or 'key' in self.code.lower()):
            self.issues.append({
                'type': 'security',
                'message': "Math.random() used in security context. Use SecureRandom for cryptographic operations",
                'priority': 'high'
            })
        
        # Check for path traversal vulnerabilities
        path_traversal = len(re.findall(r'new\s+File\s*\([^)]*\+[^)]*\)', self.code))
        if path_traversal > 0:
            self.issues.append({
                'type': 'security',
                'message': "Potential path traversal vulnerability. Validate and sanitize file paths",
                'priority': 'medium'
            })
        
        # Store security metrics
        self.metrics['security'] = {
            'sql_injection_risk': sql_injection_risk,
            'unsafe_deserialization': unsafe_deserialization,
            'hardcoded_credentials': hardcoded_creds,
            'weak_random_usage': weak_random,
            'path_traversal_risk': path_traversal
        }
    
    def _analyze_complexity(self):
        """Analyze Java code complexity."""
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
        
        if max_nesting > 6:  # Java allows deeper nesting due to class structure
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
        
        # Count methods to get average complexity
        method_pattern = re.compile(r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{')
        method_count = len(method_pattern.findall(self.code))
        
        avg_complexity = total_complexity / max(method_count, 1)
        
        if avg_complexity > 10:
            self.issues.append({
                'type': 'complexity',
                'message': f"High average cyclomatic complexity ({avg_complexity:.1f})",
                'priority': 'medium'
            })
        
        # Check for long methods
        method_blocks = re.findall(r'(?:public|private|protected)[^{]*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', self.code, re.DOTALL)
        long_methods = 0
        for method_body in method_blocks:
            lines_count = len([line for line in method_body.split('\n') if line.strip()])
            if lines_count > 50:
                long_methods += 1
        
        if long_methods > 0:
            self.issues.append({
                'type': 'complexity',
                'message': f"Long methods detected ({long_methods}). Consider breaking down into smaller methods",
                'priority': 'medium'
            })
        
        # Store complexity metrics
        self.metrics['complexity'] = {
            'max_nesting': max_nesting,
            'cyclomatic_complexity': total_complexity,
            'avg_complexity': avg_complexity,
            'method_count': method_count,
            'long_methods': long_methods,
            'control_structures': control_counts
        }
    
    def _analyze_oop_principles(self):
        """Analyze adherence to OOP principles."""
        # Check for proper encapsulation
        public_fields = len(re.findall(r'public\s+(?!class|interface|static|final)\w+\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[;=]', self.code))
        if public_fields > 0:
            self.issues.append({
                'type': 'oop',
                'message': f"Public fields detected ({public_fields}). Use private fields with getters/setters for encapsulation",
                'priority': 'medium'
            })
        
        # Check for large classes (violation of Single Responsibility Principle)
        class_sizes = []
        class_blocks = re.findall(r'class\s+[A-Za-z_][A-Za-z0-9_]*[^{]*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', self.code, re.DOTALL)
        
        large_classes = 0
        for class_body in class_blocks:
            lines_count = len([line for line in class_body.split('\n') if line.strip() and not line.strip().startswith('//')])
            class_sizes.append(lines_count)
            if lines_count > 200:
                large_classes += 1
        
        if large_classes > 0:
            self.issues.append({
                'type': 'oop',
                'message': f"Large classes detected ({large_classes}). Consider splitting for better maintainability",
                'priority': 'medium'
            })
        
        # Check for God classes (too many methods)
        method_counts_per_class = []
        for class_body in class_blocks:
            method_count = len(re.findall(r'(?:public|private|protected)\s+(?:static\s+)?\w+\s+\w+\s*\([^)]*\)\s*\{', class_body))
            method_counts_per_class.append(method_count)
        
        god_classes = sum(1 for count in method_counts_per_class if count > 20)
        if god_classes > 0:
            self.issues.append({
                'type': 'oop',
                'message': f"Classes with too many methods detected ({god_classes}). Consider refactoring",
                'priority': 'medium'
            })
        
        # Check for proper inheritance usage
        inheritance_depth = 0
        extends_pattern = re.findall(r'class\s+\w+\s+extends\s+(\w+)', self.code)
        if extends_pattern:
            inheritance_depth = len(extends_pattern)
            if inheritance_depth > 5:
                self.issues.append({
                    'type': 'oop',
                    'message': "Deep inheritance hierarchy detected. Favor composition over inheritance",
                    'priority': 'low'
                })
        
        # Store OOP metrics
        self.metrics['oop'] = {
            'public_fields': public_fields,
            'large_classes': large_classes,
            'god_classes': god_classes,
            'inheritance_depth': inheritance_depth,
            'avg_class_size': sum(class_sizes) / max(len(class_sizes), 1) if class_sizes else 0
        }
    
    def _calculate_maintainability_index(self):
        """
        Calculate maintainability index for Java code.
        
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
        for op in ['+', '-', '*', '/', '%', '=', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^']:
            if op in self.code:
                operators.add(op)
        
        # Count identifiers
        identifier_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
        identifiers = set(identifier_pattern.findall(self.code))
        
        # Remove Java keywords
        keywords = {'public', 'private', 'protected', 'class', 'interface', 'extends', 'implements',
                   'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
                   'return', 'try', 'catch', 'finally', 'throw', 'throws', 'import', 'package',
                   'static', 'final', 'abstract', 'synchronized', 'volatile', 'transient',
                   'int', 'long', 'short', 'byte', 'char', 'float', 'double', 'boolean', 'void'}
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


def analyze_java_code(code):
    """
    Analyze Java code for quality, style, and best practices.
    
    Args:
        code (str): Java code to analyze
        
    Returns:
        dict: Analysis results
    """
    analyzer = JavaAnalyzer(code)
    return analyzer.analyze()